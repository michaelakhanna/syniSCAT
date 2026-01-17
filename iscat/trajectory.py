import numpy as np
from config import BOLTZMANN_CONSTANT
from chip_pattern import is_position_in_chip_solid, project_position_to_fluid_region


def stokes_einstein_diffusion_coefficient(diameter_nm, temp_K, viscosity_Pa_s):
    """
    Calculate the diffusion coefficient for a spherical particle in a fluid
    using the Stokes-Einstein equation.

    Args:
        diameter_nm (float): Diameter of the particle in nanometers.
        temp_K (float): Absolute temperature of the fluid in Kelvin.
        viscosity_Pa_s (float): Dynamic viscosity of the fluid in Pascal-seconds.

    Returns:
        float: Diffusion coefficient in square meters per second (m^2/s).
    """
    radius_m = diameter_nm * 1e-9 / 2.0
    return (BOLTZMANN_CONSTANT * temp_K) / (6.0 * np.pi * viscosity_Pa_s * radius_m)


def simulate_trajectories(params):
    """
    Simulate 3D Brownian motion trajectories for a set of particles.

    This implementation generates trajectories in a per-particle, per-frame
    update loop. From the user's perspective the behavior is standard
    Stokes–Einstein Brownian motion, with two optional modifications:

        1. Chip/substrate exclusion in the lateral (x, y) directions when a
           solid chip pattern is present.

        2. A configurable Z-axis motion constraint model that can enforce a
           non-penetrable planar surface.

    Lateral (x, y) chip/substrate exclusion
    ---------------------------------------
    When the chip pattern configuration indicates a gold film with circular
    holes, lateral positions whose projection lies in the gold film are not
    allowed. The behavior is:

        - When PARAMS["chip_pattern_enabled"] is False, or
          PARAMS["chip_substrate_preset"] == "empty_background", or
          PARAMS["chip_pattern_model"] == "none", the motion is fully
          unconstrained in x and y.

        - When a gold film with circular holes is enabled via
          chip_pattern_model == "gold_holes_v1" and chip_substrate_preset in
          {"default_gold_holes", "lab_default_gold_holes"}, lateral positions
          whose projection lies in the gold film are deterministically mapped
          back into the nearest fluid region (a hole) after each Brownian step
          using project_position_to_fluid_region. This enforces excluded volume
          without resampling steps.

    Z-axis motion constraint model
    ------------------------------
    The Z-axis behavior is controlled by PARAMS["z_motion_constraint_model"]:

        - "unconstrained":
            Fully free 3D Brownian motion in z. The z-coordinate executes a
            standard random walk with no boundaries. This corresponds to the
            behavior that previously existed.

        - "surface_interaction_v1":
            Brownian motion in the half-space z >= 0 nm with a perfectly
            reflecting planar boundary at z = 0 nm. The interpretation is that
            z = 0 represents the chip/substrate surface, and the particle
            center cannot cross into z < 0 (i.e., it cannot occupy the same
            space as the solid substrate in the axial direction). Any Brownian
            step that would place the particle at z < 0 is reflected across
            the plane.

            Initial z-positions for this model are sampled uniformly from
            [0, z_stack_range_nm / 2) when explicit positions are not
            provided, and user-specified initial positions must satisfy
            z >= 0 nm.

    Args:
        params (dict): Simulation parameter dictionary (PARAMS).

    Returns:
        numpy.ndarray: A 3D array of shape (num_particles, num_frames, 3)
            containing the [x, y, z] coordinates of each particle for each
            frame, in nanometers.
    """
    # --- Basic simulation timing and counts ---
    fps = float(params["fps"])
    duration_seconds = float(params["duration_seconds"])
    num_frames = int(fps * duration_seconds)

    if num_frames <= 0:
        raise ValueError(
            "The product PARAMS['fps'] * PARAMS['duration_seconds'] must be "
            "positive to generate at least one frame."
        )

    dt = 1.0 / fps
    num_particles = int(params["num_particles"])

    # --- Z-motion constraint model selection and validation ---
    z_model_raw = params.get("z_motion_constraint_model", "unconstrained")
    z_model = str(z_model_raw).strip().lower()

    if z_model not in ("unconstrained", "surface_interaction_v1"):
        raise ValueError(
            f"Unsupported z_motion_constraint_model '{z_model_raw}'. "
            "Currently supported values are 'unconstrained' and 'surface_interaction_v1'."
        )

    # --- Chip/substrate exclusion model selection (lateral) ---
    # Determine once per simulation whether we should enforce excluded volume
    # with respect to a solid chip/substrate. This is governed by the chip
    # pattern configuration (Sections 3.2 and 3.6 of the CDD).
    chip_enabled = bool(params.get("chip_pattern_enabled", False))
    pattern_model_raw = params.get("chip_pattern_model", "none")
    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    apply_substrate_exclusion = (
        chip_enabled
        and pattern_model == "gold_holes_v1"
        and substrate_preset in ("default_gold_holes", "lab_default_gold_holes")
    )

    # Field-of-view extents used for sampling initial positions. These are
    # needed regardless of whether substrate exclusion is active.
    img_size_nm = float(params["image_size_pixels"]) * float(params["pixel_size_nm"])
    z_range_nm = float(params["z_stack_range_nm"])

    # --- Initialize particle positions ---
    # If explicit initial positions are provided, validate and use them.
    if "particle_initial_positions_nm" in params and params["particle_initial_positions_nm"] is not None:
        initial_positions = np.asarray(params["particle_initial_positions_nm"], dtype=float)

        if initial_positions.shape != (num_particles, 3):
            raise ValueError(
                "PARAMS['particle_initial_positions_nm'] must be an array of shape "
                f"({num_particles}, 3) when provided. Got shape {initial_positions.shape}."
            )

        # Validate user-provided initial positions against the chip/substrate
        # exclusion (x, y) and the Z-surface model for surface_interaction_v1.
        for i in range(num_particles):
            x_nm = float(initial_positions[i, 0])
            y_nm = float(initial_positions[i, 1])
            z_nm = float(initial_positions[i, 2])

            if apply_substrate_exclusion and is_position_in_chip_solid(params, x_nm, y_nm):
                raise ValueError(
                    "PARAMS['particle_initial_positions_nm'] entry "
                    f"for particle index {i} lies inside a solid chip/substrate "
                    "region according to the current chip pattern configuration. "
                    "Initial positions must be chosen outside solid regions "
                    "when substrate exclusion is active."
                )

            if z_model == "surface_interaction_v1" and z_nm < 0.0:
                raise ValueError(
                    "PARAMS['particle_initial_positions_nm'] entry "
                    f"for particle index {i} has z = {z_nm} nm, which is below the "
                    "surface plane z = 0 nm for z_motion_constraint_model "
                    "=='surface_interaction_v1'. Initial z-positions for this model "
                    "must satisfy z >= 0 nm."
                )

    else:
        # Otherwise, sample initial positions uniformly within a plausible volume:
        #   x, y within the field of view, and z according to the chosen
        #   z-motion model.
        initial_positions = np.empty((num_particles, 3), dtype=float)

        for i in range(num_particles):
            if apply_substrate_exclusion:
                # Sample x, y until we find a position outside any solid region.
                # This implements free Brownian initial positions conditioned on
                # starting in the fluid region (holes).
                max_attempts = 1000
                for _ in range(max_attempts):
                    x_nm = float(np.random.rand() * img_size_nm)
                    y_nm = float(np.random.rand() * img_size_nm)
                    if not is_position_in_chip_solid(params, x_nm, y_nm):
                        initial_positions[i, 0] = x_nm
                        initial_positions[i, 1] = y_nm
                        break
                else:
                    raise RuntimeError(
                        "Failed to sample a valid initial (x, y) position outside the "
                        "solid chip/substrate region after many attempts. Please "
                        "verify the chip pattern geometry parameters."
                    )
            else:
                initial_positions[i, 0:2] = np.random.rand(2) * img_size_nm

            # Initialize z according to the selected z-motion model.
            if z_model == "unconstrained":
                # Symmetric distribution around z = 0, spanning the full Z-stack
                # range as before.
                initial_positions[i, 2] = (float(np.random.rand()) - 0.5) * z_range_nm
            elif z_model == "surface_interaction_v1":
                # Start in the half-space z >= 0. For simplicity and consistency
                # with the iPSF stack, sample z uniformly from [0, z_range_nm / 2).
                initial_positions[i, 2] = float(np.random.rand()) * (z_range_nm / 2.0)

    # --- Allocate trajectory array and set initial positions ---
    trajectories = np.zeros((num_particles, num_frames, 3), dtype=float)
    trajectories[:, 0, :] = initial_positions

    # --- Precompute per-particle diffusion statistics ---
    diameters_nm = params["particle_diameters_nm"]
    if len(diameters_nm) != num_particles:
        raise ValueError(
            "Length of PARAMS['particle_diameters_nm'] "
            f"({len(diameters_nm)}) must match PARAMS['num_particles'] ({num_particles})."
        )

    temp_K = float(params["temperature_K"])
    viscosity_Pa_s = float(params["viscosity_Pa_s"])

    # Loop over particles and generate their trajectories one time step at a time.
    for i in range(num_particles):
        diameter_nm = float(diameters_nm[i])

        # Diffusion coefficient for this particle (m^2/s).
        D_m2_s = stokes_einstein_diffusion_coefficient(
            diameter_nm, temp_K, viscosity_Pa_s
        )

        # Standard deviation of displacement in each Cartesian dimension for
        # one time step, converted to nanometers.
        sigma_m = np.sqrt(2.0 * D_m2_s * dt)
        sigma_nm = float(sigma_m * 1e9)  # m -> nm

        # Generate the random walk over time.
        for frame_idx in range(1, num_frames):
            # Draw a 3D Brownian step [dx, dy, dz] in nanometers.
            step_nm = np.random.normal(loc=0.0, scale=sigma_nm, size=3)

            # Previous position at the last frame.
            prev_position_nm = trajectories[i, frame_idx - 1, :]

            # --- Lateral (x, y) update with optional substrate exclusion ---
            proposed_x_nm = float(prev_position_nm[0] + step_nm[0])
            proposed_y_nm = float(prev_position_nm[1] + step_nm[1])

            if apply_substrate_exclusion:
                x_nm_new, y_nm_new = project_position_to_fluid_region(
                    params,
                    proposed_x_nm,
                    proposed_y_nm,
                )
            else:
                x_nm_new, y_nm_new = proposed_x_nm, proposed_y_nm

            # --- Z-axis update according to the chosen z-motion model ---
            prev_z_nm = float(prev_position_nm[2])
            dz_nm = float(step_nm[2])

            if z_model == "unconstrained":
                z_nm_new = prev_z_nm + dz_nm
            elif z_model == "surface_interaction_v1":
                # Reflective boundary at z = 0 nm. If the proposed step would
                # cross into z < 0, reflect it across the plane.
                z_candidate = prev_z_nm + dz_nm
                if z_candidate >= 0.0:
                    z_nm_new = z_candidate
                else:
                    z_nm_new = -z_candidate
            else:
                # This should not be reachable due to the earlier validation.
                raise RuntimeError(
                    f"Unexpected z_motion_constraint_model '{z_model_raw}' encountered during simulation."
                )

            trajectories[i, frame_idx, 0] = x_nm_new
            trajectories[i, frame_idx, 1] = y_nm_new
            trajectories[i, frame_idx, 2] = z_nm_new

    print("Generated Brownian motion trajectories.")
    return trajectories