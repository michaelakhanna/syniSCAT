# File: trajectory.py
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
    When the chip pattern configuration indicates a solid pattern, lateral
    positions whose projection lies in the solid region are not allowed. The
    behavior is:

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

        - When a nanopillar array is enabled via
          chip_pattern_model == "nanopillars_v1" and
          chip_substrate_preset == "nanopillars", lateral positions whose
          projection lies inside a nanopillar are deterministically mapped
          back into the nearest fluid region just outside the pillar boundary,
          again using project_position_to_fluid_region. This enforces excluded
          volume without introducing trapping or non-random motion.

    Z-axis motion constraint model
    ------------------------------
    The Z-axis behavior is controlled by PARAMS["z_motion_constraint_model"].

    Supported values and their semantics:

        - "unconstrained":
            Fully free 3D Brownian motion in z. The z-coordinate executes a
            standard random walk with no boundaries. This corresponds to the
            original behavior without any surface interaction.

        - "reflective_boundary_v1":
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

        - "surface_interaction_v1" (alias for "reflective_boundary_v1"):
            For backward compatibility with earlier configurations, the legacy
            name "surface_interaction_v1" is accepted and mapped internally
            to the same behavior as "reflective_boundary_v1". No physical
            difference exists between these two options.

        - "surface_interaction_v2":
            Brownian motion in the half-space z <= 0 nm with a perfectly
            reflecting planar boundary at z = 0 nm. This is the mirror image
            of "reflective_boundary_v1": the particle center lives below the
            plane and cannot cross into z > 0. Any Brownian step that would
            place the particle at z > 0 is reflected across the plane.

            Initial z-positions for this model are sampled uniformly from
            [-z_stack_range_nm / 2, 0] when explicit positions are not
            provided, and user-specified initial positions must satisfy
            z <= 0 nm.

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
    z_model_key = str(z_model_raw).strip().lower()

    # Map legacy names and canonical names to a small set of internal modes.
    if z_model_key == "unconstrained":
        z_model = "unconstrained"
    elif z_model_key in ("reflective_boundary_v1", "surface_interaction_v1"):
        # Canonical reflective boundary at z = 0 with z >= 0 half-space.
        z_model = "reflective_boundary_v1"
    elif z_model_key == "surface_interaction_v2":
        # Mirror of reflective_boundary_v1: z <= 0 half-space.
        z_model = "surface_interaction_v2"
    else:
        raise ValueError(
            f"Unsupported z_motion_constraint_model '{z_model_raw}'. "
            "Currently supported values are 'unconstrained', "
            "'reflective_boundary_v1', 'surface_interaction_v1' (alias), "
            "and 'surface_interaction_v2'."
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

    apply_substrate_exclusion = chip_enabled and (
        (
            pattern_model == "gold_holes_v1"
            and substrate_preset in ("default_gold_holes", "lab_default_gold_holes")
        )
        or (
            pattern_model == "nanopillars_v1"
            and substrate_preset == "nanopillars"
        )
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
        # exclusion (x, y) and the Z-surface model.
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

            if z_model == "reflective_boundary_v1" and z_nm < 0.0:
                raise ValueError(
                    "PARAMS['particle_initial_positions_nm'] entry "
                    f"for particle index {i} has z = {z_nm} nm, which is below the "
                    "surface plane z = 0 nm for z_motion_constraint_model "
                    "=='reflective_boundary_v1' (or its alias 'surface_interaction_v1'). "
                    "Initial z-positions for this model must satisfy z >= 0 nm."
                )

            if z_model == "surface_interaction_v2" and z_nm > 0.0:
                raise ValueError(
                    "PARAMS['particle_initial_positions_nm'] entry "
                    f"for particle index {i} has z = {z_nm} nm, which is above the "
                    "surface plane z = 0 nm for z_motion_constraint_model "
                    "=='surface_interaction_v2'. Initial z-positions for this model "
                    "must satisfy z <= 0 nm."
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
                # starting in the fluid region (holes or background between pillars).
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
            elif z_model == "reflective_boundary_v1":
                # Start in the half-space z >= 0. For simplicity and consistency
                # with the iPSF stack, sample z uniformly from [0, z_range_nm / 2).
                initial_positions[i, 2] = float(np.random.rand()) * (z_range_nm / 2.0)
            elif z_model == "surface_interaction_v2":
                # Start in the half-space z <= 0. Mirror of reflective_boundary_v1:
                # sample z uniformly from [-z_stack_range_nm / 2, 0].
                initial_positions[i, 2] = -float(np.random.rand()) * (z_range_nm / 2.0)
            else:
                # This should not be reachable due to earlier validation.
                raise RuntimeError(
                    f"Unexpected z_motion_constraint_model '{z_model_raw}' encountered during initialization."
                )

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
            elif z_model == "reflective_boundary_v1":
                # Reflective boundary at z = 0 nm. If the proposed step would
                # cross into z < 0, reflect it across the plane so the particle
                # remains in the half-space z >= 0.
                z_candidate = prev_z_nm + dz_nm
                if z_candidate >= 0.0:
                    z_nm_new = z_candidate
                else:
                    z_nm_new = -z_candidate
            elif z_model == "surface_interaction_v2":
                # Reflective boundary at z = 0 nm, mirrored relative to v1.
                # If the proposed step would cross into z > 0, reflect it across
                # the plane so the particle remains in the half-space z <= 0.
                z_candidate = prev_z_nm + dz_nm
                if z_candidate <= 0.0:
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


def _random_small_rotation_matrix(rng: np.random.Generator, std_angle_rad: float) -> np.ndarray:
    """
    Generate a random 3D rotation matrix corresponding to a small, isotropic
    rotation drawn from a zero-mean Gaussian distribution on the rotation
    angle.

    The rotation axis is uniformly random on the unit sphere; the rotation
    angle around that axis is drawn from N(0, std_angle_rad^2). For small
    std_angle_rad this produces a Brownian-like angular step.

    Args:
        rng (np.random.Generator): Random number generator.
        std_angle_rad (float): Standard deviation of the rotation angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    std_angle_rad = float(std_angle_rad)
    if std_angle_rad <= 0.0:
        # No rotation: identity.
        return np.eye(3, dtype=float)

    # Sample a random rotation axis uniformly on the unit sphere.
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        # Extremely unlikely; fall back to a fixed axis.
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = vec / norm

    # Sample a small rotation angle from a zero-mean Gaussian.
    angle = rng.normal(loc=0.0, scale=std_angle_rad)

    # Rodrigues' rotation formula.
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c

    R = np.array(
        [
            [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
        ],
        dtype=float,
    )
    return R


def simulate_orientations(params: dict, num_particles: int, num_frames: int) -> np.ndarray | None:
    """
    Simulate rotational Brownian motion (orientation trajectories) for a set
    of particles.

    This function provides the structural counterpart to simulate_trajectories
    for translation: it defines a per-particle, per-frame orientation timebase
    as a sequence of 3x3 rotation matrices. The representation is:

        orientations[i, t] -> 3x3 rotation matrix for particle i at frame t,

    where each matrix maps body-fixed coordinates into the lab frame.

    Current usage and behavior:
        - The main pipeline currently renders only spherical particles, for
          which the PSF is rotationally symmetric and independent of
          orientation. As a result, the renderer does not yet use these
          orientations. This function therefore has no effect on the visual
          output in the current spherical configuration.

        - The orientation trajectories are nevertheless generated and attached
          to ParticleInstance objects so that non-spherical composite particles
          can be added later without changing the core interfaces or motion-
          blur timing logic.

    Configuration:
        - The model is controlled by the following optional PARAMS entries:

            "rotational_diffusion_enabled": bool
                Master switch. If False or absent, this function returns None
                and no orientations are simulated; ParticleInstance objects
                will then carry orientation_matrices=None.

            "rotational_step_std_deg": float
                Standard deviation (in degrees) of the per-frame rotation
                angle around a random axis. Typical values for small, smooth
                rotational Brownian motion are in the range 1–10 degrees.
                Default: 5.0 degrees.

        - The time step for the rotational updates is the frame interval
          dt = 1 / fps, matching the translational integration in
          simulate_trajectories. Motion blur sub-sampling uses interpolation
          between these frame-level orientations via ParticleInstance, which
          will be implemented when rotational appearance matters for
          non-spherical composites.

    RNG and reproducibility:
        - Rotational steps are driven by per-particle NumPy Generators whose
          seeds are drawn from the global np.random RNG. Since the dataset
          generator seeds np.random once per video, translational and
          rotational Brownian motion remain tied to the same per-video seed
          and are fully reproducible under the existing seeding scheme.

    Args:
        params (dict): Simulation parameter dictionary (PARAMS). Must contain
            "fps" when rotational_diffusion_enabled is True.
        num_particles (int): Number of particles being simulated.
        num_frames (int): Number of frames in the video.

    Returns:
        np.ndarray | None:
            - If rotational_diffusion_enabled is False (or not present),
              returns None.
            - Otherwise, returns a numpy array of shape
              (num_particles, num_frames, 3, 3) with dtype float, where each
              [i, t] entry is an SO(3) rotation matrix.
    """
    rotational_enabled = bool(params.get("rotational_diffusion_enabled", False))
    if not rotational_enabled:
        return None

    num_particles = int(num_particles)
    num_frames = int(num_frames)
    if num_particles <= 0 or num_frames <= 0:
        raise ValueError(
            "simulate_orientations requires positive num_particles and num_frames "
            f"(got num_particles={num_particles}, num_frames={num_frames})."
        )

    # Frame interval; kept for potential future use in connecting to a
    # physically derived rotational diffusion coefficient. Currently, we
    # treat rotational_step_std_deg as directly specifying the per-frame
    # angular step scale.
    fps = float(params["fps"])
    if fps <= 0.0:
        raise ValueError("PARAMS['fps'] must be positive when simulating orientations.")
    _dt = 1.0 / fps  # noqa: F841  # reserved for future physics-based refinement

    # Standard deviation of per-frame rotation angle in radians.
    step_std_deg = float(params.get("rotational_step_std_deg", 5.0))
    if step_std_deg < 0.0:
        raise ValueError(
            "PARAMS['rotational_step_std_deg'] must be non-negative if provided."
        )
    step_std_rad = np.deg2rad(step_std_deg)

    # Derive a deterministic set of per-particle seeds from the global
    # np.random RNG. The dataset generator seeds np.random once per video,
    # so drawing seeds here keeps rotational trajectories reproducible under
    # the same per-video seed used for translational trajectories and noise.
    #
    # We restrict seeds to a safe 32-bit range compatible with default_rng.
    particle_seeds_int = np.random.randint(
        0,
        2**31,
        size=num_particles,
        dtype=np.int64,
    )

    # Allocate orientation array and initialize all particles to identity
    # orientation at frame 0.
    orientations = np.zeros((num_particles, num_frames, 3, 3), dtype=float)
    orientations[:, 0, :, :] = np.eye(3, dtype=float)

    # Perform a random walk on SO(3) for each particle using its own Generator.
    for i in range(num_particles):
        rng_i = np.random.default_rng(int(particle_seeds_int[i]))
        for t in range(1, num_frames):
            R_prev = orientations[i, t - 1]
            R_step = _random_small_rotation_matrix(rng_i, step_std_rad)
            # Post-multiply so that R_t maps body frame to lab frame after
            # applying the incremental rotation.
            orientations[i, t] = R_step @ R_prev

    print("Generated rotational Brownian orientation trajectories.")
    return orientations