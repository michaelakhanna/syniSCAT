import numpy as np
from config import BOLTZMANN_CONSTANT


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
    update loop. From the user's perspective the behavior is unchanged
    (3D Brownian motion with the correct Stokes-Einstein statistics), but
    the stepwise structure makes it straightforward to insert future
    constraints (e.g., z-motion constraint models or chip/substrate
    exclusion) without another structural rewrite.

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

    # --- Initialize particle positions ---
    # If explicit initial positions are provided, validate and use them.
    if "particle_initial_positions_nm" in params and params["particle_initial_positions_nm"] is not None:
        initial_positions = np.asarray(params["particle_initial_positions_nm"], dtype=float)

        if initial_positions.shape != (num_particles, 3):
            raise ValueError(
                "PARAMS['particle_initial_positions_nm'] must be an array of shape "
                f"({num_particles}, 3) when provided. Got shape {initial_positions.shape}."
            )
    else:
        # Otherwise, sample initial positions uniformly within a plausible volume:
        #   x, y within the field of view, and z within the precomputed z-stack range.
        img_size_nm = float(params["image_size_pixels"]) * float(params["pixel_size_nm"])
        z_range_nm = float(params["z_stack_range_nm"])

        initial_positions = np.empty((num_particles, 3), dtype=float)
        # x and y: [0, img_size_nm)
        initial_positions[:, 0:2] = np.random.rand(num_particles, 2) * img_size_nm
        # z: [-z_range_nm/2, +z_range_nm/2)
        initial_positions[:, 2] = (np.random.rand(num_particles) - 0.5) * z_range_nm

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
        sigma_nm = sigma_m * 1e9  # m -> nm

        # Generate the random walk over time.
        for frame_idx in range(1, num_frames):
            # Draw a 3D Brownian step [dx, dy, dz] in nanometers.
            step_nm = np.random.normal(loc=0.0, scale=sigma_nm, size=3)

            # Previous position at the last frame.
            prev_position_nm = trajectories[i, frame_idx - 1, :]

            # For now, we simulate unconstrained Brownian motion: the proposed
            # step is always accepted. This explicit per-step structure is
            # intentionally chosen so that future z-motion or chip/substrate
            # constraints can be injected at this point without another
            # structural rewrite.
            new_position_nm = prev_position_nm + step_nm

            trajectories[i, frame_idx, :] = new_position_nm

    print("Generated Brownian motion trajectories.")
    return trajectories