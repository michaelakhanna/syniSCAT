import numpy as np
import cv2
from tqdm import tqdm
from scipy.special import j1

from mask_generation import generate_and_save_mask_for_particle
from trackability import TrackabilityModel


def add_noise(frame, params):
    """
    Applies simulated shot (Poisson) and read (Gaussian) noise to an image frame.

    Args:
        frame (numpy.ndarray): The ideal, noise-free image frame.
        params (dict): The main simulation parameter dictionary.

    Returns:
        numpy.ndarray: The noisy image frame.
    """
    noisy_frame = frame.copy()
    if params["shot_noise_enabled"]:
        # Scale the noise component to allow for artistic control.
        ideal_frame = noisy_frame
        full_noisy_frame = np.random.poisson(ideal_frame).astype(float)
        noise_component = full_noisy_frame - ideal_frame
        noisy_frame = ideal_frame + noise_component * params["shot_noise_scaling_factor"]

    if params["gaussian_noise_enabled"]:
        noisy_frame += np.random.normal(scale=params["read_noise_std"], size=frame.shape)

    return noisy_frame


def estimate_psf_padding_radius_pixels(params):
    """
    Estimate the extra padding radius (in oversampled pixels) required around
    the simulated field of view so that circular wrap-around from np.roll does
    not contaminate the central region that is ultimately written to the video.

    The estimate uses an Airy-pattern approximation for the PSF of a circular
    aperture. We compute the normalized intensity

        I_rel(r) = I(r) / I(0) ~= [2 J1(pi * rho) / (pi * rho)]^2,

    where rho = (NA * r) / lambda_medium is a dimensionless radial coordinate.

    We then find the largest radius r such that I_rel(r) is still above a
    user-controllable fraction:

        I_rel(r) >= psf_intensity_fraction_threshold,

    and treat everything beyond that radius as negligible. The corresponding
    physical radius is converted into oversampled pixels using the current
    imaging geometry.

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS).

    Returns:
        int: Padding radius in oversampled pixels (>= 0).
    """
    img_size = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    os_factor = int(params["psf_oversampling_factor"])
    NA = float(params["numerical_aperture"])
    n_medium = float(params["refractive_index_medium"])
    wavelength_nm = float(params["wavelength_nm"])

    if img_size <= 0 or pixel_size_nm <= 0 or os_factor <= 0:
        raise ValueError(
            "PARAMS['image_size_pixels'], PARAMS['pixel_size_nm'], and "
            "PARAMS['psf_oversampling_factor'] must all be positive."
        )

    # If the optical parameters are degenerate, no meaningful Airy model exists.
    # In that case, fall back to zero padding (equivalent to the old behavior).
    if NA <= 0.0 or wavelength_nm <= 0.0 or n_medium <= 0.0:
        return 0

    threshold = float(params.get("psf_intensity_fraction_threshold", 1e-3))
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            "PARAMS['psf_intensity_fraction_threshold'] must be in the open interval (0, 1)."
        )

    # Wavelength inside the medium in nanometers.
    wavelength_medium_nm = wavelength_nm / n_medium

    # Dimensionless radial coordinate rho = (NA * r) / lambda_medium.
    # We sample rho over a generous range so that multiple Airy rings are covered.
    rho_max = 50.0
    num_samples = 20000
    rho = np.linspace(1e-4, rho_max, num_samples)
    x = np.pi * rho

    # Airy pattern intensity normalized to the on-axis value (I(0) = 1 in this model).
    I_rel = (2.0 * j1(x) / x) ** 2

    # We want to keep all radii where the intensity is at least `threshold`.
    # The outermost such radius defines the padding.
    indices_above = np.where(I_rel >= threshold)[0]
    if indices_above.size == 0:
        rho_crit = 0.0
    else:
        rho_crit = float(rho[indices_above[-1]])

    # Convert the critical rho into a physical radius in nanometers:
    #   rho = NA * r / lambda_medium  =>  r = rho * lambda_medium / NA
    radius_nm = rho_crit * wavelength_medium_nm / NA

    # Do not let the padding radius exceed half the physical field of view.
    psf_size_nm = img_size * pixel_size_nm
    max_radius_nm = 0.5 * psf_size_nm
    radius_nm = min(radius_nm, max_radius_nm)

    # Convert to oversampled pixels. The oversampled pixel size in nm is:
    #   pixel_size_nm / os_factor
    radius_pixels_oversampled = radius_nm / pixel_size_nm * os_factor

    # Add a one-pixel safety margin to account for discretization.
    padding_pixels = int(np.ceil(radius_pixels_oversampled)) + 1

    # Ensure non-negative.
    return max(padding_pixels, 0)


def generate_video_and_masks(params, trajectories, ipsf_interpolators):
    """
    Generates all video frames and segmentation masks by placing particles according
    to their trajectories and applying the appropriate iPSF. Includes motion blur.

    This implementation renders each frame on an oversampled, padded canvas that
    is larger than the final field of view. The padding width is chosen so that
    circular wrap-around from np.roll occurs only outside the central region
    that is ultimately cropped and used for the video. This fixes periodicity
    artifacts when particles approach or cross the image boundaries while still
    using np.roll as the only positioning primitive.

    Args:
        params (dict): The main simulation parameter dictionary.
        trajectories (numpy.ndarray): The 3D array of particle trajectories with
            shape (num_particles, num_frames, 3) in nanometers.
        ipsf_interpolators (list): A list of iPSF interpolator objects, one for
            each particle.

    Returns:
        tuple[list, list]: A tuple containing two lists: one for the raw signal
            frames and one for the raw reference frames, both as 16-bit integer
            arrays.
    """
    num_frames = int(params["fps"] * params["duration_seconds"])
    dt = 1.0 / params["fps"]
    num_particles = params["num_particles"]

    img_size = params["image_size_pixels"]
    pixel_size_nm = params["pixel_size_nm"]
    os_factor = params["psf_oversampling_factor"]
    final_size = (img_size, img_size)
    os_size = img_size * os_factor

    # --- Determine PSF padding to eliminate np.roll periodicity in the FOV ---
    # We render on a larger oversampled canvas and crop the central os_size×os_size
    # region afterward. The padding radius is chosen so that PSF contributions
    # outside this central region are below a user-defined fraction of the
    # on-axis intensity.
    psf_padding_radius = estimate_psf_padding_radius_pixels(params)
    os_canvas_size = os_size + 2 * psf_padding_radius
    crop_start = psf_padding_radius
    crop_end = crop_start + os_size

    # --- Bit depth and camera count handling ---
    # The raw simulated frames are stored as uint16 but their meaningful dynamic
    # range is controlled by PARAMS["bit_depth"]. This allows us to simulate
    # 12-bit, 14-bit, or 16-bit cameras while keeping the storage format simple.
    bit_depth = params["bit_depth"]
    if not isinstance(bit_depth, int) or bit_depth <= 0:
        raise ValueError("PARAMS['bit_depth'] must be a positive integer.")

    max_supported_bit_depth = 16  # Limited by uint16 storage in this implementation.
    if bit_depth > max_supported_bit_depth:
        raise ValueError(
            f"PARAMS['bit_depth']={bit_depth} exceeds the maximum supported bit depth "
            f"of {max_supported_bit_depth} for uint16 storage."
        )

    max_camera_count = (1 << bit_depth) - 1

    E_ref = params["reference_field_amplitude"]
    background = params["background_intensity"]

    num_subsamples = params["motion_blur_subsamples"] if params["motion_blur_enabled"] else 1
    sub_dt = dt / num_subsamples

    all_signal_frames = []
    all_reference_frames = []

    # Trackability gating master switch: when False, the model must not gate
    # masks or stop video early.
    trackability_enabled = bool(params.get("trackability_enabled", True))

    # Initialize the human trackability confidence model if masks are enabled.
    # The model may still be constructed when trackability_enabled is False, but
    # in that case its state is not allowed to affect mask generation or the
    # video length.
    if params["mask_generation_enabled"]:
        trackability_model = TrackabilityModel(params, num_particles)
        trackability_threshold = params.get("trackability_confidence_threshold", 0.8)
        if not (0.0 <= trackability_threshold <= 1.0):
            raise ValueError(
                "PARAMS['trackability_confidence_threshold'] must be between 0 and 1."
            )
    else:
        trackability_model = None
        trackability_threshold = None

    print("Generating video frames and masks...")
    for f in tqdm(range(num_frames)):
        # Accumulators for the motion-blurred electric field of each particle,
        # defined on the padded oversampled canvas.
        blurred_particle_fields = [
            np.zeros((os_canvas_size, os_canvas_size), dtype=np.complex128)
            for _ in range(num_particles)
        ]

        # Temporary canvas used to place a single PSF before shifting with np.roll.
        psf_canvas = np.zeros((os_canvas_size, os_canvas_size), dtype=np.complex128)

        # --- Subsample rendering for motion blur ---
        for s in range(num_subsamples):
            current_time = f * dt + s * sub_dt
            frame_idx_floor = int(current_time / dt)
            frame_idx_ceil = min(frame_idx_floor + 1, num_frames - 1)
            interp_factor = (current_time / dt) - frame_idx_floor

            # Linearly interpolate particle positions between trajectory points.
            current_pos_nm = (
                (1.0 - interp_factor) * trajectories[:, frame_idx_floor, :]
                + interp_factor * trajectories[:, frame_idx_ceil, :]
            )

            for i in range(num_particles):
                px, py, pz = current_pos_nm[i]

                # Get the pre-computed scattered field (iPSF) for the particle's z-position.
                E_sca_2D = ipsf_interpolators[i]([pz])[0]

                # Upscale to the oversampled resolution for higher accuracy placement.
                resized_real = cv2.resize(
                    np.real(E_sca_2D),
                    (os_size, os_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_imag = cv2.resize(
                    np.imag(E_sca_2D),
                    (os_size, os_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                E_sca_2D_rescaled = resized_real + 1j * resized_imag

                # --- Position the PSF on the padded oversampled canvas using np.roll ---
                #
                # The PSF returned by the interpolator is centered in the os_size×os_size
                # array. We first embed this PSF into the central os_size×os_size region
                # of the larger padded canvas, corresponding to the field of view
                # centered in the larger array. We then use np.roll to translate this
                # pattern so its center coincides with the particle's (x, y) position.
                #
                # Because we have added sufficient padding around the field of view,
                # any circular wrap-around from np.roll occurs only in the padded
                # margins and never contaminates the central region that will be
                # cropped and used for the final video.
                psf_canvas.fill(0.0)
                psf_canvas[crop_start:crop_end, crop_start:crop_end] = E_sca_2D_rescaled

                center_x_px = int(round(px / pixel_size_nm * os_factor))
                center_y_px = int(round(py / pixel_size_nm * os_factor))

                # Compute integer shifts relative to the optical center of the field of view
                # in the oversampled coordinates. This is identical to the previous logic,
                # but the shift is now applied to the padded canvas.
                shift_x = center_x_px - os_size // 2
                shift_y = center_y_px - os_size // 2

                # Circularly shift the PSF to the particle position on the padded canvas.
                E_sca_particle_inst = np.roll(
                    psf_canvas,
                    shift=(shift_y, shift_x),
                    axis=(0, 1),
                )

                # Apply signal multiplier and accumulate for motion blur.
                blurred_particle_fields[i] += (
                    E_sca_particle_inst * params["particle_signal_multipliers"][i]
                )

        # Average the fields from all subsamples to create the final motion-blurred field.
        for i in range(num_particles):
            blurred_particle_fields[i] /= num_subsamples

        # --- Mask Generation for this Frame ---
        if params["mask_generation_enabled"]:
            for i in range(num_particles):
                # When trackability gating is enabled, skip particles that have
                # already been declared "lost" by the trackability model. When
                # disabled, always attempt a mask for every particle in every frame.
                if trackability_enabled and trackability_model.is_particle_lost(i):
                    continue

                # Crop the particle's field back to the central oversampled field of view.
                E_sca_particle_blurred_canvas = blurred_particle_fields[i]
                E_sca_particle_blurred_fov = E_sca_particle_blurred_canvas[
                    crop_start:crop_end, crop_start:crop_end
                ]

                # Contrast is the change in intensity caused by the particle's scattered field.
                contrast_os = (
                    np.abs(E_ref + E_sca_particle_blurred_fov) ** 2 - np.abs(E_ref) ** 2
                )
                contrast_final = cv2.resize(
                    contrast_os, final_size, interpolation=cv2.INTER_AREA
                )

                # Compute the human trackability confidence and gate mask generation
                # only if trackability is enabled. When disabled, generate masks
                # unconditionally for all particles and frames.
                if trackability_enabled:
                    position_nm = trajectories[i, f, :]  # [x, y, z] at the frame time
                    confidence = trackability_model.update_and_compute_confidence(
                        particle_index=i,
                        frame_index=f,
                        position_nm=position_nm,
                        contrast_image=contrast_final,
                    )

                    if confidence >= trackability_threshold:
                        generate_and_save_mask_for_particle(
                            contrast_image=contrast_final,
                            params=params,
                            particle_index=i,
                            frame_index=f,
                        )
                    else:
                        # Once a particle is considered lost, its mask is no longer generated
                        # for the remainder of the video.
                        trackability_model.lost[i] = True
                else:
                    # Trackability disabled: the model must not gate masks. Always
                    # generate and save a mask for this particle in this frame.
                    generate_and_save_mask_for_particle(
                        contrast_image=contrast_final,
                        params=params,
                        particle_index=i,
                        frame_index=f,
                    )

        # --- Final Video Frame Generation ---
        # Sum the motion-blurred fields from all particles on the padded canvas.
        E_sca_total_canvas = np.sum(blurred_particle_fields, axis=0)

        # Crop back to the central oversampled field of view.
        E_sca_total_fov = E_sca_total_canvas[crop_start:crop_end, crop_start:crop_end]

        # Interfere the total scattered field with the reference field to get intensity.
        intensity_os = np.abs(E_ref + E_sca_total_fov) ** 2
        intensity = cv2.resize(intensity_os, final_size, interpolation=cv2.INTER_AREA)

        # Scale intensity to camera counts.
        if np.max(intensity) > 0:
            intensity_scaled = background + (intensity - E_ref**2) * background
        else:
            intensity_scaled = background * np.ones_like(intensity)

        signal_frame_noisy = add_noise(intensity_scaled, params)
        all_signal_frames.append(
            np.clip(signal_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

        # Generate a corresponding noisy reference frame for background subtraction.
        reference_frame_ideal = np.full(final_size, background, dtype=float)
        reference_frame_noisy = add_noise(reference_frame_ideal, params)
        all_reference_frames.append(
            np.clip(reference_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

        # If all particles are lost according to the trackability model, we can
        # terminate video generation early. This early termination must only
        # occur when trackability gating is enabled; when disabled, the video
        # always runs for the full duration.
        if (
            params["mask_generation_enabled"]
            and trackability_enabled
            and trackability_model.are_all_particles_lost()
        ):
            print(
                f"All particles lost according to the trackability model at frame {f}. "
                "Terminating video generation early."
            )
            break

    print("Frame and mask generation complete.")
    return all_signal_frames, all_reference_frames