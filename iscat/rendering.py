import numpy as np
import cv2
from tqdm import tqdm
from scipy.special import j1

from mask_generation import generate_and_save_mask_for_particle
from trackability import TrackabilityModel
from chip_pattern import generate_reference_and_background_maps, compute_contrast_scale_for_frame


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
    the simulated field of view so that PSF contributions from particles
    located just outside the nominal FOV can be represented on the padded
    canvas without significant truncation in the central region that is
    ultimately written to the video.

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
    # In that case, fall back to zero padding.
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


def _accumulate_psf_on_canvas(canvas, psf, center_x, center_y):
    """
    Add a complex-valued PSF patch into a larger complex canvas at a given
    center position, clipping at the canvas boundaries.

    This helper performs non-periodic placement: any part of the PSF kernel
    that would fall outside the canvas is simply discarded, which corresponds
    to truncation by the finite sensor area.

    Args:
        canvas (np.ndarray): 2D complex array to accumulate into (modified in-place).
        psf (np.ndarray): 2D complex array representing the PSF kernel. It is
            assumed to be centered at (psf.shape[1]//2, psf.shape[0]//2).
        center_x (int): X-index of the PSF center in canvas coordinates.
        center_y (int): Y-index of the PSF center in canvas coordinates.
    """
    H, W = canvas.shape
    kh, kw = psf.shape

    # Kernel center indices in its own coordinate system.
    kc_y = kh // 2
    kc_x = kw // 2

    # Intended bounds of the kernel on the canvas.
    x0 = center_x - kc_x
    y0 = center_y - kc_y
    x1 = x0 + kw
    y1 = y0 + kh

    # Clip the bounds to the canvas extent.
    x0_c = max(0, x0)
    y0_c = max(0, y0)
    x1_c = min(W, x1)
    y1_c = min(H, y1)

    # If there is no overlap, nothing to do.
    if x0_c >= x1_c or y0_c >= y1_c:
        return

    # Corresponding region in the PSF kernel.
    kx0 = x0_c - x0
    ky0 = y0_c - y0
    kx1 = kx0 + (x1_c - x0_c)
    ky1 = ky0 + (y1_c - y0_c)

    canvas[y0_c:y1_c, x0_c:x1_c] += psf[ky0:ky1, kx0:kx1]


def generate_video_and_masks(params, trajectories, ipsf_interpolators):
    """
    Generates all video frames and segmentation masks by placing particles according
    to their trajectories and applying the appropriate iPSF. Includes motion blur.

    This implementation renders each frame on an oversampled, padded canvas that
    is larger than the final field of view. Each particle's PSF is added directly
    to this canvas at its physical location using explicit clipping at the canvas
    boundaries (non-periodic placement). The padding width is chosen so that any
    truncation of the PSF occurs only where its intensity is negligible in the
    central region that is ultimately cropped and used for the video. This removes
    periodic wrap-around artifacts that arise when using circular shifts for PSF
    placement, while preserving the overall simulation behavior.

    The stationary reference field and background intensity are represented as
    2D maps generated by `chip_pattern.generate_reference_and_background_maps`.
    The base maps are time-independent; when a time-dependent chip pattern
    contrast model is selected (e.g., "time_dependent_v1"), the dimensionless
    pattern is reconstructed from these base maps and modulated per-frame using
    `compute_contrast_scale_for_frame`. When the contrast model is "static",
    the behavior is identical to the original implementation.

    The temporal sampling used for motion blur within each frame is controlled
    by PARAMS["exposure_time_ms"]. For a given frame at index f:
        - The frame interval is 1 / fps.
        - The exposure window is a contiguous interval of length
          exposure_time_ms (converted to seconds) centered on the frame's
          midpoint time.
        - Particle positions for motion blur are sampled uniformly over this
          exposure window and interpolated between the stored trajectory
          positions at integer frame times.
    """
    # --- Basic timing parameters ---
    fps = float(params["fps"])
    duration_seconds = float(params["duration_seconds"])
    num_frames = int(fps * duration_seconds)
    if num_frames <= 0:
        raise ValueError(
            "The product PARAMS['fps'] * PARAMS['duration_seconds'] must be "
            "positive to generate at least one frame."
        )

    frame_interval_s = 1.0 / fps

    # Exposure time in seconds for motion blur integration. If the parameter is
    # omitted, assume a full-frame exposure so that behavior matches the
    # original implementation.
    exposure_time_ms = float(params.get("exposure_time_ms", 1000.0 * frame_interval_s))
    exposure_time_s = exposure_time_ms / 1000.0

    if exposure_time_s <= 0.0:
        raise ValueError("PARAMS['exposure_time_ms'] must be positive.")
    if exposure_time_s > frame_interval_s + 1e-12:
        raise ValueError(
            "PARAMS['exposure_time_ms'] must satisfy exposure_time_ms <= 1000 / fps "
            "so that the exposure window is contained within a single frame interval."
        )

    num_particles = int(params["num_particles"])

    img_size = params["image_size_pixels"]
    pixel_size_nm = params["pixel_size_nm"]
    os_factor = params["psf_oversampling_factor"]
    final_size = (img_size, img_size)
    os_size = img_size * os_factor

    # --- Determine PSF padding to avoid truncation of relevant PSF energy ---
    # We render on a larger oversampled canvas and crop the central os_size×os_size
    # region afterward. The padding radius is chosen so that PSF contributions
    # outside this central region are below a user-defined fraction of the
    # on-axis intensity.
    psf_padding_radius = estimate_psf_padding_radius_pixels(params)
    os_canvas_size = os_size + 2 * psf_padding_radius
    crop_start = psf_padding_radius
    crop_end = crop_start + os_size

    # --- Precompute oversampled radius grid for radial PSF upsampling ---
    # This grid is used to radially resample the PSF from the pupil_samples×pupil_samples
    # grid to the oversampled os_size×os_size grid without introducing anisotropy
    # from separable interpolation (e.g., cv2.resize).
    if os_factor > 1:
        yy_os, xx_os = np.indices((os_size, os_size))
        center_os = os_size // 2
        r_os_pix_float = np.sqrt((xx_os - center_os) ** 2 + (yy_os - center_os) ** 2)
        r_os_flat = r_os_pix_float.ravel()
    else:
        r_os_flat = None

    # --- Stationary reference field and background maps (base) ---
    # These maps encode the spatially varying reference field and background
    # intensity at a reference contrast level. Temporal evolution of the chip
    # pattern contrast is applied on top of these base maps if requested.
    fov_shape_os = (os_size, os_size)
    E_ref_os_base, E_ref_final_base, background_final_base = generate_reference_and_background_maps(
        params,
        fov_shape_os=fov_shape_os,
        final_fov_shape=final_size,
    )
    E_ref_intensity_os_base = np.abs(E_ref_os_base) ** 2
    E_ref_intensity_final_base = np.abs(E_ref_final_base) ** 2

    # Determine how (or if) the chip pattern contrast evolves over time.
    contrast_model_raw = params.get("chip_pattern_contrast_model", "static")
    contrast_model = str(contrast_model_raw).strip().lower()
    if contrast_model not in ("static", "time_dependent_v1"):
        raise ValueError(
            "Unsupported chip_pattern_contrast_model "
            f"'{contrast_model_raw}'. Supported values are 'static' and 'time_dependent_v1'."
        )
    use_dynamic_contrast = (contrast_model == "time_dependent_v1")

    # Scalars controlling the mapping from dimensionless pattern to physical
    # reference field and background intensity.
    E_ref_amplitude = float(params["reference_field_amplitude"])
    background_intensity = float(params["background_intensity"])

    # For time-dependent contrast, reconstruct the dimensionless base pattern
    # maps so that a per-frame contrast scale can be applied without changing
    # the underlying geometry.
    if use_dynamic_contrast:
        if E_ref_amplitude > 0.0:
            pattern_os_base = E_ref_intensity_os_base / (E_ref_amplitude ** 2)
        else:
            pattern_os_base = np.ones_like(E_ref_intensity_os_base, dtype=float)

        if background_intensity > 0.0:
            pattern_final_base = background_final_base / background_intensity
        else:
            pattern_final_base = np.ones_like(background_final_base, dtype=float)

        # Normalize patterns to unit mean so that the global brightness is
        # controlled solely by background_intensity.
        mean_os = float(pattern_os_base.mean())
        if mean_os > 0.0:
            pattern_os_base /= mean_os

        mean_final = float(pattern_final_base.mean())
        if mean_final > 0.0:
            pattern_final_base /= mean_final

    # --- Bit depth and camera count handling ---
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

    num_subsamples = params["motion_blur_subsamples"] if params["motion_blur_enabled"] else 1
    if not isinstance(num_subsamples, int) or num_subsamples <= 0:
        raise ValueError(
            "PARAMS['motion_blur_subsamples'] must be a positive integer."
        )
    sub_dt = exposure_time_s / num_subsamples

    all_signal_frames = []
    all_reference_frames = []

    # Trackability gating master switch: when False, the model must not gate
    # masks or stop video early.
    trackability_enabled = bool(params.get("trackability_enabled", True))

    # Initialize the human trackability confidence model if masks are enabled.
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
        # --- Per-frame reference field and background maps ---
        if use_dynamic_contrast:
            alpha_f = compute_contrast_scale_for_frame(params, f, num_frames)

            pattern_os_f = 1.0 + alpha_f * (pattern_os_base - 1.0)
            pattern_final_f = 1.0 + alpha_f * (pattern_final_base - 1.0)

            pattern_os_f = np.maximum(pattern_os_f, 1e-8)
            pattern_final_f = np.maximum(pattern_final_f, 1e-8)

            E_ref_os = (E_ref_amplitude * np.sqrt(pattern_os_f)).astype(np.complex128)
            E_ref_intensity_os = (E_ref_amplitude ** 2) * pattern_os_f

            E_ref_intensity_final = (E_ref_amplitude ** 2) * pattern_final_f
            background_final = background_intensity * pattern_final_f
        else:
            E_ref_os = E_ref_os_base
            E_ref_intensity_os = E_ref_intensity_os_base
            E_ref_intensity_final = E_ref_intensity_final_base
            background_final = background_final_base

        # Accumulators for the motion-blurred electric field of each particle,
        # defined on the padded oversampled canvas.
        blurred_particle_fields = [
            np.zeros((os_canvas_size, os_canvas_size), dtype=np.complex128)
            for _ in range(num_particles)
        ]

        # --- Subsample rendering for motion blur ---
        for s in range(num_subsamples):
            frame_center_time = (f + 0.5) * frame_interval_s
            start_time = frame_center_time - 0.5 * exposure_time_s
            current_time = start_time + (s + 0.5) * sub_dt

            normalized_time = current_time / frame_interval_s
            frame_idx_floor = int(np.floor(normalized_time))
            if frame_idx_floor < 0:
                frame_idx_floor = 0

            if frame_idx_floor >= num_frames - 1:
                frame_idx_floor = num_frames - 1
                frame_idx_ceil = num_frames - 1
                interp_factor = 0.0
            else:
                frame_idx_ceil = frame_idx_floor + 1
                interp_factor = normalized_time - frame_idx_floor

            current_pos_nm = (
                (1.0 - interp_factor) * trajectories[:, frame_idx_floor, :]
                + interp_factor * trajectories[:, frame_idx_ceil, :]
            )

            for i in range(num_particles):
                px, py, pz = current_pos_nm[i]

                # Get the pre-computed scattered field (iPSF) for the particle's z-position.
                E_sca_2D = ipsf_interpolators[i]([pz])[0]

                # Upscale to the oversampled resolution for higher accuracy placement.
                # Instead of cv2.resize (separable, axis-biased), we radially resample
                # the PSF from the central radial profile to preserve rotational symmetry.
                if os_factor > 1:
                    psf_size = E_sca_2D.shape[0]
                    center_psf = psf_size // 2

                    # 1D radial profile from the central row. compute_ipsf_stack has
                    # already enforced radial symmetry, so any radial line is valid.
                    E_radial_line = E_sca_2D[center_psf, center_psf:]

                    max_bin_psf = E_radial_line.size - 1
                    if max_bin_psf <= 0:
                        E_sca_2D_rescaled = np.zeros((os_size, os_size), dtype=np.complex128)
                    else:
                        # Radii in PSF pixel units: 0,1,2,...,max_bin_psf
                        r_bins = np.arange(max_bin_psf + 1, dtype=float)

                        # Map oversampled radii (os_size grid) back to PSF radii.
                        # This uses the same index-space scaling as a 512->os_size resize:
                        #   r_psf = r_os * (psf_size / os_size)
                        scale = psf_size / float(os_size)
                        r_src = r_os_flat * scale

                        E_real_interp = np.interp(
                            r_src,
                            r_bins,
                            E_radial_line.real,
                            left=E_radial_line.real[0],
                            right=0.0,
                        )
                        E_imag_interp = np.interp(
                            r_src,
                            r_bins,
                            E_radial_line.imag,
                            left=E_radial_line.imag[0],
                            right=0.0,
                        )
                        E_sca_2D_rescaled = (
                            E_real_interp + 1j * E_imag_interp
                        ).reshape(os_size, os_size)
                else:
                    # No oversampling: work directly at the final resolution.
                    E_sca_2D_rescaled = E_sca_2D

                # --- Position the PSF on the padded oversampled canvas using explicit clipping ---
                # Convert the particle position (in nm) into oversampled FOV pixel coordinates.
                center_x_px = int(round(px / pixel_size_nm * os_factor))
                center_y_px = int(round(py / pixel_size_nm * os_factor))

                # Map from FOV coordinates into the padded canvas coordinates by
                # offsetting with the padding margin.
                center_x_canvas = crop_start + center_x_px
                center_y_canvas = crop_start + center_y_px

                # Scale the PSF by the per-particle amplitude multiplier and
                # accumulate it non-periodically onto the particle's canvas.
                psf_scaled = E_sca_2D_rescaled * params["particle_signal_multipliers"][i]
                _accumulate_psf_on_canvas(
                    blurred_particle_fields[i],
                    psf_scaled,
                    center_x_canvas,
                    center_y_canvas,
                )

        # Average the fields from all subsamples to create the final motion-blurred field.
        for i in range(num_particles):
            blurred_particle_fields[i] /= num_subsamples

        # --- Mask Generation for this Frame ---
        if params["mask_generation_enabled"]:
            for i in range(num_particles):
                if trackability_enabled and trackability_model.is_particle_lost(i):
                    continue

                E_sca_particle_blurred_canvas = blurred_particle_fields[i]
                E_sca_particle_blurred_fov = E_sca_particle_blurred_canvas[
                    crop_start:crop_end, crop_start:crop_end
                ]

                contrast_os = (
                    np.abs(E_ref_os + E_sca_particle_blurred_fov) ** 2
                    - E_ref_intensity_os
                )
                # Keep your original intensity downsampling behavior (cv2.INTER_AREA)
                # so the overall look remains close to your previous videos.
                contrast_final = cv2.resize(
                    contrast_os, final_size, interpolation=cv2.INTER_AREA
                )

                if trackability_enabled:
                    position_nm = trajectories[i, f, :]
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
                        trackability_model.lost[i] = True
                else:
                    generate_and_save_mask_for_particle(
                        contrast_image=contrast_final,
                        params=params,
                        particle_index=i,
                        frame_index=f,
                    )

        # --- Final Video Frame Generation ---
        E_sca_total_canvas = np.sum(blurred_particle_fields, axis=0)

        E_sca_total_fov = E_sca_total_canvas[crop_start:crop_end, crop_start:crop_end]

        intensity_os = np.abs(E_ref_os + E_sca_total_fov) ** 2
        # Keep original cv2.resize for final intensity as well for now.
        intensity = cv2.resize(intensity_os, final_size, interpolation=cv2.INTER_AREA)

        if np.max(intensity) > 0:
            E_ref_intensity_safe = np.maximum(E_ref_intensity_final, 1e-12)
            intensity_scaled = background_final * (intensity / E_ref_intensity_safe)
        else:
            intensity_scaled = background_final.copy()

        signal_frame_noisy = add_noise(intensity_scaled, params)
        all_signal_frames.append(
            np.clip(signal_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

        reference_frame_ideal = background_final.copy()
        reference_frame_noisy = add_noise(reference_frame_ideal, params)
        all_reference_frames.append(
            np.clip(reference_frame_noisy, 0, max_camera_count).astype(np.uint16)
        )

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