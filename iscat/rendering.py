import numpy as np
import cv2
from tqdm import tqdm
from scipy.special import j1

from mask_generation import generate_and_save_mask_for_particle
from trackability import TrackabilityModel
from chip_pattern import generate_reference_and_background_maps, compute_contrast_scale_for_frame
from particle_model import ParticleInstance, ParticleType, SubParticle


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

    if NA <= 0.0 or wavelength_nm <= 0.0 or n_medium <= 0.0:
        return 0

    threshold = float(params.get("psf_intensity_fraction_threshold", 1e-3))
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            "PARAMS['psf_intensity_fraction_threshold'] must be in the open interval (0, 1)."
        )

    wavelength_medium_nm = wavelength_nm / n_medium

    rho_max = 50.0
    num_samples = 20000
    rho = np.linspace(1e-4, rho_max, num_samples)
    x = np.pi * rho

    I_rel = (2.0 * j1(x) / x) ** 2

    indices_above = np.where(I_rel >= threshold)[0]
    if indices_above.size == 0:
        rho_crit = 0.0
    else:
        rho_crit = float(rho[indices_above[-1]])

    radius_nm = rho_crit * wavelength_medium_nm / NA

    psf_size_nm = img_size * pixel_size_nm
    max_radius_nm = 0.5 * psf_size_nm
    radius_nm = min(radius_nm, max_radius_nm)

    radius_pixels_oversampled = radius_nm / pixel_size_nm * os_factor

    padding_pixels = int(np.ceil(radius_pixels_oversampled)) + 1
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

    kc_y = kh // 2
    kc_x = kw // 2

    x0 = center_x - kc_x
    y0 = center_y - kc_y
    x1 = x0 + kw
    y1 = y0 + kh

    x0_c = max(0, x0)
    y0_c = max(0, y0)
    x1_c = min(W, x1)
    y1_c = min(H, y1)

    if x0_c >= x1_c or y0_c >= y1_c:
        return

    kx0 = x0_c - x0
    ky0 = y0_c - y0
    kx1 = kx0 + (x1_c - x0_c)
    ky1 = ky0 + (y1_c - y0_c)

    canvas[y0_c:y1_c, x0_c:x1_c] += psf[ky0:ky1, kx0:kx1]


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")

    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0))
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0))
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0))
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,) as [w, x, y, z].")

    w, x, y, z = q
    norm = np.linalg.norm(q)
    if norm == 0.0:
        w, x, y, z = 1.0, 0.0, 0.0, 0.0
    else:
        w /= norm
        x /= norm
        y /= norm
        z /= norm

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),   ww - xx - yy + zz],
        ],
        dtype=float,
    )
    return R


def _slerp_quaternions(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation (slerp) between two unit quaternions.
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError("Quaternions must have shape (4,) as [w, x, y, z].")

    q0 = q0 / (np.linalg.norm(q0) or 1.0)
    q1 = q1 / (np.linalg.norm(q1) or 1.0)

    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(max(dot, -1.0), 1.0)

    if dot > 0.9995:
        q = (1.0 - t) * q0 + t * q1
        return q / (np.linalg.norm(q) or 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = s0 * q0 + s1 * q1
    return q / (np.linalg.norm(q) or 1.0)


def _interpolate_orientation_for_instance(
    instance: ParticleInstance,
    time_index_float: float,
) -> np.ndarray | None:
    """
    Interpolate the orientation of a particle instance at a fractional frame index.
    """
    orientations = instance.orientation_matrices
    if orientations is None:
        return None

    num_frames = orientations.shape[0]
    if num_frames == 0:
        return None

    t = float(time_index_float)
    if t <= 0.0:
        return orientations[0]
    if t >= num_frames - 1:
        return orientations[-1]

    t_floor = int(np.floor(t))
    t_ceil = t_floor + 1
    alpha = t - t_floor

    if t_ceil >= num_frames:
        return orientations[-1]

    R0 = orientations[t_floor]
    R1 = orientations[t_ceil]

    q0 = _rotation_matrix_to_quaternion(R0)
    q1 = _rotation_matrix_to_quaternion(R1)
    q_interp = _slerp_quaternions(q0, q1, alpha)
    return _quaternion_to_rotation_matrix(q_interp)


def _iter_subparticle_render_info(
    instance: ParticleInstance,
    base_position_nm: np.ndarray,
    orientation_matrix: np.ndarray | None,
) -> list[tuple[np.ndarray, object, float]]:
    """
    Compute the list of sub-particle render instructions for a given particle
    instance at a given (possibly interpolated) position and orientation.
    """
    ptype: ParticleType = instance.particle_type

    if not ptype.is_composite or not ptype.sub_particles:
        return [
            (
                np.asarray(base_position_nm, dtype=float),
                ptype.ipsf_interpolator,
                1.0,
            )
        ]

    base_world_pos = np.asarray(base_position_nm, dtype=float)
    if base_world_pos.shape != (3,):
        raise ValueError(
            "base_position_nm must be a length-3 vector [x, y, z] in nm."
        )

    R = None
    if orientation_matrix is not None:
        R = np.asarray(orientation_matrix, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(
                "orientation_matrix must be a 3x3 rotation matrix when provided."
            )

    sub_infos: list[tuple[np.ndarray, object, float]] = []
    for sub in ptype.sub_particles:
        offset = np.asarray(sub.offset_nm, dtype=float)
        if offset.shape != (3,):
            raise ValueError(
                "SubParticle.offset_nm must be a length-3 vector [dx, dy, dz] in nm."
            )

        if R is not None:
            rotated_offset = R @ offset
        else:
            rotated_offset = offset

        sub_pos_world = base_world_pos + rotated_offset
        sub_infos.append(
            (
                sub_pos_world,
                sub.ipsf_interpolator,
                float(sub.signal_multiplier),
            )
        )

    return sub_infos


def generate_video_and_masks(params: dict, particle_instances: list[ParticleInstance]):
    """
    Generate all video frames and segmentation masks by placing particles
    according to their trajectories and applying the appropriate iPSF. Includes
    motion blur.

    The mask generation step uses a PSF-based central-lobe definition:

        - For each particle and frame, a particle-specific, noise-free contrast
          image (from the iPSF and reference field) is computed at the final
          resolution.
        - The mask is defined as all pixels within the central lobe, where
          the ring-averaged contrast keeps the same sign as at the particle
          center; the boundary is at the first sign flip or a small-amplitude
          fallback.

    Trackability still operates on the same contrast images and gating logic as
    before; only the internal mask geometry has been changed.
    """
    fps = float(params["fps"])
    duration_seconds = float(params["duration_seconds"])
    num_frames = int(fps * duration_seconds)
    if num_frames <= 0:
        raise ValueError(
            "The product PARAMS['fps'] * PARAMS['duration_seconds'] must be "
            "positive to generate at least one frame."
        )

    frame_interval_s = 1.0 / fps

    exposure_time_ms = float(params.get("exposure_time_ms", 1000.0 * frame_interval_s))
    exposure_time_s = exposure_time_ms / 1000.0

    if exposure_time_s <= 0.0:
        raise ValueError("PARAMS['exposure_time_ms'] must be positive.")
    if exposure_time_s > frame_interval_s + 1e-12:
        raise ValueError(
            "PARAMS['exposure_time_ms'] must satisfy exposure_time_ms <= 1000 / fps "
            "so that the exposure window is contained within a single frame interval."
        )

    num_particles = len(particle_instances)
    if num_particles != int(params["num_particles"]):
        raise ValueError(
            "Number of ParticleInstance objects (%d) does not match "
            "PARAMS['num_particles'] (%d)." % (num_particles, int(params["num_particles"]))
        )

    img_size = params["image_size_pixels"]
    pixel_size_nm = params["pixel_size_nm"]
    os_factor = params["psf_oversampling_factor"]
    final_size = (img_size, img_size)
    os_size = img_size * os_factor

    psf_padding_radius = estimate_psf_padding_radius_pixels(params)
    os_canvas_size = os_size + 2 * psf_padding_radius
    crop_start = psf_padding_radius
    crop_end = crop_start + os_size

    if os_factor > 1:
        yy_os, xx_os = np.indices((os_size, os_size))
        center_os = os_size // 2
        r_os_pix_float = np.sqrt((xx_os - center_os) ** 2 + (yy_os - center_os) ** 2)
        r_os_flat = r_os_pix_float.ravel()
    else:
        r_os_flat = None

    fov_shape_os = (os_size, os_size)
    E_ref_os_base, E_ref_final_base, background_final_base = generate_reference_and_background_maps(
        params,
        fov_shape_os=fov_shape_os,
        final_fov_shape=final_size,
    )
    E_ref_intensity_os_base = np.abs(E_ref_os_base) ** 2
    E_ref_intensity_final_base = np.abs(E_ref_final_base) ** 2

    contrast_model_raw = params.get("chip_pattern_contrast_model", "static")
    contrast_model = str(contrast_model_raw).strip().lower()
    if contrast_model not in ("static", "time_dependent_v1"):
        raise ValueError(
            "Unsupported chip_pattern_contrast_model "
            f"'{contrast_model_raw}'. Supported values are 'static' and 'time_dependent_v1'."
        )
    use_dynamic_contrast = (contrast_model == "time_dependent_v1")

    E_ref_amplitude = float(params["reference_field_amplitude"])
    background_intensity = float(params["background_intensity"])

    if E_ref_amplitude <= 0.0:
        raise ValueError(
            "PARAMS['reference_field_amplitude'] must be positive. "
            "A nonzero reference field is required for interferometric contrast."
        )

    if use_dynamic_contrast:
        if E_ref_amplitude > 0.0:
            pattern_os_base = E_ref_intensity_os_base / (E_ref_amplitude ** 2)
        else:
            pattern_os_base = np.ones_like(E_ref_intensity_os_base, dtype=float)

        if background_intensity > 0.0:
            pattern_final_base = background_final_base / background_intensity
        else:
            pattern_final_base = np.ones_like(background_final_base, dtype=float)

        mean_os = float(pattern_os_base.mean())
        if mean_os > 0.0:
            pattern_os_base /= mean_os

        mean_final = float(pattern_final_base.mean())
        if mean_final > 0.0:
            pattern_final_base /= mean_final

    bit_depth = params["bit_depth"]
    if not isinstance(bit_depth, int) or bit_depth <= 0:
        raise ValueError("PARAMS['bit_depth'] must be a positive integer.")

    max_supported_bit_depth = 16
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

    trackability_enabled = bool(params.get("trackability_enabled", True))

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

        blurred_particle_fields = [
            np.zeros((os_canvas_size, os_canvas_size), dtype=np.complex128)
            for _ in range(num_particles)
        ]

        for s in range(num_subsamples):
            frame_center_time = (f + 0.5) * frame_interval_s
            start_time = frame_center_time - 0.5 * exposure_time_s
            current_time = start_time + (s + 0.5) * sub_dt

            normalized_time = current_time / frame_interval_s
            time_index_float = normalized_time

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

            for i, instance in enumerate(particle_instances):
                traj = instance.trajectory_nm
                if traj.shape[0] != num_frames or traj.shape[1] != 3:
                    raise ValueError(
                        "ParticleInstance %d has trajectory shape %s, expected (%d, 3)."
                        % (i, traj.shape, num_frames)
                    )

                pos_floor = traj[frame_idx_floor]
                pos_ceil = traj[frame_idx_ceil]
                current_pos_nm = (1.0 - interp_factor) * pos_floor + interp_factor * pos_ceil

                orientation_matrix = _interpolate_orientation_for_instance(
                    instance=instance,
                    time_index_float=time_index_float,
                )

                sub_infos = _iter_subparticle_render_info(
                    instance=instance,
                    base_position_nm=current_pos_nm,
                    orientation_matrix=orientation_matrix,
                )

                for world_pos_nm, sub_interp, local_multiplier in sub_infos:
                    px, py, pz = world_pos_nm

                    E_sca_2D = sub_interp([pz])[0]

                    if os_factor > 1:
                        pupil_samples = E_sca_2D.shape[0]
                        center_psf = pupil_samples // 2
                        E_radial_line = E_sca_2D[center_psf, center_psf:]
                        max_bin_psf = E_radial_line.size - 1

                        if max_bin_psf > 0:
                            nm_per_pixel_psf = (
                                img_size * pixel_size_nm
                            ) / (os_factor * pupil_samples)
                            r_bins_nm = np.arange(max_bin_psf + 1) * nm_per_pixel_psf

                            nm_per_pixel_os = pixel_size_nm / os_factor
                            r_os_nm = r_os_flat * nm_per_pixel_os

                            E_real_interp = np.interp(
                                r_os_nm, r_bins_nm, E_radial_line.real, right=0.0
                            )
                            E_imag_interp = np.interp(
                                r_os_nm, r_bins_nm, E_radial_line.imag, right=0.0
                            )
                            E_sca_2D_rescaled = (
                                E_real_interp + 1j * E_imag_interp
                            ).reshape(os_size, os_size)
                        else:
                            E_sca_2D_rescaled = np.zeros(
                                (os_size, os_size), dtype=np.complex128
                            )
                    else:
                        E_sca_2D_rescaled = E_sca_2D

                    center_x_px = int(round(px / pixel_size_nm * os_factor))
                    center_y_px = int(round(py / pixel_size_nm * os_factor))

                    center_x_canvas = crop_start + center_x_px
                    center_y_canvas = crop_start + center_y_px

                    psf_scaled = (
                        E_sca_2D_rescaled
                        * instance.signal_multiplier
                        * local_multiplier
                    )
                    _accumulate_psf_on_canvas(
                        blurred_particle_fields[i],
                        psf_scaled,
                        center_x_canvas,
                        center_y_canvas,
                    )

        for i in range(num_particles):
            blurred_particle_fields[i] /= num_subsamples

        if params["mask_generation_enabled"]:
            for i, instance in enumerate(particle_instances):
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
                contrast_final = cv2.resize(
                    contrast_os, final_size, interpolation=cv2.INTER_AREA
                )

                contrast_final_normalized = contrast_final / (E_ref_amplitude ** 2)

                if trackability_enabled:
                    position_nm = instance.trajectory_nm[f, :]
                    confidence = trackability_model.update_and_compute_confidence(
                        particle_index=i,
                        frame_index=f,
                        position_nm=position_nm,
                        contrast_image=contrast_final_normalized,
                    )

                    if confidence >= trackability_threshold:
                        # Use the *current* physical position to define the mask
                        # center in final image pixels, consistent with the
                        # trajectory mapping and iPSF placement.
                        center_nm = (float(position_nm[0]), float(position_nm[1]))
                        generate_and_save_mask_for_particle(
                            contrast_image=contrast_final_normalized,
                            params=params,
                            particle_index=i,
                            frame_index=f,
                            center_nm=center_nm,
                        )
                    else:
                        trackability_model.lost[i] = True
                else:
                    # Trackability disabled: generate masks for all particles.
                    pos_nm = instance.trajectory_nm[f, :]
                    center_nm = (float(pos_nm[0]), float(pos_nm[1]))
                    generate_and_save_mask_for_particle(
                        contrast_image=contrast_final_normalized,
                        params=params,
                        particle_index=i,
                        frame_index=f,
                        center_nm=center_nm,
                    )

        E_sca_total_canvas = np.sum(blurred_particle_fields, axis=0)
        E_sca_total_fov = E_sca_total_canvas[crop_start:crop_end, crop_start:crop_end]

        intensity_os = np.abs(E_ref_os + E_sca_total_fov) ** 2
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