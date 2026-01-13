import numpy as np
import cv2
from tqdm import tqdm
import os

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

def generate_video_and_masks(params, trajectories, ipsf_interpolators):
    """
    Generates all video frames and segmentation masks by placing particles according
    to their trajectories and applying the appropriate iPSF. Includes motion blur.

    Args:
        params (dict): The main simulation parameter dictionary.
        trajectories (numpy.ndarray): The 3D array of particle trajectories.
        ipsf_interpolators (list): A list of iPSF interpolator objects, one for each particle.

    Returns:
        tuple[list, list]: A tuple containing two lists: one for the raw signal
                           frames and one for the raw reference frames, both as
                           16-bit integer arrays.
    """
    num_frames = int(params["fps"] * params["duration_seconds"])
    dt = 1 / params["fps"]
    img_size = params["image_size_pixels"]
    pixel_size_nm = params["pixel_size_nm"]
    os_factor = params["psf_oversampling_factor"]
    final_size = (img_size, img_size)
    os_size = img_size * os_factor

    E_ref = params["reference_field_amplitude"]
    background = params["background_intensity"]
    
    num_subsamples = params["motion_blur_subsamples"] if params["motion_blur_enabled"] else 1
    sub_dt = dt / num_subsamples

    all_signal_frames = []
    all_reference_frames = []

    print("Generating video frames and masks...")
    for f in tqdm(range(num_frames)):
        # Accumulators for the motion-blurred electric field of each particle.
        blurred_particle_fields = [
            np.zeros((os_size, os_size), dtype=np.complex128)
            for _ in range(params["num_particles"])
        ]

        # --- Subsample rendering for motion blur ---
        for s in range(num_subsamples):
            current_time = f * dt + s * sub_dt
            frame_idx_floor = int(current_time / dt)
            frame_idx_ceil = min(frame_idx_floor + 1, num_frames - 1)
            interp_factor = (current_time / dt) - frame_idx_floor

            # Linearly interpolate particle positions between trajectory points.
            current_pos_nm = (
                (1 - interp_factor) * trajectories[:, frame_idx_floor, :] +
                interp_factor * trajectories[:, frame_idx_ceil, :]
            )

            for i in range(params["num_particles"]):
                px, py, pz = current_pos_nm[i]

                # Get the pre-computed scattered field (iPSF) for the particle's z-position.
                E_sca_2D = ipsf_interpolators[i]([pz])[0]
                
                # Upscale to the oversampled resolution for higher accuracy placement.
                resized_real = cv2.resize(
                    np.real(E_sca_2D),
                    (os_size, os_size),
                    interpolation=cv2.INTER_LINEAR
                )
                resized_imag = cv2.resize(
                    np.imag(E_sca_2D),
                    (os_size, os_size),
                    interpolation=cv2.INTER_LINEAR
                )
                E_sca_2D_rescaled = resized_real + 1j * resized_imag

                # --- Position the PSF on the oversampled canvas by circularly shifting it ---
                # The PSF returned by the interpolator is centered in the array. We translate
                # this pattern so that its center coincides with the particle's (x, y) position
                # in the oversampled image grid. This avoids creating static rectangular
                # support regions from zero-padding/cropping.
                center_x_px = int(round(px / pixel_size_nm * os_factor))
                center_y_px = int(round(py / pixel_size_nm * os_factor))

                # Compute integer shifts relative to the optical center of the field of view.
                shift_x = center_x_px - os_size // 2
                shift_y = center_y_px - os_size // 2

                # Circularly shift the PSF to the particle position.
                E_sca_particle_inst = np.roll(
                    E_sca_2D_rescaled,
                    shift=(shift_y, shift_x),
                    axis=(0, 1)
                )
                
                # Apply signal multiplier and accumulate for motion blur.
                blurred_particle_fields[i] += (
                    E_sca_particle_inst * params["particle_signal_multipliers"][i]
                )

        # Average the fields from all subsamples to create the final motion-blurred field.
        for i in range(params["num_particles"]):
            blurred_particle_fields[i] /= num_subsamples

        # --- Mask Generation for this Frame ---
        if params["mask_generation_enabled"]:
            for i in range(params["num_particles"]):
                E_sca_particle_blurred = blurred_particle_fields[i]
                
                # Contrast is the change in intensity caused by the particle's scattered field.
                contrast_os = np.abs(E_ref + E_sca_particle_blurred)**2 - np.abs(E_ref)**2
                contrast_final = cv2.resize(contrast_os, final_size, interpolation=cv2.INTER_AREA)
                
                # Create a binary mask by thresholding the particle's own signal strength.
                max_val = np.max(np.abs(contrast_final))
                if max_val > 1e-9:  # Avoid division by zero for invisible particles.
                    normalized_contrast = np.abs(contrast_final) / max_val
                    mask = (normalized_contrast > params["mask_threshold"]).astype(np.uint8) * 255
                else:  # If particle has no signal, generate an empty mask.
                    mask = np.zeros(final_size, dtype=np.uint8)

                mask_path = os.path.join(
                    params["mask_output_directory"],
                    f"particle_{i+1}",
                    f"frame_{f:04d}.png"
                )
                cv2.imwrite(mask_path, mask)

        # --- Final Video Frame Generation ---
        E_sca_total = np.sum(blurred_particle_fields, axis=0)
        
        # Interfere the total scattered field with the reference field to get intensity.
        intensity_os = np.abs(E_ref + E_sca_total)**2
        intensity = cv2.resize(intensity_os, final_size, interpolation=cv2.INTER_AREA)

        # Scale intensity to camera counts.
        if np.max(intensity) > 0:
            intensity_scaled = background + (intensity - E_ref**2) * background
        else:
            intensity_scaled = background * np.ones_like(intensity)

        signal_frame_noisy = add_noise(intensity_scaled, params)
        all_signal_frames.append(np.clip(signal_frame_noisy, 0, 65535).astype(np.uint16))

        # Generate a corresponding noisy reference frame for background subtraction.
        reference_frame_ideal = np.full(final_size, background, dtype=float)
        reference_frame_noisy = add_noise(reference_frame_ideal, params)
        all_reference_frames.append(np.clip(reference_frame_noisy, 0, 65535).astype(np.uint16))

    print("Frame and mask generation complete.")
    return all_signal_frames, all_reference_frames