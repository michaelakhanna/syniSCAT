# File: iscat_simulation.py
#run conda activate sim_env

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from scipy.special import jn, yn
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import warnings

# Suppress RankWarning from polyfit
warnings.filterwarnings("ignore", category=np.RankWarning)
print("Starting iSCAT simulation...")
# --- SIMULATION PARAMETERS ---
PARAMS = {
    # --- IMAGE & VIDEO ---
    "image_size_pixels": 1024,       # Size of the output image in pixels (e.g., 512x512)
    "pixel_size_nm": 600,            # Physical size of one pixel in nanometers
    "fps": 30,                      # Frames per second of the output video
    "duration_seconds": 1,          # Total duration of the simulation video
    "output_filename": os.path.join(os.path.expanduser('~'), 'Desktop', 'iscat_simulation.mp4'),

    # --- MASK GENERATION ---
    "mask_generation_enabled": True, # Set to True to generate masks
    "mask_output_directory": os.path.join(os.path.expanduser('~'), 'Desktop', 'iscat_masks'), # Base folder for masks
    "mask_threshold": 0.3,           # Threshold for binarizing the mask (0.0 to 1.0 of particle's max signal)

    # --- OPTICAL SETUP ---
    "wavelength_nm": 635,           # Illumination wavelength in vacuum (nm)
    "numerical_aperture": 1.4,      # Objective's numerical aperture (NA)
    "magnification": 60,            # For reference: objective magnification (e.g., 60x)
    "objective_focal_length_mm": 3.0, # For a 60x objective with a standard 180mm tube lens (180/60=3)
    "refractive_index_medium": 1.33, # Refractive index of the sample medium (e.g., water)
    "refractive_index_immersion": 1.518, # Refractive index of immersion oil

    # --- PARTICLE PROPERTIES ---
    "num_particles": 2,
    "particle_diameters_nm": [60, 100], # Diameter of each particle in nm
    "particle_refractive_indices": [ # Complex refractive index of each particle type
        0.166 + 3.15j,  # Gold (Au) at 635 nm
        0.166 + 3.15j,
    ],
    "particle_signal_multipliers": [1.0, 1.0], # "Fluorescence"-like signal strength control. 0=off, 1=normal.
    # "particle_initial_positions_nm": [[x1, y1, z1], [x2, y2, z2], ...], # In nm from corner

    # --- BROWNIAN MOTION ---
    "temperature_K": 298.15,        # Temperature in Kelvin (25 C)
    "viscosity_Pa_s": 0.00089,      # Viscosity of the medium (water at 25 C) in Pascal-seconds

    # --- iPSF & SCATTERING CALCULATION ---
    "psf_oversampling_factor": 2,   # Calculate PSF at higher resolution for accuracy. 1 = no oversampling.
    "pupil_samples": 512,           # Resolution of pupil function grid. Higher is more accurate.
    "z_stack_range_nm": 30000,       # Axial range for pre-computing the iPSF stack (e.g., +/- 10 um)
    "z_stack_step_nm": 50,         # Axial step size for the iPSF stack

    # --- ABERRATIONS & PUPIL FUNCTION ---
    "spherical_aberration_strength": 0.25,
    "apodization_factor": 1.8,
    "random_aberration_strength": 1.5,

    # --- INTERFERENCE & NOISE MODEL ---
    "reference_field_amplitude": 1, # Amplitude of the reference field E_R
    "background_intensity": 100,   # Average intensity of the background (in camera counts),
    "shot_noise_enabled": True,      # poisson noise
    "shot_noise_scaling_factor": 1.00, # Custom knob to control shot noise strength (0=off, 1=full)
    "gaussian_noise_enabled": True,  # read noise
    "read_noise_std": 15,            # Standard deviation of Gaussian readout noise (in camera counts)

    # --- MOTION BLUR ---
    "motion_blur_enabled": True,     # Enable/disable motion blur simulation
    "motion_blur_subsamples": 4,     # Number of sub-steps per frame for motion blur. 1 = no blur.
}

# --- PHYSICAL CONSTANTS ---
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

def stokes_einstein_diffusion_coefficient(diameter_nm, temp_K, viscosity_Pa_s):
    """
    Calculates the diffusion coefficient for a spherical particle in a fluid
    using the Stokes-Einstein equation.

    Args:
        diameter_nm (float): The diameter of the particle in nanometers.
        temp_K (float): The absolute temperature of the fluid in Kelvin.
        viscosity_Pa_s (float): The dynamic viscosity of the fluid in Pascal-seconds.

    Returns:
        float: The diffusion coefficient in square meters per second (m^2/s).
    """
    radius_m = diameter_nm * 1e-9 / 2
    return (BOLTZMANN_CONSTANT * temp_K) / (6 * np.pi * viscosity_Pa_s * radius_m)

def simulate_trajectories(params):
    """
    Simulates 3D Brownian motion trajectories for a set of particles.

    Args:
        params (dict): A dictionary of simulation parameters.

    Returns:
        numpy.ndarray: A 3D array of shape (num_particles, num_frames, 3)
                       containing the [x, y, z] coordinates of each particle
                       for each frame, in nanometers.
    """
    num_frames = int(params["fps"] * params["duration_seconds"])
    dt = 1 / params["fps"]
    num_particles = params["num_particles"]
    
    # Initialize particle positions randomly if not explicitly provided.
    if "particle_initial_positions_nm" in params:
        initial_positions = np.array(params["particle_initial_positions_nm"], dtype=float)
    else:
        img_size_nm = params["image_size_pixels"] * params["pixel_size_nm"]
        initial_positions = np.random.rand(num_particles, 3) * [img_size_nm, img_size_nm, params["z_stack_range_nm"]]
        initial_positions[:, 2] -= params["z_stack_range_nm"] / 2

    trajectories = np.zeros((num_particles, num_frames, 3))
    trajectories[:, 0, :] = initial_positions

    # Calculate trajectory for each particle independently.
    for i in range(num_particles):
        D_m2_s = stokes_einstein_diffusion_coefficient(
            params["particle_diameters_nm"][i],
            params["temperature_K"],
            params["viscosity_Pa_s"]
        )
        sigma_m = np.sqrt(2 * D_m2_s * dt)
        sigma_nm = sigma_m * 1e9 # Convert standard deviation to nanometers
        
        # Generate random steps from a normal distribution.
        steps = np.random.normal(scale=sigma_nm, size=(num_frames - 1, 3))
        trajectories[i, 1:, :] = initial_positions[i] + np.cumsum(steps, axis=0)

    print("Generated Brownian motion trajectories.")
    return trajectories

def mie_an_bn(m, x):
    """
    Calculates Mie scattering coefficients a_n and b_n.

    Args:
        m (complex): The complex refractive index ratio (particle/medium).
        x (float): The size parameter (2*pi*r/lambda).
    """
    nmax = int(np.ceil(x + 4 * x**(1/3) + 2))
    n = np.arange(1, nmax + 1)
    
    # Riccati-Bessel functions
    psi_n_x = np.sqrt(0.5 * np.pi * x) * jn(n + 0.5, x)
    psi_n_mx = np.sqrt(0.5 * np.pi * m * x) * jn(n + 0.5, m * x)
    chi_n_x = -np.sqrt(0.5 * np.pi * x) * yn(n + 0.5, x)
    
    # Derivatives of Riccati-Bessel functions
    psi_prime_n_x = np.sqrt(0.5 * np.pi / x) * jn(n + 0.5, x) / 2 + np.sqrt(0.5 * np.pi * x) * (jn(n-1 + 0.5, x) - (n+1)/x * jn(n + 0.5, x))
    psi_prime_n_mx = np.sqrt(0.5 * np.pi / (m*x)) * jn(n + 0.5, m*x) / 2 + np.sqrt(0.5 * np.pi * m*x) * (jn(n-1 + 0.5, m*x) - (n+1)/(m*x) * jn(n + 0.5, m*x))

    xi_n_x = psi_n_x + 1j * chi_n_x
    xi_prime_n_x = psi_prime_n_x - 1j * np.sqrt(0.5*np.pi/x)*yn(n+0.5,x)/2 - 1j*np.sqrt(0.5*np.pi*x)*(yn(n-1+0.5,x) - (n+1)/x*yn(n+0.5,x))
    
    # Mie coefficients calculation
    a_n = (m**2 * psi_n_mx * psi_prime_n_x - psi_n_x * psi_prime_n_mx) / \
          (m**2 * psi_n_mx * xi_prime_n_x - xi_n_x * psi_prime_n_mx)
    b_n = (psi_n_mx * psi_prime_n_x - psi_n_x * psi_prime_n_mx) / \
          (psi_n_mx * xi_prime_n_x - xi_n_x * psi_prime_n_mx)
          
    return a_n, b_n

def mie_S1_S2(m, x, mu):
    """
    Calculates Mie scattering amplitude functions S1 and S2.
    
    Args:
        m (complex): complex refractive index ratio
        x (float): size parameter
        mu (float): cos(theta) where theta is the scattering angle
    """
    nmax = int(np.ceil(x + 4 * x**(1/3) + 2))
    a_n, b_n = mie_an_bn(m, x)
    
    S1 = 0j
    S2 = 0j
    pi_n = np.zeros(nmax + 2)
    tau_n = np.zeros(nmax + 2)
    pi_n[1] = 1.0

    # Summation over n for S1 and S2
    for n in range(1, nmax + 1):
        if n > 1:
            pi_n[n] = ((2 * n - 1) / (n - 1)) * mu * pi_n[n - 1] - (n / (n - 1)) * pi_n[n - 2]
        
        tau_n[n] = n * mu * pi_n[n] - (n + 1) * pi_n[n - 1]
        
        factor = (2 * n + 1) / (n * (n + 1))
        S1 += factor * (a_n[n-1] * pi_n[n] + b_n[n-1] * tau_n[n])
        S2 += factor * (a_n[n-1] * tau_n[n] + b_n[n-1] * pi_n[n])
        
    return S1, S2

def compute_ipsf_stack(params, particle_diameter_nm, particle_refractive_index):
    """
    Computes a complex 3D vectorial interferometric Point Spread Function (iPSF)
    stack using the Debye-Born integral, calculated via FFT for efficiency.

    Args:
        params (dict): The main simulation parameter dictionary.
        particle_diameter_nm (float): The diameter of the particle for this iPSF.
        particle_refractive_index (complex): The complex refractive index of the particle.

    Returns:
        scipy.interpolate.RegularGridInterpolator: An interpolator object that can
                                                   return the complex 2D iPSF for any
                                                   given z-position within the stack's range.
    """
    # --- Setup k-space coordinates and optical parameters ---
    os_factor = params["psf_oversampling_factor"]
    pupil_samples = params["pupil_samples"]
    psf_size_nm = params["image_size_pixels"] * params["pixel_size_nm"]
    n_medium = params["refractive_index_medium"]
    wavelength_medium_nm = params["wavelength_nm"] / n_medium
    k_medium = 2 * np.pi / wavelength_medium_nm

    dk = (2 * np.pi / psf_size_nm) * os_factor
    kx = np.arange(-pupil_samples // 2, pupil_samples // 2) * dk
    ky = np.arange(-pupil_samples // 2, pupil_samples // 2) * dk
    Kx, Ky = np.meshgrid(kx, ky)
    K_sq = Kx**2 + Ky**2
    
    # --- Define the pupil aperture and coordinates ---
    sin_theta = np.sqrt(K_sq) / k_medium
    max_sin_theta = params["numerical_aperture"] / n_medium
    aperture_mask = (sin_theta <= max_sin_theta).astype(float)
    
    cos_theta = np.zeros_like(sin_theta)
    valid_mask = sin_theta <= 1
    cos_theta[valid_mask] = np.sqrt(1 - sin_theta[valid_mask]**2)
    
    # --- Calculate Mie scattering amplitudes across the pupil ---
    m = particle_refractive_index / n_medium
    radius_nm = particle_diameter_nm / 2
    x = 2 * np.pi * radius_nm / wavelength_medium_nm
    mu = np.zeros_like(cos_theta)
    mu[valid_mask] = cos_theta[valid_mask]
    S1_vec, S2_vec = np.vectorize(mie_S1_S2)(m, x, mu)
    
    # --- Define aberration and apodization functions ---
    z_values = np.arange(-params["z_stack_range_nm"]/2, params["z_stack_range_nm"]/2 + 1, params["z_stack_step_nm"])
    rho = sin_theta / max_sin_theta
    zernike_spherical = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    spherical_phase = params["spherical_aberration_strength"] * zernike_spherical * 2 * np.pi
    apodization = np.exp(-params["apodization_factor"] * (rho**2))
    
    print(f"Computing iPSF stack for {particle_diameter_nm} nm particle...")
    ipsf_stack_complex = np.zeros((len(z_values), pupil_samples, pupil_samples), dtype=np.complex128)

    # --- Compute the iPSF for each Z-slice ---
    for i, z in enumerate(tqdm(z_values)):
        defocus_phase = k_medium * z * cos_theta
        aberration_phase = defocus_phase + spherical_phase
        
        # This simulates the complex, random aberrations of a real lens system.
        # It now uses the new parameter from the top of the file.
        aberration_phase += (np.random.rand(pupil_samples, pupil_samples) - 0.5) * params["random_aberration_strength"] * 2 * np.pi

        pupil_function = (-1j * wavelength_medium_nm) * aperture_mask * apodization * S2_vec * np.exp(1j * aberration_phase)
        
        # The Amplitude Spread Function (ASF) is the Fourier transform of the pupil function.
        asf = fftshift(ifft2(ifftshift(pupil_function)))
        ipsf_stack_complex[i, :, :] = asf
        
    # Create an interpolator for fast lookups later.
    interpolator = RegularGridInterpolator((z_values,), ipsf_stack_complex, method='linear', bounds_error=False, fill_value=0)
    
    print("iPSF stack computation complete.")
    return interpolator

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
        blurred_particle_fields = [np.zeros((os_size, os_size), dtype=np.complex128) for _ in range(params["num_particles"])]

        # --- Subsample rendering for motion blur ---
        # Render the scene at multiple points in time within a single frame and average the result.
        for s in range(num_subsamples):
            current_time = f * dt + s * sub_dt
            frame_idx_floor = int(current_time / dt)
            frame_idx_ceil = min(frame_idx_floor + 1, num_frames - 1)
            interp_factor = (current_time / dt) - frame_idx_floor

            # Linearly interpolate particle positions between trajectory points.
            current_pos_nm = (1 - interp_factor) * trajectories[:, frame_idx_floor, :] + \
                             interp_factor * trajectories[:, frame_idx_ceil, :]

            for i in range(params["num_particles"]):
                px, py, pz = current_pos_nm[i]

                # Get the pre-computed scattered field (iPSF) for the particle's z-position.
                E_sca_2D = ipsf_interpolators[i]([pz])[0]
                
                # Upscale to the oversampled resolution for higher accuracy placement.
                resized_real = cv2.resize(np.real(E_sca_2D), (os_size, os_size), interpolation=cv2.INTER_LINEAR)
                resized_imag = cv2.resize(np.imag(E_sca_2D), (os_size, os_size), interpolation=cv2.INTER_LINEAR)
                E_sca_2D_rescaled = resized_real + 1j * resized_imag
                
                # --- Direct-space rendering to prevent wrapping artifacts from FFT-based convolution ---
                E_sca_particle_inst = np.zeros((os_size, os_size), dtype=np.complex128)
                center_x_px, center_y_px = int(round(px / pixel_size_nm * os_factor)), int(round(py / pixel_size_nm * os_factor))
                psf_h, psf_w = E_sca_2D_rescaled.shape
                top, left = center_y_px - psf_h // 2, center_x_px - psf_w // 2

                # Define the overlapping region between the canvas and the PSF.
                canvas_y_min, canvas_y_max = max(0, top), min(os_size, top + psf_h)
                canvas_x_min, canvas_x_max = max(0, left), min(os_size, left + psf_w)

                if not (canvas_y_min >= canvas_y_max or canvas_x_min >= canvas_x_max):
                    psf_y_min, psf_y_max = canvas_y_min - top, canvas_y_max - top
                    psf_x_min, psf_x_max = canvas_x_min - left, canvas_x_max - left
                    
                    # Add the relevant portion of the PSF to the canvas.
                    E_sca_particle_inst[canvas_y_min:canvas_y_max, canvas_x_min:canvas_x_max] += \
                        E_sca_2D_rescaled[psf_y_min:psf_y_max, psf_x_min:psf_x_max]
                
                # Apply signal multiplier and accumulate for motion blur.
                blurred_particle_fields[i] += E_sca_particle_inst * params["particle_signal_multipliers"][i]

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
                if max_val > 1e-9: # Avoid division by zero for invisible particles.
                    normalized_contrast = np.abs(contrast_final) / max_val
                    mask = (normalized_contrast > params["mask_threshold"]).astype(np.uint8) * 255
                else: # If particle has no signal, generate an empty mask.
                    mask = np.zeros(final_size, dtype=np.uint8)

                mask_path = os.path.join(params["mask_output_directory"], f"particle_{i+1}", f"frame_{f:04d}.png")
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

def apply_background_subtraction(signal_frames, reference_frames):
    """
    Performs background subtraction using the formula (Signal - Reference) / Reference
    and normalizes the result to an 8-bit range for video encoding.

    Args:
        signal_frames (list): A list of 16-bit signal frames.
        reference_frames (list): A list of 16-bit reference frames.

    Returns:
        list: A list of 8-bit, normalized frames ready for video encoding.
    """
    if not signal_frames or not reference_frames: return []
    
    subtracted_frames = []
    print("Applying background subtraction...")
    for signal_frame, ref_frame in tqdm(zip(signal_frames, reference_frames), total=len(signal_frames)):
        subtracted = (signal_frame.astype(float) - ref_frame.astype(float)) / (ref_frame.astype(float) + 1e-9)
        subtracted_frames.append(subtracted)
        
    # Normalize the contrast range across the whole video for consistent brightness.
    min_val, max_val = np.percentile(subtracted_frames, [0.5, 99.5])
    
    final_frames_8bit = []
    if max_val > min_val:
        for frame in subtracted_frames:
            norm_frame = 255 * (frame - min_val) / (max_val - min_val)
            final_frames_8bit.append(np.clip(norm_frame, 0, 255).astype(np.uint8))
    else: # Handle case of no contrast.
        final_frames_8bit = [np.full(signal_frames[0].shape, 128, dtype=np.uint8) for _ in signal_frames]

    return final_frames_8bit

def main():
    """
    Main function to run the entire iSCAT simulation and video generation pipeline.
    """
    # --- Setup Output Directories ---
    if PARAMS["mask_generation_enabled"]:
        base_mask_dir = PARAMS["mask_output_directory"]
        print(f"Checking for mask output directories at {base_mask_dir}...")
        os.makedirs(base_mask_dir, exist_ok=True)
        for i in range(PARAMS["num_particles"]):
            particle_mask_dir = os.path.join(base_mask_dir, f"particle_{i+1}")
            os.makedirs(particle_mask_dir, exist_ok=True)
            
    # --- Step 1: Simulate particle movement ---
    trajectories_nm = simulate_trajectories(PARAMS)
    
    # --- Step 2: Compute unique iPSF stacks ---
    # To save computation time, only compute the iPSF once for each unique type of particle.
    unique_particles = {}
    for i in range(PARAMS["num_particles"]):
        key = (PARAMS["particle_diameters_nm"][i], PARAMS["particle_refractive_indices"][i].real, PARAMS["particle_refractive_indices"][i].imag)
        if key not in unique_particles:
            unique_particles[key] = compute_ipsf_stack(
                PARAMS,
                PARAMS["particle_diameters_nm"][i],
                PARAMS["particle_refractive_indices"][i]
            )
    
    # Assign the correct pre-computed iPSF interpolator to each particle.
    ipsf_interpolators = [
        unique_particles[(PARAMS["particle_diameters_nm"][i], PARAMS["particle_refractive_indices"][i].real, PARAMS["particle_refractive_indices"][i].imag)]
        for i in range(PARAMS["num_particles"])
    ]

    # --- Step 3: Generate raw video frames and masks ---
    raw_signal_frames, raw_reference_frames = generate_video_and_masks(PARAMS, trajectories_nm, ipsf_interpolators)
    
    # --- Step 4: Process frames for final video ---
    final_frames = apply_background_subtraction(raw_signal_frames, raw_reference_frames)
    
    if not final_frames:
        print("Video generation failed. Exiting.")
        return

    # --- Step 5: Save the final video ---
    print(f"Saving final video to {PARAMS['output_filename']}...")
    img_size = (PARAMS["image_size_pixels"], PARAMS["image_size_pixels"])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(PARAMS["output_filename"], fourcc, PARAMS["fps"], img_size, isColor=False)
    
    for frame in final_frames:
        video_writer.write(frame)
        
    video_writer.release()
    print("Simulation finished successfully!")

if __name__ == '__main__':
    main()