import os

# --- SIMULATION PARAMETERS ---
# This dictionary centralizes all configurable parameters for the simulation.
PARAMS = {
    # --- IMAGE & VIDEO ---
    "image_size_pixels": 1024,       # Size of the output image in pixels (e.g., 512x512)
    "pixel_size_nm": 600,            # Physical size of one pixel in nanometers
    "fps": 24,                       # Frames per second of the output video
    "duration_seconds": 1,           # Total duration of the simulation video
    "bit_depth": 16,                 # Bit depth of the raw simulated frames (e.g., 16 for iSCAT, 12 for fluorescence)
    "output_filename": os.path.join(
        os.path.expanduser('~'),
        'Desktop',
        'iscat_simulation.mp4'
    ),

    # --- MASK GENERATION ---
    "mask_generation_enabled": True,  # Set to True to generate masks
    "mask_output_directory": os.path.join(
        os.path.expanduser('~'),
        'Desktop',
        'iscat_masks'
    ),                                # Base folder for masks
    "mask_threshold": 0.3,            # Threshold for binarizing the mask (0.0 to 1.0 of particle's max signal)

    # --- OPTICAL SETUP ---
    "wavelength_nm": 635,             # Illumination wavelength in vacuum (nm)
    "numerical_aperture": 1.4,        # Objective's numerical aperture (NA)
    "magnification": 60,              # For reference: objective magnification (e.g., 60x)
    "objective_focal_length_mm": 3.0, # For a 60x objective with a standard 180mm tube lens (180/60=3)
    "refractive_index_medium": 1.33,  # Refractive index of the sample medium (e.g., water)
    "refractive_index_immersion": 1.518,  # Refractive index of immersion oil

    # --- PARTICLE PROPERTIES ---
    "num_particles": 2,
    "particle_diameters_nm": [60, 100],  # Diameter of each particle in nm
    "particle_refractive_indices": [     # Complex refractive index of each particle type
        0.166 + 3.15j,  # Gold (Au) at 635 nm
        0.166 + 3.15j,
    ],
    # "particle_materials": [...],       # To be added when material-based lookup is implemented
    "particle_signal_multipliers": [1.0, 1.0],  # "Fluorescence"-like signal strength control. 0=off, 1=normal.
    # "particle_initial_positions_nm": [[x1, y1, z1], [x2, y2, z2], ...], # In nm from corner

    # --- BROWNIAN MOTION ---
    "temperature_K": 298.15,          # Temperature in Kelvin (25 C)
    "viscosity_Pa_s": 0.00089,        # Viscosity of the medium (water at 25 C) in Pascal-seconds

    # --- iPSF & SCATTERING CALCULATION ---
    "psf_oversampling_factor": 2,     # Calculate PSF at higher resolution for accuracy. 1 = no oversampling.
    "pupil_samples": 512,             # Resolution of pupil function grid. Higher is more accurate.
    "z_stack_range_nm": 30000,        # Axial range for pre-computing the iPSF stack (e.g., +/- 10 um)
    "z_stack_step_nm": 50,            # Axial step size for the iPSF stack

    # --- ABERRATIONS & PUPIL FUNCTION ---
    "spherical_aberration_strength": 0.25,
    "apodization_factor": 1.8,
    "random_aberration_strength": 1.5,

    # --- INTERFERENCE, NOISE & BACKGROUND SUBTRACTION ---
    "reference_field_amplitude": 1,    # Amplitude of the reference field E_R
    "background_intensity": 100,       # Average intensity of the background (in camera counts)
    "shot_noise_enabled": True,        # Poisson noise
    "shot_noise_scaling_factor": 1.00, # Custom knob to control shot noise strength (0=off, 1=full)
    "gaussian_noise_enabled": True,    # Read noise
    "read_noise_std": 17,              # Std dev of Gaussian readout noise (in camera counts)
    # Method for converting raw signal/reference into contrast frames in post-processing:
    #   'reference_frame' -> (Signal - Reference) / Reference
    #   'video_mean'      -> Signal - mean(Signal over time)
    "background_subtraction_method": "reference_frame",

    # --- MOTION BLUR ---
    "motion_blur_enabled": True,       # Enable/disable motion blur simulation
    "motion_blur_subsamples": 4,       # Number of sub-steps per frame for motion blur. 1 = no blur.
}

# --- PHYSICAL CONSTANTS ---
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K