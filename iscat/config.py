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
    "trackability_confidence_threshold": 0.8,  # Minimum confidence in [0, 1] to keep generating masks for a particle
    "trackability_enabled": False,     # Master switch: if False, the trackability model does not gate masks or stop video early

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

    # --- PSF PLACEMENT & PADDING ---
    # Fraction of the on-axis PSF intensity at which its contribution is
    # considered negligible when deciding how much extra padding to render
    # around the field of view. Smaller values keep more PSF rings (larger
    # padding and more compute); larger values cut the PSF earlier.
    "psf_intensity_fraction_threshold": 1e-3,

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

    # --- CHIP PATTERN & SUBSTRATE ---
    # Centralized configuration for non-homogeneous, stationary chip patterns
    # (e.g., gold film with circular holes) that modify the reference field and
    # background intensity maps. When disabled or when the preset is
    # "empty_background", the behavior is identical to a uniform background.
    "chip_pattern_enabled": True,          # Master switch for chip pattern simulation
    # Model that defines the geometry / structure of the pattern.
    # Supported now: "gold_holes_v1" (gold film with circular holes on a square grid).
    # Use "none" to force a uniform background even if chip_pattern_enabled is True.
    "chip_pattern_model": "gold_holes_v1",
    # Contrast evolution model for the chip pattern. "static" means time-invariant
    # contrast; "time_dependent_v1" means the contrast linearly decaying alpha_f
    "chip_pattern_contrast_model": "time_dependent_v1",
    # Maximum fractional reduction in chip-pattern contrast for time-dependent
    # models (e.g., "time_dependent_v1"). A value of 0.0 leaves the contrast
    # unchanged; 0.5 means the contrast decays from 100% to 50% over theduration of the video. This parameter has no effect when the contrast
    # duration of the video. This parameter has no effect when the contrast  model is "static".
    "chip_pattern_contrast_amplitude": 0.5,

    # Substrate/background preset:
    #   "empty_background"       -> no chip pattern (uniform background)
    #   "default_gold_holes"     -> gold film with circular holes using the
    #                               user-provided dimensions below
    #   "lab_default_gold_holes" -> gold film with circular holes using the
    #                               canonical lab defaults (15 µm holes, 2 µm spacing,
    #                               20 nm total metal thickness)
    "chip_substrate_preset": "default_gold_holes",
    # Geometry and optical-intensity parameters for the chip pattern. For the
    # "gold_holes_v1" model these are:
    #   - hole_diameter_um: diameter of circular holes in micrometers
    #   - hole_edge_to_edge_spacing_um: gold spacing between hole edges (µm)
    #   - hole_depth_nm: depth / total metal thickness in nanometers (currently
    #                    a bookkeeping parameter for future optical refinements)
    #   - hole_intensity_factor: relative background intensity inside holes
    #   - gold_intensity_factor: relative background intensity in gold regions
    #
    # The pattern map is normalized to unit mean, so the overall brightness
    # remains controlled by "background_intensity".
    "chip_pattern_dimensions": {
        "hole_diameter_um": 15.0,
        "hole_edge_to_edge_spacing_um": 2.0,
        "hole_depth_nm": 20.0,           # 5 nm Cr + 15 nm Au is a typical total
        "hole_intensity_factor": 0.7,    # Holes slightly darker than gold (reflection geometry)
        "gold_intensity_factor": 1.0,    # Reference intensity level in gold regions
    },
}

# --- PHYSICAL CONSTANTS ---
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K