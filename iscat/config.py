import os

# --- SIMULATION PARAMETERS ---
# This dictionary centralizes all configurable parameters for the simulation.
PARAMS = {
    # --- IMAGE & VIDEO ---
    # Linear size (both width and height) of the square output frames in pixels.
    # Must be a positive integer (e.g., 512, 1024).
    "image_size_pixels": 1024,

    # Physical side length of a single camera pixel in nanometers.
    # Must be a positive float. Typical values are ~100–600 nm depending on the
    # objective and camera pixel pitch.
    "pixel_size_nm": 600,

    # Frame rate (frames per second) of the output video.
    # Positive float or int. Combined with duration_seconds determines the
    # total number of frames: num_frames = fps * duration_seconds.
    "fps": 24,

    # Total duration of the simulated video in seconds.
    # Positive float or int. Combined with fps determines num_frames.
    "duration_seconds": 1,

    # Bit depth of the raw simulated frames in camera counts.
    # Supported range: 1–16 (limited by uint16 storage in the current pipeline).
    # Common values: 12, 14, 16.
    "bit_depth": 16,

    # Full path (including filename) of the final encoded .mp4 video.
    # Any valid filesystem path is allowed; parent directories are created if
    # they do not exist.
    "output_filename": os.path.join(
        os.path.expanduser('~'),
        'Desktop',
        'iscat_simulation.mp4'
    ),

    # --- MASK GENERATION ---
    # Master switch for segmentation mask generation.
    #   True  -> per-particle masks are generated and saved to disk, and the
    #            trackability model (if enabled) can gate mask generation and
    #            terminate the video early.
    #   False -> no masks are generated or saved, and trackability has no
    #            effect on the video (only the frames are rendered).
    "mask_generation_enabled": True,

    # Base folder where per-particle mask images will be written. Within this
    # directory, a subfolder "particle_i" is created for each particle.
    "mask_output_directory": os.path.join(
        os.path.expanduser('~'),
        'Desktop',
        'iscat_masks'
    ),

    # Threshold for binarizing the per-particle contrast image into a mask.
    # Value in [0.0, 1.0]. The contrast image is normalized so that its maximum
    # absolute value is 1.0; pixels above mask_threshold are set to foreground.
    "mask_threshold": 0.3,

    # Minimum human-trackability confidence (in [0.0, 1.0]) required for a
    # particle to continue receiving masks when trackability is enabled.
    # If the confidence for a particle falls below this threshold, that
    # particle is marked "lost" and no further masks are generated for it.
    "trackability_confidence_threshold": 0.8,

    # Master switch for the Human Trackability Confidence Model.
    #   True  -> the TrackabilityModel is used to decide whether to generate
    #            masks for a particle in a given frame and to terminate the
    #            video early when all particles are considered lost.
    #   False -> masks (if enabled) are generated for all particles in every
    #            frame without gating, and the video always runs for the full
    #            duration.
    "trackability_enabled": False,

    # --- OPTICAL SETUP ---
    # Illumination wavelength in vacuum, in nanometers.
    # Positive float (e.g., 445, 520, 635).
    "wavelength_nm": 635,

    # Numerical aperture (NA) of the microscope objective.
    # Positive float; must satisfy 0 < NA <= refractive_index_immersion.
    "numerical_aperture": 1.4,

    # Magnification of the objective (for reference/documentation only).
    # Positive float or int (e.g., 60, 100). Not directly used in the physics
    # calculations but useful for presets and bookkeeping.
    "magnification": 60,

    # Objective focal length in millimeters.
    # For a 60x objective with a 180 mm tube lens, this is typically ~3.0 mm.
    "objective_focal_length_mm": 3.0,

    # Refractive index of the sample medium (e.g., water).
    # Positive float (e.g., 1.33 for water).
    "refractive_index_medium": 1.33,

    # Refractive index of the immersion medium used with the objective.
    # Positive float (e.g., 1.518 for standard immersion oil).
    "refractive_index_immersion": 1.518,

    # --- PARTICLE PROPERTIES ---
    # Total number of particles to simulate.
    # Positive integer. Many other particle-related arrays must have this length.
    "num_particles": 2,

    # Diameter of each particle in nanometers.
    # List or sequence of length num_particles. Each entry must be a positive
    # float or int specifying the diameter of that particle.
    "particle_diameters_nm": [60, 100],

    # Complex refractive index (n + i k) of each particle.
    # List/sequence of length num_particles. Each entry can be:
    #   - A complex number, which explicitly sets the particle's refractive
    #     index and overrides any material-based lookup for that particle.
    #   - None, in which case the value will be filled via particle_materials
    #     (if provided) by resolve_particle_refractive_indices.
    #
    # If particle_materials is omitted, every entry here must be non-None.
    "particle_refractive_indices": [
        0.166 + 3.15j,  # Gold (Au) at ~635 nm
        0.166 + 3.15j,
    ],

    # Optional high-level material labels for each particle.
    # List/sequence of length num_particles. Each entry is a string such as
    # "Gold", "Silver", "PET", "Polyethylene", "Protein", etc. (case-insensitive).
    # When provided, material names are converted into complex refractive
    # indices using materials.lookup_refractive_index. Entries may be None
    # for particles whose refractive index is specified explicitly above.
    # If both particle_materials and particle_refractive_indices are provided,
    # explicit indices override material-based values.
    # "particle_materials": [...],

    # Per-particle scalar multipliers applied to the scattered field amplitude.
    # List/sequence of length num_particles with non-negative floats:
    #   0.0 -> particle effectively "off" (no scattered signal).
    #   1.0 -> nominal brightness.
    #  >1.0 -> brighter than nominal.
    "particle_signal_multipliers": [1.0, 1.0],

    # Optional explicit initial positions for each particle in nanometers.
    # If provided, must be an array-like of shape (num_particles, 3),
    # giving [x, y, z] for each particle. If omitted or None, positions are
    # initialized uniformly over the field of view (x, y) and within the
    # z_stack_range_nm (z).
    # "particle_initial_positions_nm": [[x1, y1, z1], [x2, y2, z2], ...],

    # --- BROWNIAN MOTION ---
    # Absolute temperature of the medium in Kelvin.
    # Positive float (e.g., 298.15 for 25 °C). Used in the Stokes–Einstein
    # diffusion coefficient calculation.
    "temperature_K": 298.15,

    # Dynamic viscosity of the medium in Pascal-seconds (Pa·s).
    # Positive float (e.g., 0.00089 for water at 25 °C).
    "viscosity_Pa_s": 0.00089,

    # Model controlling the Z-axis Brownian motion (and, in future, interactions
    # with surfaces or substrates).
    #
    # Supported values:
    #   "unconstrained"
    #       -> Fully free 3D Brownian motion. This corresponds exactly to the
    #          current implementation of simulate_trajectories and preserves
    #          existing behavior.
    #
    # Additional models such as "surface_interaction_v1" may be added later;
    # for now, any value other than "unconstrained" is treated as invalid.
    "z_motion_constraint_model": "surface_interaction_v1",

    # --- iPSF & SCATTERING CALCULATION ---
    # Oversampling factor for the internal PSF/canvas resolution.
    # Positive integer. 1 = no oversampling; 2 = 2× finer grid in x and y;
    # higher values increase accuracy at the cost of computation.
    "psf_oversampling_factor": 2,

    # Linear grid size (number of samples per dimension) for the pupil function
    # in Fourier space. Positive integer, typically a power of two (e.g., 256,
    # 512, 1024). Larger values reduce spatial aliasing at higher compute cost.
    "pupil_samples": 512,

    # Total axial (z) range, in nanometers, over which the iPSF stack is
    # precomputed. The stack is centered around z = 0 and spans
    # [-z_stack_range_nm/2, +z_stack_range_nm/2].
    # Must be positive.
    "z_stack_range_nm": 30000,

    # Step size between consecutive axial positions in the iPSF stack, in nm.
    # Positive float. Smaller values increase axial resolution and cost.
    "z_stack_step_nm": 50,

    # --- PSF PLACEMENT & PADDING ---
    # Fraction of the on-axis PSF intensity at which its contribution is
    # considered negligible when deciding how much extra padding to render
    # around the field of view. Must satisfy 0 < value < 1.
    #
    # Smaller values keep more Airy/PSF rings (larger padding and more compute);
    # larger values cut the PSF earlier to save computation but risk more
    # wrap-around if set too high.
    "psf_intensity_fraction_threshold": 1e-3,

    # --- ABERRATIONS & PUPIL FUNCTION ---
    # Dimensionless strength of spherical aberration applied as a Zernike-like
    # phase term in the pupil function. Typical values are on the order of
    # 0–1; sign and magnitude control the aberration behavior.
    "spherical_aberration_strength": 0.25,

    # Apodization factor controlling the angular dependence of illumination/
    # collection in the pupil. Larger values increase attenuation away from
    # the optical axis (stronger apodization).
    "apodization_factor": 1.8,

    # Amplitude of random aberration phase terms applied in the pupil
    # function, in units of 2π. Larger values produce more random wavefront
    # distortions. Non-negative float.
    "random_aberration_strength": 1.5,

    # --- INTERFERENCE, NOISE & BACKGROUND SUBTRACTION ---
    # Amplitude of the reference field E_R used in the interference calculation.
    # Positive float. The corresponding reference intensity is proportional
    # to reference_field_amplitude**2.
    "reference_field_amplitude": 1,

    # Average background intensity in camera counts at each pixel (before
    # adding shot and read noise). Positive float; sets the scale for the
    # simulated detector.
    "background_intensity": 100,

    # Toggle for Poisson (shot) noise.
    #   True  -> simulate shot noise based on the local intensity and
    #            shot_noise_scaling_factor.
    #   False -> no Poisson noise.
    "shot_noise_enabled": True,

    # Scaling factor for the strength of Poisson (shot) noise.
    # Non-negative float:
    #   0.0 -> no shot noise contribution.
    #   1.0 -> physically realistic strength (sqrt(I) per pixel).
    #  >1.0 -> exaggerated shot noise.
    "shot_noise_scaling_factor": 1.00,

    # Toggle for Gaussian (readout) noise.
    #   True  -> add zero-mean Gaussian noise with standard deviation
    #            read_noise_std (in camera counts).
    #   False -> no Gaussian read noise.
    "gaussian_noise_enabled": True,

    # Standard deviation of Gaussian readout noise in camera counts.
    # Non-negative float; only used when gaussian_noise_enabled is True.
    "read_noise_std": 17,

    # Method for converting raw signal/reference frames into contrast frames
    # during post-processing. Supported string options:
    #   "reference_frame" ->
    #       For each frame, compute:
    #           Contrast = (Signal - Reference) / (Reference + eps)
    #       using the corresponding noisy reference frame.
    #
    #   "video_mean" (name kept for backward compatibility) ->
    #       Compute a static background B(x, y) as the per-pixel temporal
    #       median of all raw signal frames, then for each frame compute:
    #           Contrast = Signal - B
    #       This robustly removes a static background without burned-in
    #       negative trails from moving particles.
    "background_subtraction_method": "reference_frame",

    # --- MOTION BLUR ---
    # Toggle for motion blur simulation.
    #   True  -> each frame is computed as the average of motion_blur_subsamples
    #            sub-steps, using interpolated particle positions.
    #   False -> a single position per frame (no simulated motion blur).
    "motion_blur_enabled": True,

    # Number of temporal sub-steps per frame used to simulate motion blur.
    # Positive integer. Effective when motion_blur_enabled is True.
    #   1 -> no blur (single sample per frame).
    #  >1 -> increasing numbers yield smoother blur at higher cost.
    "motion_blur_subsamples": 4,

    # --- CHIP PATTERN & SUBSTRATE ---
    # Centralized configuration for non-homogeneous, stationary chip patterns
    # (e.g., gold film with circular holes) that modify the reference field and
    # background intensity maps. When disabled or when the substrate preset is
    # "empty_background", the behavior is identical to a uniform background.
    #
    #   True  -> use chip_pattern_model and chip_substrate_preset to generate
    #            spatially varying reference/background maps.
    #   False -> always use a spatially uniform background (no chip pattern).
    "chip_pattern_enabled": True,

    # Model that defines the geometry / structure of the chip pattern.
    # Currently supported string options:
    #   "gold_holes_v1" -> gold film with circular holes on a square grid.
    #   "none"          -> force a uniform background even if
    #                      chip_pattern_enabled is True.
    "chip_pattern_model": "gold_holes_v1",

    # Contrast evolution model for the chip pattern over the duration of the
    # video. Supported string options:
    #   "static"            -> pattern contrast is time-invariant.
    #   "time_dependent_v1" -> pattern contrast decays linearly in time via
    #                          chip_pattern_contrast_amplitude.
    "chip_pattern_contrast_model": "time_dependent_v1",

    # Maximum fractional reduction in chip-pattern contrast for
    # "time_dependent_v1". Value in [0.0, 1.0]:
    #   0.0 -> no decay (contrast remains constant over the video).
    #   0.5 -> contrast decays from 100% to 50% by the last frame.
    #   1.0 -> contrast decays from 100% to 0% by the last frame.
    # Has no effect when chip_pattern_contrast_model == "static".
    "chip_pattern_contrast_amplitude": 0.5,

    # Substrate/background preset selecting a specific chip/background
    # configuration. Supported string options:
    #   "empty_background"       ->
    #       No chip pattern; spatially uniform reference and background
    #       regardless of chip_pattern_model and chip_pattern_enabled.
    #
    #   "default_gold_holes"     ->
    #       Gold film with circular holes, using the user-provided geometry
    #       in chip_pattern_dimensions exactly as specified.
    #
    #   "lab_default_gold_holes" ->
    #       Gold film with circular holes, using canonical lab defaults when
    #       fields are omitted in chip_pattern_dimensions:
    #           hole_diameter_um            = 15.0
    #           hole_edge_to_edge_spacing_um = 2.0
    #           hole_depth_nm               = 20.0  (5 nm Cr + 15 nm Au)
    "chip_substrate_preset": "default_gold_holes",

    # Geometry and optical-intensity parameters for the chip pattern.
    # Used when chip_pattern_model == "gold_holes_v1".
    #
    # Fields:
    #   hole_diameter_um (float, >0):
    #       Diameter of the circular holes in micrometers.
    #
    #   hole_edge_to_edge_spacing_um (float, >=0):
    #       Gold spacing between adjacent hole edges (µm). Determines the
    #       center-to-center pitch:
    #           pitch_um = hole_diameter_um + hole_edge_to_edge_spacing_um
    #
    #   hole_depth_nm (float, >0):
    #       Total metal thickness / hole depth in nanometers. Currently used
    #       as a bookkeeping parameter for future optical refinements; not
    #       directly used in the present contrast calculations.
    #
    #   hole_intensity_factor (float, >0):
    #       Relative background intensity inside holes before normalization.
    #
    #   gold_intensity_factor (float, >0):
    #       Relative background intensity in gold regions before normalization.
    #
    # The generated pattern map is normalized to unit mean so that the global
    # brightness remains controlled by "background_intensity". Spatial
    # variations then modulate both the reference field and detector noise.
    "chip_pattern_dimensions": {
        "hole_diameter_um": 15.0,
        "hole_edge_to_edge_spacing_um": 2.0,
        "hole_depth_nm": 20.0,           # 5 nm Cr + 15 nm Au is a typical total
        "hole_intensity_factor": 0.7,    # Holes slightly darker than gold (reflection geometry)
        "gold_intensity_factor": 1.0,    # Reference intensity level in gold regions
    },
}

# --- PHYSICAL CONSTANTS ---
# Boltzmann constant (J/K), used in the Stokes–Einstein equation for
# Brownian motion:
#   D = k_B * T / (6 * pi * eta * r)
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K