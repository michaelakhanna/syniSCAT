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
    "pixel_size_nm": 244,

    # Frame rate (frames per second) of the output video.
    # Positive float or int. Combined with duration_seconds determines the
    # total number of frames: num_frames = fps * duration_seconds.
    "fps": 24,

    # Exposure time for a single frame in milliseconds. This controls the
    # temporal window over which motion blur is simulated when
    # motion_blur_enabled is True. Must satisfy:
    #
    #     0 < exposure_time_ms <= 1000 / fps
    #
    # so that the exposure window lies entirely within a single frame
    # interval. The default value below (~41.67 ms for 24 fps) corresponds
    # to a full-frame exposure and reproduces the previous behavior where
    # the particle motion was averaged over the entire frame interval.
    "exposure_time_ms": 1000.0 / 24.0,

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
    "trackability_confidence_threshold": 0.2,

    # Master switch for the Human Trackability Confidence Model.
    #   True  -> the TrackabilityModel is used to decide whether to generate
    #            masks for a particle in a given frame and to terminate the
    #            video early when all particles are considered lost.
    #   False -> masks (if enabled) are generated for all particles in every
    #            frame without gating, and the video always runs for the full
    #            duration.
    "trackability_enabled": True,

    # --- OPTICAL SETUP ---
    # Illumination wavelength in vacuum, in nanometers.
    # Positive float (e.g., 445, 520, 635).
    "wavelength_nm": 635,

    # Numerical aperture (NA) of the microscope objective.
    # Positive float; must satisfy 0 < NA <= refractive_index_immersion.
    "numerical_aperture": 1.2,

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

    # Diameter of each particle in nanometers (OPTICAL diameter).
    #
    # Semantics:
    #   - This field always defines the **optical diameter** used for:
    #       * PSF / iPSF computation and type grouping.
    #       * Optical appearance (size/brightness through scattering).
    #   - It may coincide with, but is conceptually separate from, the
    #     hydrodynamic diameter used for Brownian motion.
    #
    # Translational Brownian motion does **not** read this field directly; it
    # uses translational diameters resolved via resolve_translational_diameters_nm.
    "particle_diameters_nm": [100, 100],

    # Optional translational equivalent diameters in nanometers (HYDRODYNAMIC).
    #
    # Semantics:
    #   - When provided, this array defines the diameter used in the
    #     Stokes–Einstein equation for translational Brownian motion and in
    #     all diffusion-based models (e.g., TrackabilityModel).
    #   - It is **never** used for optical PSF sizing or scattering strength;
    #     optical appearance remains governed by particle_diameters_nm and
    #     refractive indices.
    #   - When omitted or set to None, translational diameters default to
    #     particle_diameters_nm, preserving the original coupled behavior.
    #
    # Requirements:
    #   - Must be array-like of length num_particles.
    #   - All entries must be positive.
    #
    # Example usage (uncomment and adjust as needed):
    # "particle_translational_diameters_nm": [80.0, 150.0],
    #
    # With the above example, particle 0 would diffuse as if it had an
    # 80 nm hydrodynamic diameter but still scatter optically as a 100 nm
    # sphere; particle 1 would diffuse as a 150 nm object but scatter as a
    # 200 nm sphere.
    # "particle_translational_diameters_nm": [...],

    # Complex refractive index (n + i k) of each particle.
    # List/sequence of length num_particles. Each entry can be:
    #   - A complex number, which explicitly sets the particle's refractive
    #     index and overrides any material-based lookup for that particle.
    #   - None, in which case the value will be filled via particle_materials
    #     (if provided) by resolve_particle_refractive_indices.
    #
    # In this default configuration we rely entirely on material-based lookup
    # by setting all entries to None.
    "particle_refractive_indices": [
        None,  # Use material-based lookup for this particle
        None,
    ],

    # Optional high-level material labels for each particle.
    # List/sequence of length num_particles. Each entry is a string such as
    # "Gold", "Silver", "PET", "Polyethylene", "Protein", etc. (case-insensitive).
    # When provided, material names are converted into complex refractive
    # indices using materials.lookup_refractive_index. Entries may be None
    # for particles whose refractive index is specified explicitly above.
    # If both particle_materials and particle_refractive_indices are provided,
    # explicit indices override material-based values. In this default
    # configuration we rely entirely on material-based lookup.
    "particle_materials": [
        "Gold",  # 100 nm gold nanoparticle
        "Gold",  # 60 nm gold nanoparticle
    ],

    # Per-particle scalar multipliers applied to the scattered field amplitude.
    # List/sequence of length num_particles with non-negative floats:
    #   0.0 -> particle effectively "off" (no scattered signal).
    #   1.0 -> nominal brightness.
    #  >1.0 -> brighter than nominal.
    "particle_signal_multipliers": [0.5, 0.5],

    # Optional explicit initial positions for each particle in nanometers.
    # If provided, must be an array-like of shape (num_particles, 3),
    # giving [x, y, z] for each particle. If omitted or None, positions are
    # initialized uniformly over the field of view (x, y) and within the
    # z_stack_range_nm (z), subject to the chosen z-motion constraint model.
    # "particle_initial_positions_nm": [[x1, y1, z1], [x2, y2, z2], ...],

    # Optional per-particle shape model.
    #
    # (Docstring unchanged from previous version; omitted here for brevity.)
    "particle_shape_models": [
        "spherical",   # particle 0: simple sphere
        "spherical",    # particle 1: rigid composite defined below
    ],

    # Library of named composite particle geometries.
    # (Unchanged; omitted for brevity.)
    "composite_shape_library": {
        "h2o_like": {
            "sub_particles": [
                {
                    "offset_nm": [0.0, 0.0, 0.0],
                    "diameter_nm": None,
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                },
                {
                    "offset_nm": [2400.0, 0.0, 0.0],
                    "diameter_nm": None,
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                },
                {
                    "offset_nm": [-2400.0, 0.0, 0.0],
                    "diameter_nm": None,
                    "refractive_index": None,
                    "signal_multiplier": 1.0,
                },
            ],
        },
    },

    # --- BROWNIAN MOTION ---
    "temperature_K": 298.15,
    "viscosity_Pa_s": 0.00089,
    "z_motion_constraint_model": "reflective_boundary_v1",
    "rotational_diffusion_enabled": True,
    "rotational_step_std_deg": 10.0,

    # --- iPSF & SCATTERING CALCULATION ---
    "psf_oversampling_factor": 2,
    "pupil_samples": 512,
    "z_stack_range_nm": 30500,
    "z_stack_coverage_probability": 0.999,
    "z_stack_step_nm": 50,

    # --- PSF PLACEMENT & PADDING ---
    "psf_intensity_fraction_threshold": 1e-4,

    # --- ABERRATIONS & PUPIL FUNCTION ---
    "spherical_aberration_strength": 0.25,
    "apodization_factor": 1.8,
    "random_aberration_strength": 1.5,

    # --- INTERFERENCE, NOISE & BACKGROUND SUBTRACTION ---
    "reference_field_amplitude": 1,
    "background_intensity": 100,
    "shot_noise_enabled": True,
    "shot_noise_scaling_factor": 1.00,
    "gaussian_noise_enabled": True,
    "read_noise_std": 4,
    "background_subtraction_method": "video_median",

    # --- MOTION BLUR ---
    "motion_blur_enabled": True,
    "motion_blur_subsamples": 4,

    # --- CHIP PATTERN & SUBSTRATE ---
    # Centralized configuration for chip patterns that modulate the reference
    # field and background intensity maps.
    "chip_pattern_enabled": True,

    # Chip pattern geometry model.
    "chip_pattern_model": "gold_holes_v1",

    # Contrast evolution model for the chip pattern over the duration of the
    # video.
    "chip_pattern_contrast_model": "time_dependent_v1",
    "chip_pattern_contrast_amplitude": 0.5,

    # Substrate/background preset.
    "chip_substrate_preset": "default_gold_holes",

    # Geometry and optical-intensity parameters for the chip pattern.
    "chip_pattern_dimensions": {
        # Gold-film-with-holes defaults
        "hole_diameter_um": 15.0,
        "hole_edge_to_edge_spacing_um": 2.0,
        "hole_depth_nm": 20.0,
        "hole_intensity_factor": 0.7,
        "gold_intensity_factor": 1.0,

        # Nanopillar defaults
        "pillar_diameter_um": 1.0,
        "pillar_edge_to_edge_spacing_um": 2.0,
        "pillar_height_nm": 20.0,
        "pillar_intensity_factor": 1.3,
        "background_intensity_factor": 1.0,
    },

    # Randomization controls for chip pattern imperfections.
    #
    # chip_pattern_randomization_enabled:
    #   - False:
    #       The chip pattern is perfectly periodic and features are perfect
    #       circles with no jitter or distortion. This reproduces the original
    #       behavior exactly (up to floating-point noise).
    #   - True:
    #       Each feature is jittered and slightly distorted according to the
    #       two parameters below. The same randomized layout is used both for
    #       optical background generation and for Brownian exclusion geometry.
    "chip_pattern_randomization_enabled": True,

    # Standard deviation of the positional jitter applied independently to
    # each feature center, in nanometers. This is converted internally to
    # micrometers and used to draw Gaussian offsets (dx, dy) ~ N(0, sigma^2).
    # Reasonable values are on the order of tens to a few hundred nanometers.
    "chip_pattern_position_jitter_std_nm": 50.0,

    # Dimensionless shape regularity parameter in [0.0, 1.0]:
    #   1.0 -> perfectly regular circular features (no shape distortion).
    #   0.0 -> maximum allowed distortion (bounded internally so radii remain
    #          physically reasonable, e.g., not less than ~50% of nominal).
    #
    # Internally this is mapped to a fractional radius distortion:
    #   distortion_frac = max_distortion_frac * (1 - shape_regularity)
    # and per-feature semi-axes are drawn as:
    #   r_x = nominal_radius * (1 + delta_x)
    #   r_y = nominal_radius * (1 + delta_y)
    # with delta_x, delta_y ~ Uniform(-distortion_frac, distortion_frac).
    "chip_pattern_shape_regularity": 0.73,

    # --- EDGE PERTURBATION MODEL FOR CHIP FEATURES ---
    # Maximum relative radial deviation for per-hole edge perturbations.
    #
    # Semantics:
    #   - This parameter controls the strength of local boundary roughness for
    #     individual features (currently applied to the nanohole array
    #     'gold_holes_v1').
    #   - The perturbation is expressed as a fractional deviation δ(θ) of the
    #     baseline radius as a function of angle θ, so that:
    #
    #         r_boundary(θ) = r_baseline(θ) * (1 + δ(θ))
    #
    #   - The internal sampling strategy ensures that, in typical cases,
    #     |δ(θ)| <= chip_pattern_edge_perturbation_max_rel_radius across all
    #     angles, so the perturbed radius remains within a modest band around
    #     the underlying circle/ellipse.
    #
    # Interaction with chip_pattern_shape_regularity:
    #   - The effective amplitude used per layout is:
    #
    #         effective_amp = chip_pattern_edge_perturbation_max_rel_radius
    #                         * (1 - chip_pattern_shape_regularity)
    #
    #     so that:
    #       * chip_pattern_shape_regularity = 1.0 -> perfectly smooth edges
    #         (no edge perturbation regardless of this max parameter).
    #       * chip_pattern_shape_regularity = 0.0 -> full amplitude.
    #
    # Backward compatibility:
    #   - Setting this parameter to 0.0 disables edge perturbations entirely
    #     and recovers smooth circular/elliptical boundaries.
    #
    # Recommended defaults:
    #   - Values in the range 0.05–0.10 (5–10%) produce visually apparent but
    #     still physically plausible edge irregularities for nanoholes.
    "chip_pattern_edge_perturbation_max_rel_radius": 0.12,

    # Number of angular modes used in the edge perturbation series δ(θ).
    #
    # Semantics:
    #   - δ(θ) is represented as a short cosine series:
    #
    #         δ(θ) = Σ_{k=1..K} A_k * cos(k θ + φ_k)
    #
    #     where K = chip_pattern_edge_perturbation_mode_count.
    #   - Each feature gets its own random set of coefficients {A_k, φ_k},
    #     sampled once per layout build using the same NumPy RNG as the rest
    #     of the geometry randomization.
    #
    # Performance:
    #   - K is kept small (default 3) so that classification and projection
    #     cost per point remains modest. For each candidate feature, a handful
    #     of cosine evaluations are added to the existing ellipse logic.
    #
    # Backward compatibility:
    #   - If this is set to 0, the edge perturbation model is disabled even if
    #     chip_pattern_edge_perturbation_max_rel_radius is non-zero.
    "chip_pattern_edge_perturbation_mode_count": 3,
}

# --- PHYSICAL CONSTANTS ---
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K