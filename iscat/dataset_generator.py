# File: dataset_generator.py

"""
High-level dataset generation wrapper for the iSCAT simulation.

This module implements the automated video generation wrapper described in
the Code Design Document (CDD Section 5.2). It sits "on top" of the core
simulation pipeline and uses the existing preset system and run_simulation
entry point to generate many videos in a single run.

Core responsibilities:
    - Select an instrument and/or experiment preset for each video.
    - Optionally apply physics-based preset randomization via experiment
      builders (e.g., 'nanoplastic_surface').
    - Ensure that each video and its corresponding masks are written to
      unique, organized output locations.
    - Invoke run_simulation(params) once per video.
    - Construct and save per-video and dataset-level metadata manifests
      describing the generated samples in a machine-readable format.

Randomness & reproducibility:
    - A dataset-level seed (random_seed) controls a master NumPy Generator.
    - For each video, a per-video seed is drawn from this master Generator.
    - That per-video seed is used to:
        * Seed the legacy global np.random RNG, which is used throughout
          the core simulation (Brownian motion, random aberrations, detector
          noise, etc.).
        * Construct a per-video Generator for experiment-level parameter
          sampling (e.g., in experiment presets like 'nanoplastic_surface').

    As a result, providing the same random_seed and the same dataset
    configuration (num_videos, presets, etc.) makes the entire dataset
    generation process fully reproducible.

The underlying simulation (config, trajectory, optics, rendering, etc.)
remains unchanged. This module only orchestrates multiple runs and adds
dataset-level metadata/manifest emission.
"""

import argparse
import os
from copy import deepcopy
from typing import Any, Dict, Optional, List

import numpy as np

from config import PARAMS
from main import run_simulation
from presets import (
    apply_instrument_preset,
    create_params_for_experiment,
    create_sam2_training_base_params,
)
from materials import lookup_refractive_index
from metadata import (
    build_video_manifest,
    save_video_manifest,
    build_dataset_index_entry,
    save_dataset_manifest,
)


def _resolve_base_output_dir(base_output_dir: Optional[str]) -> str:
    """
    Resolve the base output directory for the dataset.

    If base_output_dir is None, a default path on the user's Desktop is used:
        ~/Desktop/iscat_dataset

    The directory is created if it does not exist.

    Args:
        base_output_dir (Optional[str]): User-specified base directory or None.

    Returns:
        str: Absolute path to the base output directory.
    """
    if base_output_dir is None:
        base_output_dir = os.path.join(
            os.path.expanduser("~"),
            "Desktop",
            "iscat_dataset",
        )

    base_output_dir = os.path.abspath(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    return base_output_dir


def _build_sam2_training_params_for_video(
    video_index: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Construct a fully specified PARAMS dict for a single SAM2-training video.

    This builder implements the dataset-level orchestration described in the
    implementation plan:

        - Uses a fixed optical / acquisition configuration suitable for SAM2
          training (image size, NA, wavelength, duration, etc.).
        - Samples per-video parameters from well-defined distributions:
            * fps: 90% 30 Hz, 10% 24 Hz; exposure_time_ms = 1000 / fps.
            * duration_seconds: fixed 3.0 s.
            * num_particles: 30% -> 1, 60% -> 2, 10% -> 3.
            * particle_diameters_nm: per-video mean mu ~ U(150, 300) nm,
              per-particle Gaussian N(mu, 30^2), hard-clamped to [150, 300].
            * particle_shape_models: all "spherical".
            * particle_refractive_indices: per-particle complex n interpolated
              between PET and Gold at 635 nm with small jitter.
            * particle_materials: None (indices are explicit).
            * particle_signal_multipliers: all 0.3 (fixed).
            * chip_pattern_enabled: 70% True, 30% False; when True, use
              gold_holes_v1 with randomized hole_diameter_um and
              hole_edge_to_edge_spacing_um in the specified ranges.
            * background_subtraction_method: 60% "video_median",
              40% "reference_frame".
            * read_noise_std, background_intensity, and
              chip_pattern_contrast_amplitude sampled from the specified
              uniform ranges.
        - Leaves the core physics and rendering pipeline unchanged, and
          defaults any unspecified parameters to values from the base preset
          (which should be aligned with config.PARAMS).

    The returned dict is ready to be passed directly to run_simulation.
    """
    # Start from a dedicated SAM2 base preset so constant aspects
    # (optics, duration, NA, etc.) are centralized.
    params = create_sam2_training_base_params()

    # ---------------------------------------------------------------------
    # Constants and overrides for this dataset
    # ---------------------------------------------------------------------

    # Image geometry: always 1024 x 1024, 600 nm pixels.
    params["image_size_pixels"] = 1024
    params["pixel_size_nm"] = 600.0

    # Optics: fixed wavelength and NA for this dataset.
    params["wavelength_nm"] = 635.0
    params["numerical_aperture"] = 1.2
    params["magnification"] = 60
    params["refractive_index_medium"] = 1.33
    params["refractive_index_immersion"] = 1.515

    # Duration and bit depth.
    params["duration_seconds"] = 3.0
    params["bit_depth"] = 16

    # Mask / trackability configuration.
    params["mask_generation_enabled"] = True
    params["trackability_enabled"] = True
    params["trackability_confidence_threshold"] = 0.2

    # Motion model.
    params["z_motion_constraint_model"] = "reflective_boundary_v1"
    params["z_stack_step_nm"] = 50.0

    # Motion blur.
    params["motion_blur_enabled"] = True
    params["motion_blur_subsamples"] = 4

    # ---------------------------------------------------------------------
    # FPS and exposure-time distribution
    # ---------------------------------------------------------------------
    u_fps = float(rng.random())
    if u_fps < 0.9:
        fps = 30.0
    else:
        fps = 24.0
    params["fps"] = fps
    params["exposure_time_ms"] = 1000.0 / fps

    # ---------------------------------------------------------------------
    # Background subtraction method
    # ---------------------------------------------------------------------
    v_bg = float(rng.random())
    if v_bg < 0.6:
        params["background_subtraction_method"] = "video_median"
    else:
        params["background_subtraction_method"] = "reference_frame"

    # ---------------------------------------------------------------------
    # Number of particles per video
    # ---------------------------------------------------------------------
    w_np = float(rng.random())
    if w_np < 0.30:
        num_particles = 1
    elif w_np < 0.90:
        num_particles = 2
    else:
        num_particles = 3
    params["num_particles"] = num_particles

    # ---------------------------------------------------------------------
    # Particle shape models: spherical only for this dataset.
    # ---------------------------------------------------------------------
    params["particle_shape_models"] = ["spherical"] * num_particles

    # ---------------------------------------------------------------------
    # Particle diameters: per-video Gaussian around a uniform mean.
    # ---------------------------------------------------------------------
    mu_d = float(rng.uniform(150.0, 300.0))
    sigma_d = 30.0

    diameters: List[float] = []
    for _ in range(num_particles):
        d_raw = float(rng.normal(mu_d, sigma_d))
        if d_raw < 150.0:
            d = 150.0
        elif d_raw > 300.0:
            d = 300.0
        else:
            d = d_raw
        diameters.append(d)
    params["particle_diameters_nm"] = diameters
    # Translational diameters default to optical diameters when omitted,
    # which is what you want here.

    # ---------------------------------------------------------------------
    # Refractive indices: interpolate between PET and Gold at 635 nm.
    # ---------------------------------------------------------------------
    wavelength_nm = float(params["wavelength_nm"])

    n_pet = lookup_refractive_index("PET", wavelength_nm=wavelength_nm)
    n_gold = lookup_refractive_index("Gold", wavelength_nm=wavelength_nm)

    particle_indices: List[complex] = []
    for _ in range(num_particles):
        t = float(rng.uniform(0.0, 1.0))
        n_interp = (1.0 - t) * n_pet + t * n_gold

        # Small jitter in both real and imaginary parts to add variety.
        delta_real = float(rng.normal(0.0, 0.05))
        delta_imag = float(rng.normal(0.0, 0.2))

        n_val = complex(n_interp.real + delta_real, n_interp.imag + delta_imag)

        # Clamp to physically sensible ranges.
        real_clamped = min(max(n_val.real, 1.3), 3.0)
        imag_clamped = min(max(n_val.imag, 0.0), 4.0)

        particle_indices.append(complex(real_clamped, imag_clamped))

    params["particle_refractive_indices"] = particle_indices
    # Explicit indices override any material labels; materials are None.
    params["particle_materials"] = [None] * num_particles

    # ---------------------------------------------------------------------
    # Particle brightness: fixed nominal multiplier 0.3 for all particles.
    # ---------------------------------------------------------------------
    params["particle_signal_multipliers"] = [0.3] * num_particles

    # ---------------------------------------------------------------------
    # Chip pattern usage and geometry
    # ---------------------------------------------------------------------
    c_chip = float(rng.random())
    if c_chip < 0.7:
        chip_enabled = True
    else:
        chip_enabled = False

    params["chip_pattern_enabled"] = chip_enabled

    if chip_enabled:
        params["chip_pattern_model"] = "gold_holes_v1"
        params["chip_substrate_preset"] = "default_gold_holes"
        params["background_fluorescence_enabled"] = False

        # Start from the baseline chip dimensions from global PARAMS and
        # override the fields that vary per video.
        base_chip_dims = deepcopy(PARAMS.get("chip_pattern_dimensions", {}))
        hole_diameter_um = float(rng.uniform(15.0, 90.0))
        hole_spacing_um = float(rng.uniform(2.0, 14.0))

        base_chip_dims["hole_diameter_um"] = hole_diameter_um
        base_chip_dims["hole_edge_to_edge_spacing_um"] = hole_spacing_um
        # Keep the default 20 nm depth from the config for this dataset.
        base_chip_dims["hole_depth_nm"] = 20.0

        params["chip_pattern_dimensions"] = base_chip_dims
    else:
        params["chip_pattern_model"] = "none"
        params["chip_substrate_preset"] = "empty_background"
        params["background_fluorescence_enabled"] = False

    # Keep chip pattern randomization controls as in the base config.
    params["chip_pattern_randomization_enabled"] = PARAMS.get(
        "chip_pattern_randomization_enabled", True
    )
    params["chip_pattern_position_jitter_std_nm"] = PARAMS.get(
        "chip_pattern_position_jitter_std_nm", 50.0
    )
    params["chip_pattern_shape_regularity"] = PARAMS.get(
        "chip_pattern_shape_regularity", 0.73
    )
    params["chip_pattern_edge_perturbation_max_rel_radius"] = PARAMS.get(
        "chip_pattern_edge_perturbation_max_rel_radius", 0.12
    )
    params["chip_pattern_edge_perturbation_mode_count"] = PARAMS.get(
        "chip_pattern_edge_perturbation_mode_count", 3
    )

    # ---------------------------------------------------------------------
    # Noise and background parameters
    # ---------------------------------------------------------------------
    read_noise_std = float(rng.uniform(3.0, 9.0))
    background_intensity = float(rng.uniform(75.0, 125.0))
    chip_contrast_amp = float(rng.uniform(0.4, 0.7))

    params["read_noise_std"] = read_noise_std
    params["background_intensity"] = background_intensity
    params["chip_pattern_contrast_amplitude"] = chip_contrast_amp

    # Keep existing noise toggles and shot noise scaling from global PARAMS.
    params["shot_noise_enabled"] = PARAMS.get("shot_noise_enabled", True)
    params["shot_noise_scaling_factor"] = PARAMS.get(
        "shot_noise_scaling_factor", 1.0
    )
    params["gaussian_noise_enabled"] = PARAMS.get("gaussian_noise_enabled", True)

    # Ensure mask path is set by the caller; we do not set output_filename
    # or mask_output_directory here.

    return params


def _build_params_for_video(
    video_index: int,
    rng: np.random.Generator,
    experiment_preset: Optional[str],
    instrument_preset: Optional[str],
) -> Dict[str, Any]:
    """
    Construct a parameter dictionary for a single video, using the configured
    experiment and/or instrument presets.

    Priority:
        - If experiment_preset is provided:
            * If experiment_preset == "sam2_training", use the dedicated
              SAM2 training builder and ignore instrument_preset.
            * Else, use create_params_for_experiment(experiment_preset, rng).
              This function internally applies the appropriate instrument
              preset and performs any experiment-specific randomization.

        - Else if only instrument_preset is provided:
            Apply that instrument preset on top of a deepcopy of config.PARAMS.

        - Else:
            Use a deepcopy of config.PARAMS as-is.

    The rng argument is a per-video NumPy Generator. It is used only for
    experiment presets (via create_params_for_experiment and the SAM2 builder).
    Instrument presets and the base config do not depend on rng.

    Args:
        video_index (int): Zero-based index of the video being generated.
        rng (np.random.Generator): Per-video random number generator used for
            experiment-level parameter sampling when experiment_preset is set.
        experiment_preset (Optional[str]): Name of the experiment preset, or
            None if no experiment preset should be applied.
        instrument_preset (Optional[str]): Name of the instrument preset, or
            None if no instrument preset should be applied. Ignored when
            experiment_preset is provided.

    Returns:
        Dict[str, Any]: A fresh parameter dictionary for this video.
    """
    if experiment_preset is not None:
        if experiment_preset.strip().lower() == "sam2_training":
            params = _build_sam2_training_params_for_video(
                video_index=video_index,
                rng=rng,
            )
        else:
            params = create_params_for_experiment(experiment_preset, rng)
    elif instrument_preset is not None:
        base = deepcopy(PARAMS)
        params = apply_instrument_preset(base, instrument_preset)
    else:
        params = deepcopy(PARAMS)

    return params


def generate_dataset(
    num_videos: int,
    experiment_preset: Optional[str] = None,
    instrument_preset: Optional[str] = None,
    base_output_dir: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> None:
    """
    Generate a dataset of simulated iSCAT videos and corresponding masks.

    This function is the main programmatic entry point for dataset generation.
    It repeatedly constructs a fresh parameter dictionary for each video, sets
    unique output paths, and calls run_simulation(params). For each video it
    also constructs and saves a per-video manifest JSON file and accumulates
    a minimal dataset-level index entry.

    Preset selection:
        - If experiment_preset is provided, it is used to construct the full
          parameter dictionary for each video. Experiment presets may apply
          instrument presets internally and perform physics-based randomization
          (e.g., the 'nanoplastic_surface' preset). The special value
          'sam2_training' enables the SAM2 training dataset orchestration
          described in the CDD.
        - If experiment_preset is None but instrument_preset is provided, that
          instrument preset is applied on top of a deepcopy of config.PARAMS.
        - If both are None, a deepcopy of config.PARAMS is used as-is.

    Output layout:
        base_output_dir/
            videos/
                video_0000.mp4
                video_0001.mp4
                ...
            masks/
                video_0000/
                    particle_1/
                        frame_0000.png
                        ...
                    particle_2/
                        ...
                video_0001/
                    ...
            metadata/
                video_0000.json
                video_0001.json
                ...
            dataset_manifest.json

    This layout ensures that:
        - Each video file has a unique filename.
        - Each video's masks are isolated in their own subtree.
        - Each video has a corresponding JSON manifest describing its
          parameters and particle attributes.
        - A single dataset_manifest.json summarizes all videos for easy
          iteration by downstream ML code.

    Randomness and reproducibility:
        - A dataset-level master RNG is constructed as:
              master_rng = np.random.default_rng(random_seed)
        - For each video, a per-video integer seed is drawn from master_rng.
        - That per-video seed is used to:
              * Seed the legacy global np.random RNG, which is used by the
                core simulation (Brownian motion, random aberrations, detector
                noise, etc.).
              * Construct a per-video Generator passed to experiment presets
                for parameter sampling.

        Providing the same random_seed (and the same num_videos, presets, and
        other arguments) therefore makes the entire dataset generation process
        deterministic and reproducible.

    Args:
        num_videos (int): Number of videos to generate. Must be >= 1.
        experiment_preset (Optional[str]): Name of the experiment preset to
            use (e.g., "nanoplastic_surface" or "sam2_training"), or None. If
            provided, this takes precedence over instrument_preset.
        instrument_preset (Optional[str]): Name of the instrument preset to
            use (e.g., "60x_nikon"), or None. Ignored when experiment_preset
            is provided.
        base_output_dir (Optional[str]): Base directory under which all videos
            and masks will be written. If None, defaults to a directory named
            "iscat_dataset" on the user's Desktop.
        random_seed (Optional[int]): Optional seed for the dataset-level NumPy
            random number generator. When provided, all random choices in both
            experiment-level parameter sampling and the core simulation
            (Brownian motion, optical aberration randomness, detector noise,
            etc.) become reproducible across runs with the same configuration.

    Raises:
        ValueError: If num_videos < 1.
    """
    if num_videos <= 0:
        raise ValueError("num_videos must be a positive integer.")

    base_output_dir = _resolve_base_output_dir(base_output_dir)

    # Subdirectories for videos and masks.
    video_dir = os.path.join(base_output_dir, "videos")
    masks_root_dir = os.path.join(base_output_dir, "masks")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(masks_root_dir, exist_ok=True)

    # Dataset-level RNG used for:
    #   - Deriving per-video seeds.
    #   - Experiment-level parameter sampling (via per-video Generators).
    master_rng = np.random.default_rng(random_seed)
    # Limit seeds to a safe 32-bit range compatible with np.random.seed.
    max_seed_value = 2 ** 31

    # Accumulate dataset-level manifest entries here.
    dataset_entries: List[Dict[str, Any]] = []

    print(
        f"Generating {num_videos} video(s) "
        f"using experiment_preset={experiment_preset!r}, "
        f"instrument_preset={instrument_preset!r}..."
    )

    for video_index in range(num_videos):
        print(f"\n=== Generating video {video_index + 1} / {num_videos} ===")

        # Draw a per-video seed from the dataset-level RNG.
        video_seed = int(master_rng.integers(0, max_seed_value))

        # Seed the legacy global np.random RNG so that all internal randomness
        # in the core simulation (Brownian motion, PSF aberrations, detector
        # noise, etc.) is reproducible for this video.
        np.random.seed(video_seed)

        # Construct a per-video Generator for experiment-level parameter
        # sampling (e.g., in experiment presets).
        video_rng = np.random.default_rng(video_seed)

        # Build a fresh parameter dictionary for this video.
        params = _build_params_for_video(
            video_index=video_index,
            rng=video_rng,
            experiment_preset=experiment_preset,
            instrument_preset=instrument_preset,
        )

        # Configure unique output paths for this video.
        video_filename = os.path.join(
            video_dir,
            f"video_{video_index:04d}.mp4",
        )
        masks_dir = os.path.join(
            masks_root_dir,
            f"video_{video_index:04d}",
        )

        params["output_filename"] = video_filename
        params["mask_output_directory"] = masks_dir

        # Run the full simulation pipeline for this video.
        run_simulation(params)

        # After the simulation completes, build and save the per-video manifest.
        manifest = build_video_manifest(
            params=params,
            base_output_dir=base_output_dir,
            video_index=video_index,
            experiment_preset=experiment_preset,
            instrument_preset=instrument_preset,
            video_seed=video_seed,
        )
        manifest_path = save_video_manifest(
            manifest=manifest,
            base_output_dir=base_output_dir,
            video_index=video_index,
        )
        print(f"Saved per-video manifest to {manifest_path}")

        # Append a minimal dataset-level index entry.
        dataset_entries.append(build_dataset_index_entry(manifest))

    # After all videos are generated, write the dataset-level manifest.
    dataset_manifest_path = save_dataset_manifest(
        base_output_dir=base_output_dir,
        dataset_entries=dataset_entries,
    )
    print(f"\nDataset-level manifest written to {dataset_manifest_path}")
    print("\nDataset generation complete.")


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the dataset generation script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiple simulated iSCAT videos and corresponding masks "
            "using instrument and/or experiment presets."
        )
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate (default: 1).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Name of the experiment preset to use (e.g., 'nanoplastic_surface' "
            "or 'sam2_training'). If provided, this takes precedence over "
            "--instrument."
        ),
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help=(
            "Name of the instrument preset to use (e.g., '60x_nikon'). "
            "Ignored when --experiment is provided."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Base output directory for videos and masks. "
            "Defaults to '~/Desktop/iscat_dataset' if not provided."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional dataset-level random seed. When provided, all random "
            "choices in experiment-level parameter sampling and in the core "
            "simulation (Brownian motion, aberrations, detector noise, etc.) "
            "become reproducible across runs with the same configuration."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Command-line entry point for dataset generation.

    Examples:
        # Generate a single video using the base config.PARAMS:
        python dataset_generator.py

        # Generate 10 videos using the 'nanoplastic_surface' experiment preset:
        python dataset_generator.py --num_videos 10 --experiment nanoplastic_surface

        # Generate 30 videos using the 'sam2_training' SAM2 dataset preset:
        python dataset_generator.py --num_videos 30 --experiment sam2_training

        # Generate 5 videos using the '60x_nikon' instrument preset, writing
        # outputs under a custom directory:
        python dataset_generator.py --num_videos 5 --instrument 60x_nikon --output_dir /path/to/dataset

        # Generate a reproducible dataset with a fixed random seed:
        python dataset_generator.py --num_videos 10 --experiment sam2_training --seed 12345
    """
    args = _parse_args()
    generate_dataset(
        num_videos=args.num_videos,
        experiment_preset=args.experiment,
        instrument_preset=args.instrument,
        base_output_dir=args.output_dir,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()