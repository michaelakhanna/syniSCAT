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

The underlying simulation (config, trajectory, optics, rendering, etc.)
remains unchanged. This module only orchestrates multiple runs.
"""

import argparse
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from config import PARAMS
from main import run_simulation
from presets import (
    apply_instrument_preset,
    create_params_for_experiment,
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
            Use create_params_for_experiment(experiment_preset, rng). This
            function internally applies the appropriate instrument preset and
            performs any experiment-specific randomization (e.g., exposure
            time, particle materials).

        - Else if only instrument_preset is provided:
            Apply that instrument preset on top of a deepcopy of config.PARAMS.

        - Else:
            Use a deepcopy of config.PARAMS as-is.

    Args:
        video_index (int): Zero-based index of the video being generated.
            Currently only used for debugging or future extensions.
        rng (np.random.Generator): Random number generator used for experiment
            preset sampling.
        experiment_preset (Optional[str]): Name of the experiment preset, or
            None if no experiment preset should be applied.
        instrument_preset (Optional[str]): Name of the instrument preset, or
            None if no instrument preset should be applied. Ignored when
            experiment_preset is provided.

    Returns:
        Dict[str, Any]: A fresh parameter dictionary for this video.
    """
    if experiment_preset is not None:
        # Experiment presets are the highest-level configuration. They may
        # internally apply an instrument preset and perform randomization.
        params = create_params_for_experiment(experiment_preset, rng)
    elif instrument_preset is not None:
        # Apply an instrument preset on top of a clean copy of the global
        # PARAMS template.
        base = deepcopy(PARAMS)
        params = apply_instrument_preset(base, instrument_preset)
    else:
        # No presets requested: use a clean copy of the base parameters.
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
    unique output paths, and calls run_simulation(params).

    Preset selection:
        - If experiment_preset is provided, it is used to construct the full
          parameter dictionary for each video. Experiment presets may apply
          instrument presets internally and perform physics-based randomization
          (e.g., the 'nanoplastic_surface' preset).
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

    This layout ensures that:
        - Each video file has a unique filename.
        - Each video's masks are isolated in their own subtree, avoiding any
          collisions between runs.

    Args:
        num_videos (int): Number of videos to generate. Must be >= 1.
        experiment_preset (Optional[str]): Name of the experiment preset to
            use (e.g., "nanoplastic_surface"), or None. If provided, this
            takes precedence over instrument_preset.
        instrument_preset (Optional[str]): Name of the instrument preset to
            use (e.g., "60x_nikon"), or None. Ignored when experiment_preset
            is provided.
        base_output_dir (Optional[str]): Base directory under which all videos
            and masks will be written. If None, defaults to a directory named
            "iscat_dataset" on the user's Desktop.
        random_seed (Optional[int]): Optional seed for the NumPy random number
            generator used for experiment-level parameter sampling. This does
            not control all randomness within the simulation (Brownian motion,
            optical aberration noise, etc.), which currently use the global
            NumPy RNG.

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

    # Local RNG used for experiment-level parameter sampling.
    rng = np.random.default_rng(random_seed)

    print(
        f"Generating {num_videos} video(s) "
        f"using experiment_preset={experiment_preset!r}, "
        f"instrument_preset={instrument_preset!r}..."
    )

    for video_index in range(num_videos):
        print(f"\n=== Generating video {video_index + 1} / {num_videos} ===")

        # Build a fresh parameter dictionary for this video.
        params = _build_params_for_video(
            video_index=video_index,
            rng=rng,
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
            "Name of the experiment preset to use (e.g., 'nanoplastic_surface'). "
            "If provided, this takes precedence over --instrument."
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
            "Optional seed for the NumPy random number generator used for "
            "experiment-level parameter sampling."
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

        # Generate 5 videos using the '60x_nikon' instrument preset, writing
        # outputs under a custom directory:
        python dataset_generator.py --num_videos 5 --instrument 60x_nikon --output_dir /path/to/dataset
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