"""
Metadata and manifest utilities for iSCAT dataset generation.

This module provides a small, self-contained API for constructing and
saving JSON metadata for each generated video and for the dataset as a
whole. It is used by dataset_generator.generate_dataset and does not
change the physics or rendering behavior of the simulation.

Concepts
--------
- Per-video manifest:
    A JSON file that describes:
        * The video index and IDs.
        * Paths to the .mp4 file and mask directory (relative to the
          dataset root).
        * Key simulation parameters relevant for ML training (fps,
          duration, image size, pixel size).
        * Chip/substrate and background subtraction configuration.
        * Per-particle physical and optical attributes (diameter,
          material label if provided, complex refractive index,
          signal multiplier, shape model).

- Dataset-level manifest:
    A JSON file that lists all videos in the dataset with minimal
    information needed to iterate over them (paths, presets, seeds).

Design goals
------------
- Leave the simulation pipeline unchanged: manifests are written only
  after run_simulation(params) returns for each video.
- Use only information already present in the parameter dictionary plus
  the per-video seed and preset names.
- Make paths portable by storing paths relative to the dataset root
  (base_output_dir) whenever possible.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _relative_path(base_dir: str, path: str) -> str:
    """
    Return 'path' expressed relative to 'base_dir', if possible.

    On systems where base_dir and path are on different drives or when
    relpath fails, this falls back to returning the absolute path. This
    keeps manifests robust without imposing strict requirements on how
    users specify output directories.
    """
    base_dir_abs = os.path.abspath(base_dir)
    path_abs = os.path.abspath(path)
    try:
        return os.path.relpath(path_abs, base_dir_abs)
    except ValueError:
        # E.g. Windows drive mismatch; return absolute path as a safe fallback.
        return path_abs


def _safe_float(value: Any) -> float:
    """
    Convert a numeric-like value to a plain Python float.

    This is primarily used to convert numpy scalar types (e.g., np.float64,
    np.int64) into JSON-serializable primitives.
    """
    return float(value)


def _safe_complex_to_dict(z: complex | Any) -> Dict[str, float]:
    """
    Convert a complex (or complex-like) value into a dict with 'real' and
    'imag' fields suitable for JSON serialization.

    If 'z' is not already a complex instance but supports .real and .imag,
    those attributes are used.
    """
    if isinstance(z, complex):
        real = z.real
        imag = z.imag
    else:
        # Accept numpy complex scalars, etc.
        real = getattr(z, "real", 0.0)
        imag = getattr(z, "imag", 0.0)
    return {
        "real": _safe_float(real),
        "imag": _safe_float(imag),
    }


def build_video_manifest(
    params: Dict[str, Any],
    base_output_dir: str,
    video_index: int,
    experiment_preset: str | None,
    instrument_preset: str | None,
    video_seed: int,
) -> Dict[str, Any]:
    """
    Construct a per-video manifest dictionary from the simulation parameters
    and metadata known at the dataset orchestration level.

    This function assumes:
        - run_simulation(params) has already been called for this video.
        - params["particle_refractive_indices"] has been resolved to a
          numpy array of complex values by materials.resolve_particle_refractive_indices.
        - params["output_filename"] and params["mask_output_directory"] are
          set to the paths used by the simulation.

    The returned dictionary is fully JSON-serializable and is intended to
    be written by save_video_manifest().
    """
    # Basic video-level properties
    fps = _safe_float(params["fps"])
    duration_seconds = _safe_float(params["duration_seconds"])
    num_frames = int(round(fps * duration_seconds))
    image_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = _safe_float(params["pixel_size_nm"])

    output_filename = params["output_filename"]
    mask_output_directory = params["mask_output_directory"]

    manifest: Dict[str, Any] = {
        "video_index": int(video_index),
        "experiment_preset": experiment_preset,
        "instrument_preset": instrument_preset,
        "random_seed": int(video_seed),
        "output_video_path": _relative_path(base_output_dir, output_filename),
        "mask_root_dir": _relative_path(base_output_dir, mask_output_directory),
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "image_size_pixels": image_size_pixels,
        "pixel_size_nm": pixel_size_nm,
    }

    # Chip/substrate and background-related configuration
    manifest["chip_pattern_enabled"] = bool(params.get("chip_pattern_enabled", False))
    manifest["chip_pattern_model"] = (
        str(params.get("chip_pattern_model", "none")) if "chip_pattern_model" in params else None
    )
    manifest["chip_substrate_preset"] = (
        str(params.get("chip_substrate_preset", "empty_background"))
        if "chip_substrate_preset" in params
        else None
    )
    manifest["background_subtraction_method"] = str(
        params.get("background_subtraction_method", "reference_frame")
    )
    manifest["mask_generation_enabled"] = bool(params.get("mask_generation_enabled", False))
    manifest["trackability_enabled"] = bool(params.get("trackability_enabled", False))

    # Particle-level metadata
    num_particles = int(params["num_particles"])
    diameters = params["particle_diameters_nm"]
    if len(diameters) != num_particles:
        raise ValueError(
            "Length of params['particle_diameters_nm'] "
            f"({len(diameters)}) must match params['num_particles'] ({num_particles})."
        )

    # Materials (optional)
    particle_materials = params.get("particle_materials", None)
    if particle_materials is not None and len(particle_materials) != num_particles:
        raise ValueError(
            "Length of params['particle_materials'] "
            f"({len(particle_materials)}) must match params['num_particles'] ({num_particles})."
        )

    # Refractive indices (resolved to numpy array by run_simulation)
    particle_refractive_indices = params.get("particle_refractive_indices", None)
    if particle_refractive_indices is None:
        raise ValueError(
            "params['particle_refractive_indices'] must be resolved before building a manifest."
        )
    if len(particle_refractive_indices) != num_particles:
        raise ValueError(
            "Length of params['particle_refractive_indices'] "
            f"({len(particle_refractive_indices)}) must match params['num_particles'] ({num_particles})."
        )

    # Signal multipliers
    signal_multipliers = params.get("particle_signal_multipliers", None)
    if signal_multipliers is not None and len(signal_multipliers) != num_particles:
        raise ValueError(
            "Length of params['particle_signal_multipliers'] "
            f"({len(signal_multipliers)}) must match params['num_particles'] ({num_particles})."
        )

    # Shape models (optional)
    shape_models = params.get("particle_shape_models", None)
    if shape_models is not None and len(shape_models) != num_particles:
        raise ValueError(
            "Length of params['particle_shape_models'] "
            f"({len(shape_models)}) must match params['num_particles'] ({num_particles})."
        )

    particles_meta: List[Dict[str, Any]] = []
    for i in range(num_particles):
        material_label = None
        if particle_materials is not None:
            material_label = (
                None if particle_materials[i] is None else str(particle_materials[i])
            )

        refr_idx = particle_refractive_indices[i]
        n_dict = _safe_complex_to_dict(refr_idx)

        signal_mult = None
        if signal_multipliers is not None:
            signal_mult = _safe_float(signal_multipliers[i])

        shape_model_str = None
        if shape_models is not None:
            shape_model_str = (
                None if shape_models[i] is None else str(shape_models[i])
            )

        particle_entry: Dict[str, Any] = {
            "particle_index": int(i),
            "diameter_nm": _safe_float(diameters[i]),
            "material": material_label,
            "refractive_index": n_dict,
            "signal_multiplier": signal_mult,
            "shape_model": shape_model_str,
        }
        particles_meta.append(particle_entry)

    manifest["particles"] = particles_meta

    return manifest


def save_video_manifest(
    manifest: Dict[str, Any],
    base_output_dir: str,
    video_index: int,
) -> str:
    """
    Save a per-video manifest to the dataset's metadata directory.

    The file is written as:
        <base_output_dir>/metadata/video_XXXX.json

    where XXXX is the zero-padded video index (4 digits). The parent
    'metadata' directory is created if it does not exist.

    Returns:
        str: Absolute path to the saved manifest file.
    """
    metadata_dir = os.path.join(base_output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    filename = os.path.join(metadata_dir, f"video_{video_index:04d}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return os.path.abspath(filename)


def build_dataset_index_entry(
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construct a minimal dataset-level index entry from a per-video manifest.

    This extracts only the fields needed to quickly iterate over a dataset
    and locate each video's assets, without duplicating all per-particle
    metadata. Additional fields can be added later in a backward-compatible
    way as long as existing keys are preserved.
    """
    entry: Dict[str, Any] = {
        "video_index": int(manifest["video_index"]),
        "video_id": f"video_{manifest['video_index']:04d}",
        "output_video_path": manifest["output_video_path"],
        "mask_root_dir": manifest["mask_root_dir"],
        "random_seed": int(manifest["random_seed"]),
        "experiment_preset": manifest.get("experiment_preset"),
        "instrument_preset": manifest.get("instrument_preset"),
    }
    return entry


def save_dataset_manifest(
    base_output_dir: str,
    dataset_entries: List[Dict[str, Any]],
) -> str:
    """
    Save the dataset-level manifest file listing all videos.

    The file is written as:
        <base_output_dir>/dataset_manifest.json

    The JSON structure is:

        {
          "base_output_dir": "<absolute path>",
          "num_videos": <int>,
          "videos": [ ... entries ... ]
        }

    Returns:
        str: Absolute path to the saved dataset manifest file.
    """
    if not isinstance(dataset_entries, list):
        raise TypeError("dataset_entries must be a list of per-video index entries.")

    base_output_dir_abs = os.path.abspath(base_output_dir)

    payload: Dict[str, Any] = {
        "base_output_dir": base_output_dir_abs,
        "num_videos": len(dataset_entries),
        "videos": dataset_entries,
    }

    filename = os.path.join(base_output_dir_abs, "dataset_manifest.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return filename