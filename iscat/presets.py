from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from config import PARAMS


"""
Preset system for configuring the iSCAT simulation.

This module implements the preset architecture described in the Code Design
Document (CDD Section 5.1) at two levels:

    - Instrument presets:
        Configure the simulation for a specific microscope / optical setup.
        These presets override optical and detector parameters such as
        wavelength, numerical aperture, pixel size, etc.

    - Experiment presets:
        Configure higher-level experimental scenarios (e.g., nanoplastic
        particles near a surface). Experiment presets may build on an
        instrument preset and can include randomized parameter choices in
        physically meaningful ranges (e.g., exposure time, particle
        materials, particle sizes).

High-level dataset-generation wrappers (CDD Section 5.2), which repeatedly
sample parameter sets to generate large training datasets, are intended to
sit on top of these presets and are not implemented in this module.

Design notes
------------
- This module does *not* modify config.PARAMS in-place. All public functions
  that apply presets return new dictionaries.
- Additional presets can be added by extending the internal preset
  definitions; the public functions are written so they do not need to change
  when new presets are introduced.
- Numeric values in presets are chosen to be physically reasonable and
  consistent with the current simulation capabilities. Presets only touch
  parameters that are already used by the existing code.
"""


# ---------------------------------------------------------------------------
# Instrument preset definitions
# ---------------------------------------------------------------------------

INSTRUMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    # 60x objective preset.
    #
    # This preset is based on the example in the CDD:
    #   - magnification: 60
    #   - pixel_size_nm: 122
    #   - wavelength_nm: 520
    #   - bit_depth: 16
    #   - refractive_index_immersion: 1.518
    #
    # The numerical aperture and focal length are chosen to be consistent with
    # a high-NA oil-immersion objective on a 180 mm tube lens. Other parameters
    # (e.g., chip pattern, particle properties) are inherited from the base
    # PARAMS dictionary unless explicitly overridden here.
    #
    # image_size_pixels is set to 512 so that, for typical chip geometries
    # (e.g., 15 µm holes with 2 µm spacing -> 17 µm pitch), the field of view
    # is ~62.5 µm wide at this pixel size. This yields ~3–4 holes across the
    # FOV for the 'lab_default_gold_holes' substrate, which matches the
    # lab-observed ~3x3 hole grid for this 60x configuration.
    "60x_nikon": {
        "image_size_pixels": 512,
        "magnification": 60,
        "objective_model": "Plan Apo 60x, NA 1.20 DIC N2 (representative)",
        "pixel_size_nm": 122.0,
        "wavelength_nm": 520.0,
        "bit_depth": 16,
        "numerical_aperture": 1.20,
        "refractive_index_medium": 1.33,   # water
        "refractive_index_immersion": 1.518,  # standard immersion oil
        "objective_focal_length_mm": 3.0,  # 180 mm tube lens / 60x
    },

    # Custom 100x objective preset.
    #
    # This preset represents a higher-magnification, shorter-wavelength
    # configuration, as described in the CDD. The chosen numbers are
    # representative and compatible with the existing simulation:
    #   - magnification: 100
    #   - wavelength_nm: 445
    #   - pixel_size_nm: 65  (e.g., 6.5 µm camera pixels / 100x)
    #   - numerical_aperture: 1.30 (<= immersion index)
    "100x_custom": {
        "magnification": 100,
        "objective_model": "Custom 100x, NA 1.30 objective",
        "pixel_size_nm": 65.0,
        "wavelength_nm": 445.0,
        "bit_depth": 16,
        "numerical_aperture": 1.30,
        "refractive_index_medium": 1.33,    # water or aqueous buffer
        "refractive_index_immersion": 1.518,
        "objective_focal_length_mm": 1.8,   # 180 mm / 100x
    },
}


# Aliases for user-facing instrument preset names.
# Each alias maps to a canonical key in INSTRUMENT_PRESETS. This decouples
# external naming (CLI flags, UI labels, etc.) from the internal dictionary
# keys and avoids fragile assumptions about case or exact spelling.
INSTRUMENT_PRESET_ALIASES: Dict[str, Iterable[str]] = {
    "60x_nikon": (
        "60x_nikon",
        "60X_NIKON",
        "60x nikon",
        "nikon_60x",
        "Nikon 60x",
    ),
    "100x_custom": (
        "100x_custom",
        "100X_CUSTOM",
        "100x custom",
        "custom_100x",
        "Custom 100x",
    ),
}


# Build a mapping from lowercase alias to canonical instrument preset key.
_INSTRUMENT_PRESET_NAME_MAP: Dict[str, str] = {}
for canonical, aliases in INSTRUMENT_PRESET_ALIASES.items():
    for alias in aliases:
        _INSTRUMENT_PRESET_NAME_MAP[alias.strip().lower()] = canonical
    # Ensure the canonical key itself is always accepted.
    _INSTRUMENT_PRESET_NAME_MAP[canonical.strip().lower()] = canonical


def _normalize_instrument_preset_name(preset_name: str) -> str:
    """
    Normalize a user-provided instrument preset name to a canonical key.

    This function centralizes all name handling for instrument presets so that
    public APIs (apply_instrument_preset, create_params_for_instrument, CLI
    wrappers, etc.) can accept flexible human-facing labels without depending
    on fragile, exact-string matches.

    Args:
        preset_name (str): User-provided preset name (case-insensitive, may
            include spaces or different separators for known aliases).

    Returns:
        str: Canonical instrument preset key used in INSTRUMENT_PRESETS.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    key = preset_name.strip().lower()
    if key in _INSTRUMENT_PRESET_NAME_MAP:
        canonical = _INSTRUMENT_PRESET_NAME_MAP[key]
        # As a safety check, ensure the canonical key actually exists.
        if canonical not in INSTRUMENT_PRESETS:
            raise ValueError(
                f"Internal preset alias map refers to unknown instrument preset "
                f"'{canonical}'. Available presets: {sorted(INSTRUMENT_PRESETS.keys())}"
            )
        return canonical

    available = ", ".join(sorted(INSTRUMENT_PRESETS.keys()))
    raise ValueError(
        f"Unknown instrument preset '{preset_name}'. "
        f"Available instrument presets: {available}"
    )


# ---------------------------------------------------------------------------
# Instrument preset public API
# ---------------------------------------------------------------------------

def get_instrument_preset_names() -> Iterable[str]:
    """
    Return an iterable of available instrument preset names.

    This is a thin wrapper around the keys of INSTRUMENT_PRESETS and is
    provided for introspection and UI building. It does not allocate any new
    data structures beyond the underlying dictionary view.

    Returns:
        Iterable[str]: Names of all defined instrument presets.
    """
    return INSTRUMENT_PRESETS.keys()


def apply_instrument_preset(base_params: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """
    Apply an instrument preset on top of a base parameter dictionary.

    This function does not modify the input dictionary in-place. Instead, it
    returns a deep copy of `base_params` with all key/value pairs from the
    specified instrument preset overlaid on top.

    The base parameter dictionary is typically a copy of config.PARAMS, but it
    can be any dictionary that follows the same structure.

    Name handling:
        - The user-supplied `preset_name` is normalized via
          `_normalize_instrument_preset_name`, which accepts common aliases
          (e.g., "60x nikon", "Nikon 60x") in addition to the canonical keys
          ("60x_nikon", "100x_custom"). This prevents fragile dependence on
          exact string matching.

    Args:
        base_params (Dict[str, Any]):
            The starting parameter dictionary to which the preset will be
            applied. This dictionary is not modified.
        preset_name (str):
            Name of the instrument preset to apply. Matching is
            case-insensitive and leading/trailing whitespace is ignored.
            Valid names can be obtained from get_instrument_preset_names()
            (canonical keys) or from documentation of the accepted aliases.

    Returns:
        Dict[str, Any]: A new parameter dictionary with the instrument preset
        applied.

    Raises:
        TypeError: If `base_params` is not a dictionary.
        ValueError: If `preset_name` does not correspond to a known instrument
        preset.
    """
    if not isinstance(base_params, dict):
        raise TypeError("base_params must be a dictionary.")

    canonical = _normalize_instrument_preset_name(preset_name)

    params_copy = deepcopy(base_params)
    overrides = INSTRUMENT_PRESETS[canonical]

    # Overlay preset values onto the copied base parameters.
    for key, value in overrides.items():
        params_copy[key] = value

    return params_copy


def create_params_for_instrument(preset_name: str) -> Dict[str, Any]:
    """
    Convenience helper that creates a fresh parameter dictionary for a given
    instrument preset starting from the global config.PARAMS template.

    This is equivalent to:

        from copy import deepcopy
        from config import PARAMS
        params = apply_instrument_preset(deepcopy(PARAMS), preset_name)

    but packaged in a single function for clarity. The returned dictionary is
    independent of the global PARAMS and can be safely modified or passed to
    run_simulation without affecting other simulations.

    Args:
        preset_name (str): Name of the instrument preset to apply. See
            get_instrument_preset_names() for canonical options; common
            aliases are also accepted (e.g., "60x nikon", "Nikon 60x").

    Returns:
        Dict[str, Any]: A new parameter dictionary configured for the specified
        instrument.
    """
    base = deepcopy(PARAMS)
    return apply_instrument_preset(base, preset_name)


def create_sam2_training_base_params() -> Dict[str, Any]:
    """
    Create a baseline PARAMS dictionary for SAM2-style spherical training videos.

    This function centralizes experiment-level constants that are truly fixed
    across the SAM2 training dataset, such as the optical preset, image size,
    duration, and basic toggles. Per-video variability (fps, particle counts,
    diameters, chip pattern usage, noise levels, etc.) is imposed by the
    dataset generator in dataset_generator._build_sam2_training_params_for_video.

    The baseline is:
        - Derived from the '60x_nikon' instrument preset for consistency with
          the existing Nikon configuration (image size is overridden to 1024).
        - Uses water as the medium and standard chip pattern defaults.
        - Enables mask generation and trackability.

    Returns:
        Dict[str, Any]: A new parameter dictionary with SAM2 baseline settings.
    """
    base = deepcopy(PARAMS)
    params = apply_instrument_preset(base, "60x_nikon")

    # Override image size to 1024x1024 for the SAM2 training dataset.
    params["image_size_pixels"] = 1024

    # Ensure duration is set to 3 seconds; fps will be overridden per video.
    params["duration_seconds"] = 3.0

    # Ensure mask generation and trackability are enabled by default.
    params["mask_generation_enabled"] = True
    params["trackability_enabled"] = True

    # Keep existing chip pattern defaults; the dataset generator will decide
    # per-video whether chip_pattern_enabled is True or False and will
    # override chip_pattern_dimensions when needed.
    return params


# ---------------------------------------------------------------------------
# Experiment preset implementations
# ---------------------------------------------------------------------------

def _build_nanoplastic_surface_experiment(
    base_params: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Construct parameters for the 'nanoplastic_surface' experiment preset.

    Semantics (aligned with CDD Section 5.1):
        - Inherits from the '100x_custom' instrument preset.
        - Uses water as the medium (refractive_index_medium = 1.33).
        - Constrains Z-motion via 'surface_interaction_v1', i.e., Brownian
          motion in the half-space z >= 0 nm with a reflecting surface at z = 0.
        - Randomizes exposure_time_ms between 1 and 2 ms, but never exceeding
          the frame interval 1000 / fps.
        - Randomizes particle_materials for each particle from:
              ["PET", "Polyethylene", "Polypropylene"]
          and clears explicit particle_refractive_indices so that material-based
          lookup is used.
        - Randomizes particle_diameters_nm for each particle within a
          physically plausible nanoplastic size range so that both diffusion
          and scattering vary across videos.

    All other parameters (e.g., chip pattern, number of particles, particle
    signal multipliers) are inherited from base_params and the instrument preset
    unless explicitly overridden here.

    Args:
        base_params (Dict[str, Any]):
            Starting parameter dictionary, typically a copy of config.PARAMS.
        rng (np.random.Generator):
            Random number generator used for drawing parameter values.

    Returns:
        Dict[str, Any]: A new parameter dictionary representing a single
        realization of the 'nanoplastic_surface' experiment.
    """
    if not isinstance(base_params, dict):
        raise TypeError("base_params must be a dictionary.")

    # 1. Apply the 100x instrument preset on top of the provided base parameters.
    params = apply_instrument_preset(base_params, "100x_custom")

    # 2. Fixed experiment-level overrides.
    # Medium is water.
    params["refractive_index_medium"] = 1.33
    # Z-motion constrained to a reflecting surface at z = 0 nm.
    params["z_motion_constraint_model"] = "surface_interaction_v1"

    # 3. Randomize exposure_time_ms in a physically valid range.
    #    Target: 1–2 ms, clamped so that exposure_time_ms <= 1000 / fps.
    fps = float(params.get("fps", 24.0))
    if fps <= 0.0:
        # Fallback to a sane default if fps is misconfigured.
        fps = 24.0

    frame_interval_ms = 1000.0 / fps  # duration of a frame in ms
    max_exposure_ms = min(2.0, frame_interval_ms)
    min_exposure_ms = min(1.0, max_exposure_ms)

    if max_exposure_ms <= 0.0:
        # Degenerate case: ensure a strictly positive exposure that does not
        # exceed the frame interval.
        max_exposure_ms = frame_interval_ms if frame_interval_ms > 0.0 else 1.0
        min_exposure_ms = max_exposure_ms

    exposure_ms = float(rng.uniform(min_exposure_ms, max_exposure_ms))
    params["exposure_time_ms"] = exposure_ms

    # 4. Determine number of particles for this experiment.
    num_particles = int(params.get("num_particles", 1))
    if num_particles <= 0:
        raise ValueError(
            "PARAMS['num_particles'] must be positive for the nanoplastic_surface experiment."
        )

    # 5. Randomize particle materials per particle from common nanoplastic
    #    dielectrics / biological-like materials.
    candidate_materials = ("PET", "Polyethylene", "Polypropylene")

    # Draw one material label for each particle.
    materials_array = rng.choice(candidate_materials, size=num_particles, replace=True)
    particle_materials = [str(m) for m in materials_array.tolist()]
    params["particle_materials"] = particle_materials

    # Ensure explicit refractive indices are cleared so that material-based
    # lookup is used for all particles.
    params["particle_refractive_indices"] = [None] * num_particles

    # 6. Randomize particle diameters within a plausible nanoplastic range.
    #
    # This range (40–150 nm) is chosen to:
    #   - Stay well within the assumptions of the current optical and
    #     Brownian-motion models.
    #   - Provide meaningful variability in both diffusion speed and iSCAT
    #     scattering amplitude across different videos and particles.
    min_diameter_nm = 40.0
    max_diameter_nm = 150.0
    if max_diameter_nm < min_diameter_nm:
        raise ValueError(
            "Internal error in nanoplastic_surface diameter range: "
            "max_diameter_nm must be >= min_diameter_nm."
        )

    diameters = rng.uniform(min_diameter_nm, max_diameter_nm, size=num_particles)
    params["particle_diameters_nm"] = [float(d) for d in diameters]

    return params


def _build_nanoplastic_surface_nonspherical_experiment(
    base_params: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Construct parameters for the 'nanoplastic_surface_nonspherical' experiment.

    This preset is a minimal extension of 'nanoplastic_surface' that exercises
    the existing composite/rotational pipeline in a controlled, visible way,
    without changing the behavior of any other presets or the base PARAMS:

        - It first builds a standard 'nanoplastic_surface' configuration
          (instrument, materials, diameters, z-model, exposure).
        - It then configures one particle (index 0) to use the 'h2o_like'
          composite shape from PARAMS['composite_shape_library'].
        - Rotational diffusion is enabled with a moderate per-frame step
          (~10 degrees) so that the orientation changes over the course of
          the video, but not so rapidly that adjacent frames look unrelated.

    The composite shape 'h2o_like' is already defined in config.PARAMS with
    sub-particle offsets on the order of the pixel size (e.g., ±2400 nm arms
    for 600 nm pixels). Combined with the existing oversampling and PSF
    placement, this makes the non-spherical appearance clearly visible in
    standard videos generated via run_simulation(params).

    All numeric ranges remain physically reasonable, and the preset does not
    modify any global state. It only returns a self-contained params dict.
    """
    if not isinstance(base_params, dict):
        raise TypeError("base_params must be a dictionary.")

    # Start from the standard nanoplastic_surface experiment.
    params = _build_nanoplastic_surface_experiment(base_params, rng)

    # Ensure we have at least one particle to assign the composite shape to.
    num_particles = int(params.get("num_particles", 1))
    if num_particles <= 0:
        raise ValueError(
            "PARAMS['num_particles'] must be positive for the nanoplastic_surface_nonspherical experiment."
        )

    # Initialize shape models: default all to 'spherical', then set particle 0
    # to the 'h2o_like' composite shape defined in config.PARAMS.
    shape_models = ["spherical"] * num_particles
    shape_models[0] = "h2o_like"
    params["particle_shape_models"] = shape_models

    # Enable rotational diffusion so that the composite particle actually
    # changes orientation over time. The rotational_step_std_deg value is
    # chosen to be moderate: small enough to keep adjacent frames related,
    # but large enough that the particle explores different orientations over
    # the duration of a typical short video.
    params["rotational_diffusion_enabled"] = True
    # If the base params already specify a step size, keep it; otherwise use 10°.
    if "rotational_step_std_deg" not in params:
        params["rotational_step_std_deg"] = 10.0

    return params


# Mapping from canonical experiment preset names to their builder functions.
_EXPERIMENT_PRESET_BUILDERS: Dict[
    str, Callable[[Dict[str, Any], np.random.Generator], Dict[str, Any]]
] = {
    "nanoplastic_surface": _build_nanoplastic_surface_experiment,
    "nanoplastic_surface_nonspherical": _build_nanoplastic_surface_nonspherical_experiment,
}


# ---------------------------------------------------------------------------
# Experiment preset public API
# ---------------------------------------------------------------------------

def get_experiment_preset_names() -> Iterable[str]:
    """
    Return an iterable of available experiment preset names.

    This is a thin wrapper around the keys of the internal experiment preset
    builder mapping and is provided for introspection and UI building.

    Returns:
        Iterable[str]: Names of all defined experiment presets.
    """
    return _EXPERIMENT_PRESET_BUILDERS.keys()


def apply_experiment_preset(
    base_params: Dict[str, Any],
    experiment_name: str,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Apply an experiment preset on top of a base parameter dictionary.

    An experiment preset may:
        - Apply a specific instrument preset (e.g., '100x_custom').
        - Override or randomize additional parameters (e.g., exposure time,
          particle materials, particle sizes, z-motion model).

    This function does not modify the input dictionary in-place. Instead, it
    passes `base_params` into the experiment builder, which is responsible for
    producing a new parameter dictionary.

    Args:
        base_params (Dict[str, Any]):
            The starting parameter dictionary. Typically a copy of config.PARAMS,
            but it can be any dictionary compatible with the simulation code.
        experiment_name (str):
            Name of the experiment preset to apply. Matching is
            case-insensitive and leading/trailing whitespace is ignored.
            Valid names can be obtained from get_experiment_preset_names().
        rng (Optional[np.random.Generator]):
            Optional NumPy random Generator. If None, a fresh default_rng()
            instance is created. Supplying a seeded generator allows
            reproducible experiment realizations.

    Returns:
        Dict[str, Any]: A new parameter dictionary configured for one
        realization of the requested experiment.

    Raises:
        TypeError: If base_params is not a dictionary.
        ValueError: If `experiment_name` does not correspond to a known
        experiment preset.
    """
    if not isinstance(base_params, dict):
        raise TypeError("base_params must be a dictionary.")

    canonical = experiment_name.strip().lower()
    if canonical not in _EXPERIMENT_PRESET_BUILDERS:
        available = ", ".join(sorted(_EXPERIMENT_PRESET_BUILDERS.keys()))
        raise ValueError(
            f"Unknown experiment preset '{experiment_name}'. "
            f"Available experiment presets: {available}"
        )

    if rng is None:
        rng = np.random.default_rng()

    builder = _EXPERIMENT_PRESET_BUILDERS[canonical]
    params = builder(base_params, rng)

    if not isinstance(params, dict):
        raise TypeError(
            f"Experiment preset builder for '{experiment_name}' did not return a dictionary."
        )

    return params


def create_params_for_experiment(
    experiment_name: str,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Convenience helper that creates a fresh parameter dictionary for a given
    experiment preset starting from the global config.PARAMS template.

    This is equivalent to:

        from copy import deepcopy
        from config import PARAMS
        from presets import apply_experiment_preset
        rng = np.random.default_rng()
        params = apply_experiment_preset(deepcopy(PARAMS), experiment_name, rng)

    but packaged in a single function for clarity. The returned dictionary is
    independent of the global PARAMS and can be safely modified or passed to
    run_simulation without affecting other simulations.

    Args:
        experiment_name (str):
            Name of the experiment preset to apply. See
            get_experiment_preset_names() for available options.
        rng (Optional[np.random.Generator]):
            Optional NumPy random Generator used for parameter sampling. If
            None, a fresh default_rng() instance is created.

    Returns:
        Dict[str, Any]: A new parameter dictionary configured for one
        realization of the specified experiment.
    """
    base = deepcopy(PARAMS)
    return apply_experiment_preset(base, experiment_name, rng)