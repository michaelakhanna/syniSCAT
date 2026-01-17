from copy import deepcopy
from typing import Any, Dict, Iterable

from config import PARAMS


"""
Preset system for configuring the iSCAT simulation.

This module implements the first layer of the preset architecture described
in the Code Design Document (CDD Section 5.1): *instrument presets*.

An instrument preset is a dictionary of parameter overrides that configures
the simulation for a specific microscope / optical setup. It is applied on
top of a "base" parameter dictionary (typically a copy of config.PARAMS),
leaving the base dictionary unchanged.

This module is intentionally limited to instrument presets only. Experiment
presets (which can build on an instrument preset and may include randomized
parameter ranges) and high-level dataset-generation wrappers will be layered
on top of this interface in future steps.

Usage examples
--------------
    from copy import deepcopy
    from config import PARAMS
    from presets import apply_instrument_preset, create_params_for_instrument
    from main import run_simulation

    # Option 1: start from the global PARAMS and apply a preset
    base_params = deepcopy(PARAMS)
    params_60x = apply_instrument_preset(base_params, "60x_nikon")
    run_simulation(params_60x)

    # Option 2: convenience helper that already starts from PARAMS
    params_100x = create_params_for_instrument("100x_custom")
    run_simulation(params_100x)

Design notes
------------
- This module does *not* modify config.PARAMS in-place. All functions return
  new dictionaries.
- Additional presets can be added by extending the INSTRUMENT_PRESETS
  dictionary; the public functions are written so they do not need to change
  when new presets are introduced.
- The numeric values in the presets are chosen to be physically reasonable
  and consistent with the current simulation capabilities. They only touch
  parameters that are already used by the existing code.
"""


# ---------------------------------------------------------------------------
# Instrument preset definitions
# ---------------------------------------------------------------------------

INSTRUMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    # Nikon 60x objective preset.
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
    "60x_nikon": {
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


# ---------------------------------------------------------------------------
# Public API
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

    Args:
        base_params (Dict[str, Any]):
            The starting parameter dictionary to which the preset will be
            applied. This dictionary is not modified.
        preset_name (str):
            Name of the instrument preset to apply. Matching is
            case-insensitive and leading/trailing whitespace is ignored.
            Valid names can be obtained from get_instrument_preset_names().

    Returns:
        Dict[str, Any]: A new parameter dictionary with the instrument preset
        applied.

    Raises:
        ValueError: If `preset_name` does not correspond to a known instrument
        preset.
    """
    if not isinstance(base_params, dict):
        raise TypeError("base_params must be a dictionary.")

    canonical = preset_name.strip().lower()
    if canonical not in INSTRUMENT_PRESETS:
        available = ", ".join(sorted(INSTRUMENT_PRESETS.keys()))
        raise ValueError(
            f"Unknown instrument preset '{preset_name}'. "
            f"Available instrument presets: {available}"
        )

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
            get_instrument_preset_names() for available options.

    Returns:
        Dict[str, Any]: A new parameter dictionary configured for the specified
        instrument.
    """
    base = deepcopy(PARAMS)
    return apply_instrument_preset(base, preset_name)