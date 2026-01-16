from typing import Dict, List, Optional, Sequence

import numpy as np


"""
Material library and refractive index resolution utilities.

This module provides:
    - A library of common metallic/plasmonic and dielectric materials,
      each associated with a (possibly wavelength-dependent) complex
      refractive index model.
    - A lookup function that maps a material name (and optionally wavelength
      and particle size) to a complex refractive index suitable for use in
      Mie scattering.
    - A helper that resolves per-particle refractive indices from the global
      PARAMS dictionary, combining `particle_materials` and
      `particle_refractive_indices` according to the Code Design Document.

Semantics (aligned with the CDD):
    - `particle_materials`: specifies high-level material labels per particle.
    - `particle_refractive_indices`: optional complex-valued overrides.
      When provided for a particle, it overrides the material-based lookup
      for that particle.

For metals (e.g., gold, silver), this module uses simple tabulated,
wavelength-dependent optical constants (n + i k) with linear interpolation
in wavelength. For common dielectrics and biological-like materials,
it uses wavelength-independent constants, which is a good approximation
for narrow-band visible illumination.

The function interfaces are designed to be stable so that more detailed
dispersion or size-dependent models can be introduced later without
changing the rest of the codebase.
"""


# --- Material property tables -------------------------------------------------
# Canonical material names mapped to approximate *constant* complex
# refractive indices. These are used for dielectric / weakly dispersive
# materials where treating n as wavelength-independent over the visible is
# acceptable for our current use case.
#
# Values are dimensionless refractive indices n + i k.
_MATERIAL_REFRACTIVE_INDEX: Dict[str, complex] = {
    # Common dielectrics / lab-relevant materials (mostly lossless in this model)
    "pet": 1.57 + 0.0j,         # Polyethylene terephthalate
    "polyethylene": 1.51 + 0.0j,
    "polypropylene": 1.49 + 0.0j,
    "polystyrene": 1.59 + 0.0j,
    "silica": 1.46 + 0.0j,      # SiO2
    "water": 1.33 + 0.0j,
    "protein": 1.45 + 0.0j,     # Representative protein-rich material
    "lipid": 1.47 + 0.0j,       # Representative lipid-rich material
    "glass": 1.52 + 0.0j,       # Generic microscope glass
}

# Aliases for user-facing material names. Each alias maps to a canonical key
# in the internal material tables.
_MATERIAL_ALIASES: Dict[str, List[str]] = {
    "gold": [
        "gold",
        "au",
        "gold nanoparticle",
        "nanogold",
    ],
    "silver": [
        "silver",
        "ag",
        "silver nanoparticle",
        "nanosilver",
    ],
    "pet": [
        "pet",
        "polyethylene terephthalate",
        "pet plastic",
    ],
    "polyethylene": [
        "polyethylene",
        "pe",
    ],
    "polypropylene": [
        "polypropylene",
        "pp",
    ],
    "polystyrene": [
        "polystyrene",
        "ps",
    ],
    "silica": [
        "silica",
        "sio2",
        "silicon dioxide",
    ],
    "water": [
        "water",
        "h2o",
    ],
    "protein": [
        "protein",
        "proteins",
    ],
    "lipid": [
        "lipid",
        "lipids",
    ],
    "glass": [
        "glass",
        "bk7",
        "borosilicate glass",
    ],
}

# Build a mapping from lowercase alias to canonical material key.
_MATERIAL_NAME_MAP: Dict[str, str] = {}
for canonical, aliases in _MATERIAL_ALIASES.items():
    for alias in aliases:
        _MATERIAL_NAME_MAP[alias.lower()] = canonical
    # Also allow the canonical name itself.
    _MATERIAL_NAME_MAP[canonical.lower()] = canonical


# --- Wavelength-dependent data for plasmonic metals ---------------------------
# The following optical constants (n and k) are approximate values in the
# visible range for bulk materials (e.g., Johnson & Christy-type data).
# They are sufficient for realistic iSCAT-style simulations and can be
# refined later without changing any external interfaces.

# Gold (Au): wavelengths in nm and corresponding n, k values.
_GOLD_WAVELENGTHS_NM = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)
_GOLD_N = np.array([1.46, 0.97, 0.57, 0.27, 0.17], dtype=float)
_GOLD_K = np.array([1.94, 1.87, 2.37, 3.06, 3.76], dtype=float)

# Silver (Ag): wavelengths in nm and corresponding n, k values.
_SILVER_WAVELENGTHS_NM = np.array([450.0, 500.0, 550.0, 600.0, 650.0], dtype=float)
_SILVER_N = np.array([0.13, 0.13, 0.14, 0.14, 0.15], dtype=float)
_SILVER_K = np.array([2.98, 3.15, 3.35, 3.54, 3.70], dtype=float)


def _normalize_material_name(name: str) -> str:
    """
    Normalize a user-provided material name to a canonical key.

    Args:
        name (str): User-provided material name (case-insensitive).

    Returns:
        str: Canonical material key used in the internal tables.

    Raises:
        ValueError: If the material name is not recognized.
    """
    key = name.strip().lower()
    if key in _MATERIAL_NAME_MAP:
        return _MATERIAL_NAME_MAP[key]

    supported = sorted(set(_MATERIAL_NAME_MAP.values()))
    raise ValueError(
        f"Unknown particle material '{name}'. Supported materials include: {supported}"
    )


def _interp_complex_from_table(
    wavelengths_nm: np.ndarray,
    n_values: np.ndarray,
    k_values: np.ndarray,
    wavelength_nm: float,
) -> complex:
    """
    Linearly interpolate complex refractive index n + i k from tabulated data.

    For wavelengths outside the tabulated range, the nearest endpoint value
    is used (clamping).

    Args:
        wavelengths_nm (np.ndarray): 1D array of wavelengths in nanometers.
        n_values (np.ndarray): 1D array of real refractive indices at those wavelengths.
        k_values (np.ndarray): 1D array of extinction coefficients at those wavelengths.
        wavelength_nm (float): Query wavelength in nanometers.

    Returns:
        complex: Interpolated complex refractive index n + i k.
    """
    wl = float(wavelength_nm)
    w = wavelengths_nm
    n = n_values
    k = k_values

    if wl <= w[0]:
        n_interp = n[0]
        k_interp = k[0]
    elif wl >= w[-1]:
        n_interp = n[-1]
        k_interp = k[-1]
    else:
        idx = int(np.searchsorted(w, wl) - 1)
        idx = max(0, min(idx, len(w) - 2))
        wl0 = w[idx]
        wl1 = w[idx + 1]
        t = (wl - wl0) / (wl1 - wl0) if wl1 != wl0 else 0.0
        n_interp = (1.0 - t) * n[idx] + t * n[idx + 1]
        k_interp = (1.0 - t) * k[idx] + t * k[idx + 1]

    return complex(float(n_interp), float(k_interp))


def lookup_refractive_index(
    material_name: str,
    wavelength_nm: float,
    diameter_nm: Optional[float] = None,
) -> complex:
    """
    Look up the complex refractive index for a given material.

    Metals (e.g., gold, silver) are modeled with wavelength-dependent
    optical constants using small tabulated datasets with linear
    interpolation. Common dielectrics and biological-like materials are
    modeled as wavelength-independent over the visible range, which is an
    appropriate approximation for narrow-band illumination.

    Args:
        material_name (str): The name of the material, e.g., "Gold", "PET",
            "Polyethylene", "Protein". Case-insensitive; common aliases are
            accepted.
        wavelength_nm (float): Illumination wavelength in nanometers.
        diameter_nm (Optional[float]): Particle diameter in nanometers.
            (Currently unused in the constant-index and tabulated models,
             but kept in the interface for future size-dependent models.)

    Returns:
        complex: Complex refractive index n + i k for the requested material.
    """
    canonical = _normalize_material_name(material_name)

    if canonical == "gold":
        return _interp_complex_from_table(
            _GOLD_WAVELENGTHS_NM,
            _GOLD_N,
            _GOLD_K,
            wavelength_nm,
        )
    if canonical == "silver":
        return _interp_complex_from_table(
            _SILVER_WAVELENGTHS_NM,
            _SILVER_N,
            _SILVER_K,
            wavelength_nm,
        )

    if canonical in _MATERIAL_REFRACTIVE_INDEX:
        # Dielectrics / weakly dispersive materials: treat n as constant
        # over the visible; variations are small compared to metals for
        # our purposes.
        return complex(_MATERIAL_REFRACTIVE_INDEX[canonical])

    # If we ever get here, the alias map and constant table are inconsistent.
    raise ValueError(
        f"Material '{material_name}' normalized to '{canonical}', "
        "but no refractive index model is defined for this key."
    )


def resolve_particle_refractive_indices(params: dict) -> np.ndarray:
    """
    Resolve a complex refractive index for each particle using the PARAMS
    dictionary, combining material-based lookup and explicit overrides.

    Semantics:
        - If PARAMS["particle_materials"] is provided, it supplies a material
          name for each particle, which is converted into a complex refractive
          index via the material library (including wavelength-dependent
          models where appropriate).
        - If PARAMS["particle_refractive_indices"] is provided, any non-None
          entry overrides the material-based value for that particle.
        - At the end, every particle must have a defined complex refractive
          index; otherwise, a ValueError is raised.

    This function also updates PARAMS["particle_refractive_indices"] in-place to
    the resolved numpy array so that downstream code has a single, consistent
    source of truth.

    Args:
        params (dict): Global simulation parameter dictionary (PARAMS). Must
            contain:
                - "num_particles"
                - "particle_diameters_nm"
                - "wavelength_nm"
            and may optionally contain:
                - "particle_materials"
                - "particle_refractive_indices"

    Returns:
        np.ndarray: 1D array of complex refractive indices with shape
            (num_particles,), dtype=np.complex128.
    """
    num_particles = int(params["num_particles"])

    diameters: Sequence[float] = params["particle_diameters_nm"]
    if len(diameters) != num_particles:
        raise ValueError(
            "Length of PARAMS['particle_diameters_nm'] "
            f"({len(diameters)}) must match PARAMS['num_particles'] ({num_particles})."
        )

    explicit_indices = params.get("particle_refractive_indices", None)
    materials = params.get("particle_materials", None)

    wavelength_nm = float(params["wavelength_nm"])

    # Initialize per-particle refractive index list; we fill entries from
    # materials first, then apply explicit overrides.
    resolved: List[Optional[complex]] = [None] * num_particles

    # 1. Material-based lookup (if provided).
    if materials is not None:
        if len(materials) != num_particles:
            raise ValueError(
                "Length of PARAMS['particle_materials'] "
                f"({len(materials)}) must match PARAMS['num_particles'] ({num_particles})."
            )
        for i in range(num_particles):
            material_name = materials[i]
            if material_name is None:
                # Allow per-particle omission so that explicit indices can be used instead.
                continue
            resolved[i] = lookup_refractive_index(
                material_name=str(material_name),
                wavelength_nm=wavelength_nm,
                diameter_nm=float(diameters[i]),
            )

    # 2. Explicit complex refractive indices (if provided) override materials.
    if explicit_indices is not None:
        if len(explicit_indices) != num_particles:
            raise ValueError(
                "Length of PARAMS['particle_refractive_indices'] "
                f"({len(explicit_indices)}) must match PARAMS['num_particles'] ({num_particles})."
            )
        for i in range(num_particles):
            n_val = explicit_indices[i]
            if n_val is None:
                continue
            resolved[i] = complex(n_val)

    # 3. Ensure every particle has a defined refractive index.
    missing_indices = [i for i, val in enumerate(resolved) if val is None]
    if missing_indices:
        raise ValueError(
            "Refractive index is undefined for the following particle indices: "
            f"{missing_indices}. Provide either 'particle_refractive_indices' or "
            "'particle_materials' (or both, with overrides) for every particle."
        )

    # Convert to a numpy array and store back into PARAMS so that all downstream
    # code sees a consistent, resolved value.
    resolved_array = np.asarray(resolved, dtype=np.complex128)
    params["particle_refractive_indices"] = resolved_array
    return resolved_array