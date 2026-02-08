import math
import numpy as np
from typing import Dict, Tuple, Optional


# -----------------------------------------------------------------------------
# Internal feature layout structures and cache
# -----------------------------------------------------------------------------

class _LatticeFeature:
    """
    Represents a single chip-feature (hole or pillar) in pattern coordinates.

    Attributes:
        center_x_um (float): Center x-coordinate in micrometers.
        center_y_um (float): Center y-coordinate in micrometers.
        r_x_um (float): Semi-axis length along feature's local x-axis (µm).
        r_y_um (float): Semi-axis length along feature's local y-axis (µm).
        theta_rad (float): Orientation of the ellipse in radians. For now we
            use axis-aligned ellipses and set theta_rad = 0.0.
    """
    __slots__ = ("center_x_um", "center_y_um", "r_x_um", "r_y_um", "theta_rad")

    def __init__(
        self,
        center_x_um: float,
        center_y_um: float,
        r_x_um: float,
        r_y_um: float,
        theta_rad: float = 0.0,
    ) -> None:
        self.center_x_um = float(center_x_um)
        self.center_y_um = float(center_y_um)
        self.r_x_um = float(r_x_um)
        self.r_y_um = float(r_y_um)
        self.theta_rad = float(theta_rad)


class _FeatureLayout:
    """
    Single source of truth for chip-feature geometry (holes or pillars).

    This layout is:
        - Computed once per simulation (per parameter set).
        - Used by optical background generation and Brownian geometry checks.

    Attributes:
        pattern_model (str): "gold_holes_v1" or "nanopillars_v1".
        pitch_um (float): Lattice pitch in micrometers.
        nominal_radius_um (float): Nominal feature radius before distortion.
        features_by_cell (dict): Mapping (i, j) -> _LatticeFeature.
        i_min, i_max, j_min, j_max (int): Lattice index bounds that cover the
            full field-of-view (with margin) for the current run.
    """

    __slots__ = (
        "pattern_model",
        "pitch_um",
        "nominal_radius_um",
        "features_by_cell",
        "i_min",
        "i_max",
        "j_min",
        "j_max",
    )

    def __init__(
        self,
        pattern_model: str,
        pitch_um: float,
        nominal_radius_um: float,
        features_by_cell: Dict[Tuple[int, int], _LatticeFeature],
        i_min: int,
        i_max: int,
        j_min: int,
        j_max: int,
    ) -> None:
        self.pattern_model = pattern_model
        self.pitch_um = float(pitch_um)
        self.nominal_radius_um = float(nominal_radius_um)
        self.features_by_cell = features_by_cell
        self.i_min = int(i_min)
        self.i_max = int(i_max)
        self.j_min = int(j_min)
        self.j_max = int(j_max)


# Cache keyed by a simple signature so that all calls in a run share one layout.
_LAYOUT_CACHE: Dict[Tuple, _FeatureLayout] = {}


def _get_randomization_settings(params: dict) -> Tuple[bool, float, float]:
    """
    Extract and validate chip pattern randomization settings.

    Returns:
        chip_pattern_randomization_enabled (bool),
        position_jitter_std_um (float),
        shape_regularity (float)
    """
    enabled = bool(params.get("chip_pattern_randomization_enabled", False))
    jitter_nm = float(params.get("chip_pattern_position_jitter_std_nm", 0.0))
    shape_reg = float(params.get("chip_pattern_shape_regularity", 1.0))

    if jitter_nm < 0.0:
        raise ValueError(
            "PARAMS['chip_pattern_position_jitter_std_nm'] must be non-negative."
        )
    if not (0.0 <= shape_reg <= 1.0):
        raise ValueError(
            "PARAMS['chip_pattern_shape_regularity'] must be in the interval [0, 1]."
        )

    # Convert to micrometers for internal use.
    jitter_um = jitter_nm * 1e-3
    return enabled, jitter_um, shape_reg


def _compute_lattice_bounds(
    img_size_pixels: int,
    pixel_size_nm: float,
    pitch_um: float,
    oversampling_factor: float,
) -> Tuple[int, int, int, int]:
    """
    Determine the lattice index bounds (i_min, i_max, j_min, j_max) that cover
    the full field-of-view (and a margin) in pattern coordinates.

    We treat the FOV as a square of side length:
        L_nm = img_size_pixels * pixel_size_nm
        L_um = L_nm * 1e-3

    We then compute the min/max lattice indices whose nominal centers fall
    within [-L_um/2 - margin, L_um/2 + margin] in both x and y.

    A small margin of one lattice period is used so that modest jitter cannot
    produce features that affect the FOV but fall outside the bounds.
    """
    img_size_pixels = int(img_size_pixels)
    pixel_size_nm = float(pixel_size_nm)
    pitch_um = float(pitch_um)
    os_factor = float(oversampling_factor)

    if img_size_pixels <= 0 or pixel_size_nm <= 0.0 or pitch_um <= 0.0 or os_factor <= 0.0:
        raise ValueError(
            "Image size, pixel_size_nm, pitch_um, and oversampling_factor must be positive."
        )

    L_nm = img_size_pixels * pixel_size_nm
    # Use oversampling factor to ensure bounds cover the oversampled FOV area.
    L_um = (L_nm * 1e-3)  # physical FOV is independent of oversampling

    half_L = 0.5 * L_um
    margin = pitch_um  # one extra lattice period in each direction

    x_min = -half_L - margin
    x_max = half_L + margin
    y_min = -half_L - margin
    y_max = half_L + margin

    i_min = int(math.floor(x_min / pitch_um))
    i_max = int(math.ceil(x_max / pitch_um))
    j_min = int(math.floor(y_min / pitch_um))
    j_max = int(math.ceil(y_max / pitch_um))

    return i_min, i_max, j_min, j_max


def _build_feature_layout(
    params: dict,
    pattern_model: str,
    pitch_um: float,
    nominal_radius_um: float,
) -> _FeatureLayout:
    """
    Construct a randomized (or ideal) lattice feature layout for the current
    parameter set.

    Randomization is controlled by:
        - chip_pattern_randomization_enabled
        - chip_pattern_position_jitter_std_nm
        - chip_pattern_shape_regularity

    The layout is built in pattern coordinates aligned with the FOV, using the
    same centered convention as _generate_gold_hole_pattern / optical maps.
    """
    chip_enabled = bool(params.get("chip_pattern_enabled", False))
    if not chip_enabled:
        # Should not normally be called if chip pattern is disabled, but guard
        # against accidental use.
        features_by_cell: Dict[Tuple[int, int], _LatticeFeature] = {}
        return _FeatureLayout(pattern_model, pitch_um, nominal_radius_um,
                              features_by_cell, 0, -1, 0, -1)

    img_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    os_factor = float(params.get("psf_oversampling_factor", 1.0))

    i_min, i_max, j_min, j_max = _compute_lattice_bounds(
        img_size_pixels=img_size_pixels,
        pixel_size_nm=pixel_size_nm,
        pitch_um=pitch_um,
        oversampling_factor=os_factor,
    )

    randomization_enabled, jitter_std_um, shape_regularity = _get_randomization_settings(params)

    # Max fractional radius distortion. This determines how "irregular" shapes
    # can become when shape_regularity = 0.
    max_distortion_frac = 0.25  # <= 25% distortion for extreme case
    distortion_frac = max_distortion_frac * (1.0 - shape_regularity)

    features_by_cell: Dict[Tuple[int, int], _LatticeFeature] = {}

    for i in range(i_min, i_max + 1):
        center_x_nominal_um = i * pitch_um
        for j in range(j_min, j_max + 1):
            center_y_nominal_um = j * pitch_um

            if randomization_enabled:
                # Gaussian jitter in position
                dx_jitter = np.random.normal(loc=0.0, scale=jitter_std_um)
                dy_jitter = np.random.normal(loc=0.0, scale=jitter_std_um)
                center_x_um = center_x_nominal_um + dx_jitter
                center_y_um = center_y_nominal_um + dy_jitter

                if distortion_frac > 0.0:
                    delta_x = np.random.uniform(-distortion_frac, distortion_frac)
                    delta_y = np.random.uniform(-distortion_frac, distortion_frac)
                else:
                    delta_x = 0.0
                    delta_y = 0.0

                r_x_um = nominal_radius_um * (1.0 + delta_x)
                r_y_um = nominal_radius_um * (1.0 + delta_y)

                # Prevent degenerate shapes: enforce a minimum radius factor.
                min_factor = 0.5
                r_x_um = max(r_x_um, nominal_radius_um * min_factor)
                r_y_um = max(r_y_um, nominal_radius_um * min_factor)

                theta_rad = 0.0  # keep axis-aligned ellipses for now
            else:
                # Ideal periodic circles (original behavior).
                center_x_um = center_x_nominal_um
                center_y_um = center_y_nominal_um
                r_x_um = nominal_radius_um
                r_y_um = nominal_radius_um
                theta_rad = 0.0

            features_by_cell[(i, j)] = _LatticeFeature(
                center_x_um=center_x_um,
                center_y_um=center_y_um,
                r_x_um=r_x_um,
                r_y_um=r_y_um,
                theta_rad=theta_rad,
            )

    return _FeatureLayout(
        pattern_model=pattern_model,
        pitch_um=pitch_um,
        nominal_radius_um=nominal_radius_um,
        features_by_cell=features_by_cell,
        i_min=i_min,
        i_max=i_max,
        j_min=j_min,
        j_max=j_max,
    )


def _get_feature_layout_for_params(
    params: dict,
    pattern_model: str,
    pitch_um: float,
    nominal_radius_um: float,
) -> _FeatureLayout:
    """
    Retrieve (or build and cache) the feature layout corresponding to the
    current chip configuration.

    The cache key uses only values that affect geometry; if any of these
    change between runs, a new layout is built.
    """
    chip_enabled = bool(params.get("chip_pattern_enabled", False))
    if not chip_enabled:
        # No chip = no layout; return an empty layout so callers can still run.
        empty_key = ("none", 0.0, 0.0, 0, 0, 0, 0, 0.0, 1.0)
        layout = _LAYOUT_CACHE.get(empty_key)
        if layout is None:
            layout = _FeatureLayout("none", 1.0, 0.0, {}, 0, -1, 0, -1)
            _LAYOUT_CACHE[empty_key] = layout
        return layout

    img_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    os_factor = float(params.get("psf_oversampling_factor", 1.0))

    random_enabled, jitter_std_um, shape_reg = _get_randomization_settings(params)

    cache_key = (
        pattern_model,
        float(pitch_um),
        float(nominal_radius_um),
        int(img_size_pixels),
        float(pixel_size_nm),
        float(os_factor),
        bool(random_enabled),
        float(jitter_std_um),
        float(shape_reg),
    )

    layout = _LAYOUT_CACHE.get(cache_key)
    if layout is None:
        layout = _build_feature_layout(
            params=params,
            pattern_model=pattern_model,
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
        )
        _LAYOUT_CACHE[cache_key] = layout

    return layout


def _classify_point_against_layout(
    layout: _FeatureLayout,
    x_um: float,
    y_um: float,
) -> bool:
    """
    Classify a point (x_um, y_um) in pattern coordinates against a feature
    layout.

    Returns:
        inside_feature (bool): True if the point lies inside any feature
        (hole OR pillar, depending on pattern semantics).
    """
    pitch_um = layout.pitch_um
    if pitch_um <= 0.0 or not layout.features_by_cell:
        return False

    # Approximate lattice indices of the nearest feature in the ideal grid.
    i0 = int(round(x_um / pitch_um))
    j0 = int(round(y_um / pitch_um))

    inside = False

    # Check small neighborhood around (i0, j0) because jitter prevents exact
    # alignment with the ideal grid. 3x3 neighborhood is sufficient for modest
    # jitter.
    for di in (-1, 0, 1):
        i = i0 + di
        if i < layout.i_min or i > layout.i_max:
            continue
        for dj in (-1, 0, 1):
            j = j0 + dj
            if j < layout.j_min or j > layout.j_max:
                continue
            feature = layout.features_by_cell.get((i, j))
            if feature is None:
                continue

            dx = x_um - feature.center_x_um
            dy = y_um - feature.center_y_um

            if feature.theta_rad != 0.0:
                ct = math.cos(-feature.theta_rad)
                st = math.sin(-feature.theta_rad)
                ex = ct * dx - st * dy
                ey = st * dx + ct * dy
            else:
                ex = dx
                ey = dy

            if feature.r_x_um <= 0.0 or feature.r_y_um <= 0.0:
                continue

            val = (ex / feature.r_x_um) ** 2 + (ey / feature.r_y_um) ** 2
            if val <= 1.0:
                inside = True
                return inside

    return inside


def _nearest_feature_and_vector(
    layout: _FeatureLayout,
    x_um: float,
    y_um: float,
) -> Tuple[Optional[_LatticeFeature], float, float]:
    """
    Find the nearest feature center to (x_um, y_um) in pattern coordinates
    using the same local lattice neighborhood assumption as the classifier.

    Returns:
        feature (Optional[_LatticeFeature]): The nearest feature or None if
            no feature is found (should not happen in normal configurations).
        dx (float): x-offset from feature center to point (x_um - center_x_um).
        dy (float): y-offset from feature center to point.
    """
    pitch_um = layout.pitch_um
    if pitch_um <= 0.0 or not layout.features_by_cell:
        return None, 0.0, 0.0

    i0 = int(round(x_um / pitch_um))
    j0 = int(round(y_um / pitch_um))

    best_feature = None
    best_dx = 0.0
    best_dy = 0.0
    best_dist2 = float("inf")

    for di in (-1, 0, 1):
        i = i0 + di
        if i < layout.i_min or i > layout.i_max:
            continue
        for dj in (-1, 0, 1):
            j = j0 + dj
            if j < layout.j_min or j > layout.j_max:
                continue
            feature = layout.features_by_cell.get((i, j))
            if feature is None:
                continue

            dx = x_um - feature.center_x_um
            dy = y_um - feature.center_y_um
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_feature = feature
                best_dx = dx
                best_dy = dy

    return best_feature, best_dx, best_dy


# -----------------------------------------------------------------------------
# Existing parameter resolvers (unchanged except for using layout downstream)
# -----------------------------------------------------------------------------

def _generate_gold_hole_pattern(
    shape: tuple,
    pixel_size_nm: float,
    hole_diameter_um: float,
    hole_edge_to_edge_spacing_um: float,
    hole_intensity_factor: float,
    gold_intensity_factor: float,
    params: Optional[dict] = None,
) -> np.ndarray:
    """
    Generate a dimensionless intensity pattern map for a gold film with circular
    holes arranged on a square grid.

    Updated behavior:
        - When a PARAMS dictionary is provided, the function uses the shared
          feature layout (with possible randomization) so that the optical
          pattern geometry is identical to the Brownian exclusion geometry.
        - When params is None, the function falls back to the original ideal
          circular, perfectly periodic pattern for backward compatibility in
          isolated uses.

    See previous docstring for detailed geometry description.
    """
    height, width = int(shape[0]), int(shape[1])

    if height <= 0 or width <= 0:
        raise ValueError("Pattern shape must have positive height and width.")

    pixel_size_nm = float(pixel_size_nm)
    if pixel_size_nm <= 0.0:
        raise ValueError("pixel_size_nm must be positive for pattern generation.")

    hole_diameter_um = float(hole_diameter_um)
    hole_edge_to_edge_spacing_um = float(hole_edge_to_edge_spacing_um)
    if hole_diameter_um <= 0.0:
        raise ValueError("hole_diameter_um must be positive.")
    if hole_edge_to_edge_spacing_um < 0.0:
        raise ValueError("hole_edge_to_edge_spacing_um must be non-negative.")

    pitch_um = hole_diameter_um + hole_edge_to_edge_spacing_um
    radius_um = hole_diameter_um / 2.0

    if pitch_um <= 0.0:
        raise ValueError(
            "Computed pitch (hole_diameter_um + hole_edge_to_edge_spacing_um) "
            "must be positive."
        )

    hole_intensity_factor = float(hole_intensity_factor)
    gold_intensity_factor = float(gold_intensity_factor)
    if hole_intensity_factor <= 0.0 or gold_intensity_factor <= 0.0:
        raise ValueError(
            "hole_intensity_factor and gold_intensity_factor must be positive."
        )

    pixel_size_um = pixel_size_nm * 1e-3

    x_indices = np.arange(width, dtype=float)
    y_indices = np.arange(height, dtype=float)

    x_um = (x_indices - width / 2.0 + 0.5) * pixel_size_um
    y_um = (y_indices - height / 2.0 + 0.5) * pixel_size_um

    X_um, Y_um = np.meshgrid(x_um, y_um)

    if params is None:
        # Original ideal, perfectly periodic circle model (no shared layout).
        half_pitch = pitch_um / 2.0
        dx_um = (X_um + half_pitch) % pitch_um - half_pitch
        dy_um = (Y_um + half_pitch) % pitch_um - half_pitch
        r_um = np.sqrt(dx_um * dx_um + dy_um * dy_um)

        pattern = np.full((height, width), gold_intensity_factor, dtype=float)
        hole_mask = r_um <= radius_um
        pattern[hole_mask] = hole_intensity_factor
    else:
        # Use shared feature layout.
        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="gold_holes_v1",
            pitch_um=pitch_um,
            nominal_radius_um=radius_um,
        )

        pattern = np.full((height, width), gold_intensity_factor, dtype=float)

        # Vectorized classification: compute classification per pixel using
        # small local neighborhoods on the lattice.
        # For performance and clarity we loop over rows and use broadcasted calls
        # into the classifier for each pixel; the 3x3 neighborhood keeps cost
        # bounded.
        for iy in range(height):
            for ix in range(width):
                inside_hole = _classify_point_against_layout(
                    layout,
                    X_um[iy, ix],
                    Y_um[iy, ix],
                )
                if inside_hole:
                    pattern[iy, ix] = hole_intensity_factor

    mean_val = float(pattern.mean())
    if mean_val > 0.0:
        pattern /= mean_val

    return pattern


def _generate_nanopillar_pattern(
    shape: tuple,
    pixel_size_nm: float,
    pillar_diameter_um: float,
    pillar_edge_to_edge_spacing_um: float,
    pillar_intensity_factor: float,
    background_intensity_factor: float,
    params: Optional[dict] = None,
) -> np.ndarray:
    """
    Generate a dimensionless intensity pattern map for a nanopillar array.

    Updated behavior:
        - When a PARAMS dictionary is provided, the same shared feature layout
          used for Brownian dynamics is used here, so the optical pattern
          matches the exclusion geometry.
        - When params is None, falls back to the original ideal periodic
          circular pattern via the gold-hole helper.
    """
    return _generate_gold_hole_pattern(
        shape=shape,
        pixel_size_nm=pixel_size_nm,
        hole_diameter_um=pillar_diameter_um,
        hole_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
        hole_intensity_factor=pillar_intensity_factor,
        gold_intensity_factor=background_intensity_factor,
        params=params,
    )


def _resolve_gold_hole_parameters(params: dict) -> dict:
    """
    Resolve geometry and optical-intensity parameters for the gold film with
    circular holes from the global PARAMS dictionary.

    (Unchanged except for usage downstream with feature layouts.)
    """
    dims = params.get("chip_pattern_dimensions", {})
    if not isinstance(dims, dict):
        raise TypeError(
            "PARAMS['chip_pattern_dimensions'] must be a dictionary when "
            "using chip_pattern_model 'gold_holes_v1'."
        )

    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    hole_diameter_um = float(dims.get("hole_diameter_um", 15.0))
    hole_edge_to_edge_spacing_um = float(dims.get("hole_edge_to_edge_spacing_um", 2.0))
    hole_depth_nm = float(dims.get("hole_depth_nm", 20.0))  # bookkeeping only

    if hole_diameter_um <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['hole_diameter_um'] must be positive."
        )
    if hole_edge_to_edge_spacing_um < 0.0:
        raise ValueError(
            "chip_pattern_dimensions['hole_edge_to_edge_spacing_um'] must be "
            "non-negative."
        )

    pitch_um = hole_diameter_um + hole_edge_to_edge_spacing_um
    if pitch_um <= 0.0:
        raise ValueError(
            "Computed pitch (hole_diameter_um + hole_edge_to_edge_spacing_um) "
            "must be positive."
        )

    radius_um = hole_diameter_um / 2.0

    hole_intensity_factor = float(dims.get("hole_intensity_factor", 0.7))
    gold_intensity_factor = float(dims.get("gold_intensity_factor", 1.0))

    if hole_intensity_factor <= 0.0 or gold_intensity_factor <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['hole_intensity_factor'] and "
            "'gold_intensity_factor' must be positive."
        )

    return {
        "hole_diameter_um": hole_diameter_um,
        "hole_edge_to_edge_spacing_um": hole_edge_to_edge_spacing_um,
        "hole_depth_nm": hole_depth_nm,
        "hole_intensity_factor": hole_intensity_factor,
        "gold_intensity_factor": gold_intensity_factor,
        "pitch_um": pitch_um,
        "radius_um": radius_um,
        "substrate_preset": substrate_preset,
    }


def _resolve_nanopillar_parameters(params: dict) -> dict:
    """
    Resolve geometry and optical-intensity parameters for a circular nanopillar
    array from the global PARAMS dictionary.

    (Unchanged except for usage downstream with feature layouts.)
    """
    dims = params.get("chip_pattern_dimensions", {})
    if not isinstance(dims, dict):
        raise TypeError(
            "PARAMS['chip_pattern_dimensions'] must be a dictionary when "
            "using chip_pattern_model 'nanopillars_v1'."
        )

    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    pillar_diameter_um = float(dims.get("pillar_diameter_um", 1.0))
    pillar_edge_to_edge_spacing_um = float(
        dims.get("pillar_edge_to_edge_spacing_um", 2.0)
    )
    pillar_height_nm = float(dims.get("pillar_height_nm", 20.0))  # bookkeeping only

    if pillar_diameter_um <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['pillar_diameter_um'] must be positive."
        )
    if pillar_edge_to_edge_spacing_um < 0.0:
        raise ValueError(
            "chip_pattern_dimensions['pillar_edge_to_edge_spacing_um'] must be "
            "non-negative."
        )

    pitch_um = pillar_diameter_um + pillar_edge_to_edge_spacing_um
    if pitch_um <= 0.0:
        raise ValueError(
            "Computed pitch (pillar_diameter_um + pillar_edge_to_edge_spacing_um) "
            "must be positive."
        )

    radius_um = pillar_diameter_um / 2.0

    pillar_intensity_factor = float(dims.get("pillar_intensity_factor", 1.3))
    background_intensity_factor = float(dims.get("background_intensity_factor", 1.0))

    if pillar_intensity_factor <= 0.0 or background_intensity_factor <= 0.0:
        raise ValueError(
            "chip_pattern_dimensions['pillar_intensity_factor'] and "
            "'background_intensity_factor'] must be positive."
        )

    return {
        "pillar_diameter_um": pillar_diameter_um,
        "pillar_edge_to_edge_spacing_um": pillar_edge_to_edge_spacing_um,
        "pillar_height_nm": pillar_height_nm,
        "pillar_intensity_factor": pillar_intensity_factor,
        "background_intensity_factor": background_intensity_factor,
        "pitch_um": pitch_um,
        "radius_um": radius_um,
        "substrate_preset": substrate_preset,
    }


def _map_position_nm_to_gold_hole_unit_cell(
    params: dict,
    x_nm: float,
    y_nm: float,
    pitch_um: float,
) -> tuple:
    """
    (Kept for backward-compatibility; now only used to convert to centered
    pattern coordinates and to recover img_size_nm. Radius-based classification
    has been replaced by feature-layout-based classification.)
    """
    img_size_pixels = int(params["image_size_pixels"])
    pixel_size_nm = float(params["pixel_size_nm"])
    if img_size_pixels <= 0 or pixel_size_nm <= 0.0:
        raise ValueError(
            "PARAMS['image_size_pixels'] and PARAMS['pixel_size_nm'] must be "
            "positive when substrate exclusion is active."
        )

    img_size_nm = img_size_pixels * pixel_size_nm

    x_nm_centered = float(x_nm) - img_size_nm / 2.0
    y_nm_centered = float(y_nm) - img_size_nm / 2.0

    x_um = x_nm_centered * 1e-3
    y_um = y_nm_centered * 1e-3

    half_pitch = pitch_um / 2.0
    dx_um = (x_um + half_pitch) % pitch_um - half_pitch
    dy_um = (y_um + half_pitch) % pitch_um - half_pitch
    r_um = math.hypot(dx_um, dy_um)

    return dx_um, dy_um, r_um, x_um, y_um, img_size_nm


def is_position_in_chip_solid(params: dict, x_nm: float, y_nm: float) -> bool:
    """
    Determine whether a lateral position (x_nm, y_nm) lies inside a solid region
    of the configured chip/substrate pattern.

    Updated behavior:
        - Uses the shared feature layout with imperfections when available,
          ensuring the geometry matches the optical chip pattern.
        - Gold holes:
            solid = gold film (outside holes).
        - Nanopillars:
            solid = pillar interior.
    """
    chip_enabled = bool(params.get("chip_pattern_enabled", False))

    pattern_model_raw = params.get("chip_pattern_model", "none")
    pattern_model = str(pattern_model_raw).strip().lower()

    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    if (
        not chip_enabled
        or substrate_preset == "empty_background"
        or pattern_model == "none"
    ):
        return False

    # Gold film with circular holes: solid is gold (outside any hole feature).
    if pattern_model == "gold_holes_v1":
        if substrate_preset not in ("default_gold_holes", "lab_default_gold_holes"):
            return False

        geom = _resolve_gold_hole_parameters(params)
        pitch_um = geom["pitch_um"]

        # Convert (x_nm, y_nm) to centered pattern coordinates (x_um, y_um).
        _, _, _, x_um, y_um, _ = _map_position_nm_to_gold_hole_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="gold_holes_v1",
            pitch_um=pitch_um,
            nominal_radius_um=geom["radius_um"],
        )

        inside_hole = _classify_point_against_layout(layout, x_um, y_um)
        return not inside_hole

    # Nanopillars: solid is pillar interior.
    if pattern_model == "nanopillars_v1":
        if substrate_preset != "nanopillars":
            return False

        geom = _resolve_nanopillar_parameters(params)
        pitch_um = geom["pitch_um"]

        _, _, _, x_um, y_um, _ = _map_position_nm_to_gold_hole_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="nanopillars_v1",
            pitch_um=pitch_um,
            nominal_radius_um=geom["radius_um"],
        )

        inside_pillar = _classify_point_against_layout(layout, x_um, y_um)
        return inside_pillar

    return False


def project_position_to_fluid_region(params: dict, x_nm: float, y_nm: float) -> tuple:
    """
    Given a lateral position (x_nm, y_nm), project it into the nearest fluid
    region of the chip (i.e., outside solid regions) if it currently lies in a
    solid region.

    Updated behavior:
        - Uses the same feature layout (with imperfections) as the classifier.
        - Gold holes:
            solid -> gold film. We move the point into the nearest hole
            interior by projecting toward the nearest feature's center and
            placing it just inside the effective feature boundary.
        - Nanopillars:
            solid -> pillar interior. We move the point outward to just
            outside the effective pillar boundary.

    The projection is approximate for elliptical features but guaranteed to
    end in a fluid region for the current layout and thresholds.
    """
    if not is_position_in_chip_solid(params, x_nm, y_nm):
        return float(x_nm), float(y_nm)

    chip_enabled = bool(params.get("chip_pattern_enabled", False))
    pattern_model_raw = params.get("chip_pattern_model", "none")
    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")
    substrate_preset = str(substrate_preset_raw).strip().lower()

    if not chip_enabled:
        return float(x_nm), float(y_nm)

    # --- Gold film with circular holes: project from gold into nearest hole ---
    if (
        pattern_model == "gold_holes_v1"
        and substrate_preset in ("default_gold_holes", "lab_default_gold_holes")
    ):
        geom = _resolve_gold_hole_parameters(params)
        pitch_um = geom["pitch_um"]
        nominal_radius_um = geom["radius_um"]

        dx_um_cell, dy_um_cell, _, x_um, y_um, img_size_nm = _map_position_nm_to_gold_hole_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="gold_holes_v1",
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
        )

        feature, dx, dy = _nearest_feature_and_vector(layout, x_um, y_um)
        if feature is None:
            return float(x_nm), float(y_nm)

        # Use an effective radius to guarantee we end inside the feature for
        # both circular and modestly elliptical shapes.
        r_eff_um = min(feature.r_x_um, feature.r_y_um)
        dist_um = math.hypot(dx, dy)
        if dist_um == 0.0:
            # If we are exactly at the feature center (unlikely for solid region),
            # pick an arbitrary direction.
            dx = r_eff_um
            dy = 0.0
            dist_um = r_eff_um

        # Target radius just inside the feature.
        epsilon_um = 1e-3  # 1 nm
        r_target_um = max(r_eff_um - epsilon_um, 0.0)
        scale = r_target_um / dist_um

        new_x_um = feature.center_x_um + dx * scale
        new_y_um = feature.center_y_um + dy * scale

        new_x_nm_centered = new_x_um * 1e3
        new_y_nm_centered = new_y_um * 1e3

        new_x_nm = new_x_nm_centered + img_size_nm / 2.0
        new_y_nm = new_y_nm_centered + img_size_nm / 2.0

        # Safety: ensure projected position is fluid.
        if is_position_in_chip_solid(params, new_x_nm, new_y_nm):
            # As a fallback, place point at feature center minus epsilon.
            new_x_um = feature.center_x_um
            new_y_um = feature.center_y_um
            new_x_nm_centered = new_x_um * 1e3
            new_y_nm_centered = new_y_um * 1e3
            new_x_nm = new_x_nm_centered + img_size_nm / 2.0
            new_y_nm = new_y_nm_centered + img_size_nm / 2.0

        return float(new_x_nm), float(new_y_nm)

    # --- Nanopillars: project from pillar interior to background fluid ---
    if pattern_model == "nanopillars_v1" and substrate_preset == "nanopillars":
        geom = _resolve_nanopillar_parameters(params)
        pitch_um = geom["pitch_um"]
        nominal_radius_um = geom["radius_um"]

        dx_um_cell, dy_um_cell, _, x_um, y_um, img_size_nm = _map_position_nm_to_gold_hole_unit_cell(
            params, x_nm, y_nm, pitch_um
        )

        layout = _get_feature_layout_for_params(
            params=params,
            pattern_model="nanopillars_v1",
            pitch_um=pitch_um,
            nominal_radius_um=nominal_radius_um,
        )

        feature, dx, dy = _nearest_feature_and_vector(layout, x_um, y_um)
        if feature is None:
            return float(x_nm), float(y_nm)

        r_eff_um = min(feature.r_x_um, feature.r_y_um)
        dist_um = math.hypot(dx, dy)
        epsilon_um = 1e-3  # 1 nm

        if dist_um == 0.0:
            # If exactly at center, choose a direction along +x.
            new_x_um = feature.center_x_um + r_eff_um + epsilon_um
            new_y_um = feature.center_y_um
        else:
            # Move to just outside the effective radius.
            r_target_um = r_eff_um + epsilon_um
            scale = r_target_um / dist_um
            new_x_um = feature.center_x_um + dx * scale
            new_y_um = feature.center_y_um + dy * scale

        new_x_nm_centered = new_x_um * 1e3
        new_y_nm_centered = new_y_um * 1e3

        new_x_nm = new_x_nm_centered + img_size_nm / 2.0
        new_y_nm = new_y_nm_centered + img_size_nm / 2.0

        if is_position_in_chip_solid(params, new_x_nm, new_y_nm):
            # Fallback: step further outward along the same direction.
            dx2 = new_x_um - feature.center_x_um
            dy2 = new_y_um - feature.center_y_um
            norm2 = math.hypot(dx2, dy2) or 1.0
            step_um = r_eff_um
            new_x_um = feature.center_x_um + dx2 / norm2 * (r_eff_um + step_um)
            new_y_um = feature.center_y_um + dy2 / norm2 * (r_eff_um + step_um)
            new_x_nm_centered = new_x_um * 1e3
            new_y_nm_centered = new_y_um * 1e3
            new_x_nm = new_x_nm_centered + img_size_nm / 2.0
            new_y_nm = new_y_nm_centered + img_size_nm / 2.0

        return float(new_x_nm), float(new_y_nm)

    return float(x_nm), float(y_nm)


def generate_reference_and_background_maps(
    params: dict,
    fov_shape_os: tuple,
    final_fov_shape: tuple,
):
    """
    Generate stationary reference field and background intensity maps for the
    simulated field of view.

    Updated behavior:
        - When a chip pattern is enabled, the gold-hole and nanopillar pattern
          generators use the same randomized feature layout that drives
          is_position_in_chip_solid / project_position_to_fluid_region, so
          optical backgrounds and Brownian exclusion are geometrically
          consistent, including imperfections.
    """
    E_ref_amplitude = float(params["reference_field_amplitude"])
    background_intensity = float(params["background_intensity"])

    chip_enabled = bool(params.get("chip_pattern_enabled", False))

    pattern_model_raw = params.get("chip_pattern_model", "gold_holes_v1")
    substrate_preset_raw = params.get("chip_substrate_preset", "empty_background")

    pattern_model = str(pattern_model_raw).strip().lower()
    substrate_preset = str(substrate_preset_raw).strip().lower()

    use_uniform_background = (
        (not chip_enabled)
        or (substrate_preset == "empty_background")
        or (pattern_model == "none")
    )

    if use_uniform_background:
        E_ref_os = np.full(fov_shape_os, E_ref_amplitude, dtype=np.complex128)
        E_ref_final = np.full(final_fov_shape, E_ref_amplitude, dtype=np.complex128)
        background_final = np.full(final_fov_shape, background_intensity, dtype=float)
        return E_ref_os, E_ref_final, background_final

    pixel_size_nm = float(params["pixel_size_nm"])
    if pixel_size_nm <= 0.0:
        raise ValueError("PARAMS['pixel_size_nm'] must be positive.")

    os_factor = float(params.get("psf_oversampling_factor", 1.0))
    if os_factor <= 0.0:
        raise ValueError("PARAMS['psf_oversampling_factor'] must be positive.")

    if pattern_model == "gold_holes_v1":
        if substrate_preset not in ("default_gold_holes", "lab_default_gold_holes"):
            raise ValueError(
                f"Unsupported chip_substrate_preset '{substrate_preset_raw}' for "
                "chip_pattern_model 'gold_holes_v1'. Supported presets are "
                "'empty_background', 'default_gold_holes', and 'lab_default_gold_holes'."
            )

        geom = _resolve_gold_hole_parameters(params)
        hole_diameter_um = geom["hole_diameter_um"]
        hole_edge_to_edge_spacing_um = geom["hole_edge_to_edge_spacing_um"]
        hole_intensity_factor = geom["hole_intensity_factor"]
        gold_intensity_factor = geom["gold_intensity_factor"]

        pattern_final = _generate_gold_hole_pattern(
            shape=final_fov_shape,
            pixel_size_nm=pixel_size_nm,
            hole_diameter_um=hole_diameter_um,
            hole_edge_to_edge_spacing_um=hole_edge_to_edge_spacing_um,
            hole_intensity_factor=hole_intensity_factor,
            gold_intensity_factor=gold_intensity_factor,
            params=params,
        )

        pattern_os = _generate_gold_hole_pattern(
            shape=fov_shape_os,
            pixel_size_nm=pixel_size_nm / os_factor,
            hole_diameter_um=hole_diameter_um,
            hole_edge_to_edge_spacing_um=hole_edge_to_edge_spacing_um,
            hole_intensity_factor=hole_intensity_factor,
            gold_intensity_factor=gold_intensity_factor,
            params=params,
        )

    elif pattern_model == "nanopillars_v1":
        if substrate_preset != "nanopillars":
            raise ValueError(
                f"Unsupported chip_substrate_preset '{substrate_preset_raw}' for "
                "chip_pattern_model 'nanopillars_v1'. Supported presets are "
                "'empty_background' and 'nanopillars'."
            )

        geom = _resolve_nanopillar_parameters(params)
        pillar_diameter_um = geom["pillar_diameter_um"]
        pillar_edge_to_edge_spacing_um = geom["pillar_edge_to_edge_spacing_um"]
        pillar_intensity_factor = geom["pillar_intensity_factor"]
        background_intensity_factor = geom["background_intensity_factor"]

        pattern_final = _generate_nanopillar_pattern(
            shape=final_fov_shape,
            pixel_size_nm=pixel_size_nm,
            pillar_diameter_um=pillar_diameter_um,
            pillar_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
            pillar_intensity_factor=pillar_intensity_factor,
            background_intensity_factor=background_intensity_factor,
            params=params,
        )

        pattern_os = _generate_nanopillar_pattern(
            shape=fov_shape_os,
            pixel_size_nm=pixel_size_nm / os_factor,
            pillar_diameter_um=pillar_diameter_um,
            pillar_edge_to_edge_spacing_um=pillar_edge_to_edge_spacing_um,
            pillar_intensity_factor=pillar_intensity_factor,
            background_intensity_factor=background_intensity_factor,
            params=params,
        )

    else:
        raise ValueError(
            f"Unsupported chip_pattern_model '{pattern_model_raw}'. "
            "Currently supported models are 'none', 'gold_holes_v1', and 'nanopillars_v1'."
        )

    E_ref_os = (E_ref_amplitude * np.sqrt(pattern_os)).astype(np.complex128)
    E_ref_final = (E_ref_amplitude * np.sqrt(pattern_final)).astype(np.complex128)

    background_final = (background_intensity * pattern_final).astype(float)

    return E_ref_os, E_ref_final, background_final


def compute_contrast_scale_for_frame(
    params: dict,
    frame_index: int,
    num_frames: int,
) -> float:
    """
    (Unchanged from previous implementation.)
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive when computing contrast scale.")
    if frame_index < 0 or frame_index >= num_frames:
        raise ValueError(
            f"frame_index={frame_index} is out of range for num_frames={num_frames}."
        )

    model_raw = params.get("chip_pattern_contrast_model", "static")
    model = str(model_raw).strip().lower()

    if model == "static":
        return 1.0

    if model == "time_dependent_v1":
        amplitude = float(params.get("chip_pattern_contrast_amplitude", 0.5))
        if amplitude <= 0.0:
            return 1.0
        if amplitude > 1.0:
            amplitude = 1.0

        if num_frames == 1:
            t_frac = 0.0
        else:
            t_frac = frame_index / float(num_frames - 1)

        alpha = 1.0 - amplitude * t_frac
        return float(alpha)

    raise ValueError(
        f"Unsupported chip_pattern_contrast_model '{model_raw}'. "
        "Supported models are 'static' and 'time_dependent_v1'."
    )