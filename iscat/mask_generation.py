import os
import cv2
import numpy as np


def generate_binary_mask(contrast_image: np.ndarray, mask_threshold: float) -> np.ndarray:
    """
    Legacy binary mask generator based on magnitude thresholding.

    This function is retained for compatibility but is no longer used by the
    main rendering pipeline. It normalizes the absolute value of the contrast
    image by its maximum and then thresholds it.

    Args:
        contrast_image (np.ndarray): 2D array representing the per-particle
            contrast image at the final (non-oversampled) resolution.
        mask_threshold (float): Threshold in [0, 1] applied to the normalized
            contrast magnitude.

    Returns:
        np.ndarray: A uint8 binary mask with values 0 or 255.
    """
    max_val = np.max(np.abs(contrast_image))

    if max_val > 1e-9:
        normalized_contrast = np.abs(contrast_image) / max_val
        mask = (normalized_contrast > mask_threshold).astype(np.uint8) * 255
    else:
        mask = np.zeros_like(contrast_image, dtype=np.uint8)

    return mask


def _compute_central_lobe_radius_pixels(
    contrast_image: np.ndarray,
    center_yx: tuple[int, int],
    tiny_abs: float = 1e-9,
    zero_level_fraction: float = 1e-3,
) -> float:
    """
    Compute the central-lobe radius (in pixels) for a single particle in a
    contrast image, using the sign-flip definition:

        - The central lobe is the region around the particle center where the
          sign of the contrast matches the sign at the center.
        - The lobe boundary is at the smallest radius where the *ring-averaged*
          contrast changes sign relative to the center.
        - If no sign flip is detected, the boundary is defined where the ring
          contrast magnitude falls below a small fraction of the central
          magnitude.

    The algorithm operates entirely in the final image grid so that the mask
    aligns exactly with the rendered contrast image.

    Args:
        contrast_image (np.ndarray):
            2D float array of the particle-specific contrast at final
            resolution. Can be in arbitrary units; only relative sign and
            magnitude are used.
        center_yx (tuple[int, int]):
            (cy, cx) integer indices of the particle center in the contrast
            image. These must be consistent with the coordinate mapping used
            in rendering (trajectory_nm / pixel_size_nm).
        tiny_abs (float):
            Absolute threshold below which values are treated as zero to avoid
            numerical noise.
        zero_level_fraction (float):
            Fraction of the central ring-averaged magnitude used as a fallback
            cutoff when no sign flip occurs. Must be >0.

    Returns:
        float: Central-lobe boundary radius in pixels (>= 0). If the contrast
        image carries essentially no signal, returns 0.0.
    """
    img = np.asarray(contrast_image, dtype=float)
    if img.ndim != 2:
        raise ValueError("contrast_image must be a 2D array.")

    H, W = img.shape
    cy, cx = center_yx

    # Clamp center to valid range to avoid index errors at the edges.
    cy = int(np.clip(cy, 0, H - 1))
    cx = int(np.clip(cx, 0, W - 1))

    # Center value and sign.
    center_val = float(img[cy, cx])
    if abs(center_val) < tiny_abs:
        # If the center is near zero, search a small neighborhood for a
        # stronger signal to define the central sign.
        y0 = max(cy - 2, 0)
        y1 = min(cy + 3, H)
        x0 = max(cx - 2, 0)
        x1 = min(cx + 3, W)
        neighborhood = img[y0:y1, x0:x1]
        if neighborhood.size == 0:
            return 0.0
        # Find the pixel with the largest absolute contrast.
        idx_flat = np.argmax(np.abs(neighborhood))
        max_val = float(neighborhood.flat[idx_flat])
        if abs(max_val) < tiny_abs:
            # No meaningful signal in the neighborhood.
            return 0.0
        center_val = max_val

    center_sign = 1.0 if center_val >= 0.0 else -1.0

    # Build a radial map around the center.
    yy, xx = np.indices((H, W))
    dy = yy - cy
    dx = xx - cx
    r_float = np.sqrt(dx * dx + dy * dy)
    r_index = r_float.astype(np.int64)

    # If the image is extremely small, just return zero radius or one pixel.
    if r_index.size == 0:
        return 0.0

    max_ring = int(r_index.max())
    if max_ring == 0:
        # Single-pixel image or everything at center; treat as trivial lobe.
        return 0.0

    flat_vals = img.ravel()
    flat_rings = r_index.ravel()

    # Compute ring-averaged contrast as a function of integer radius.
    counts = np.bincount(flat_rings, minlength=max_ring + 1)
    sum_vals = np.bincount(flat_rings, weights=flat_vals, minlength=max_ring + 1)

    # Avoid division by zero.
    ring_mean = np.zeros(max_ring + 1, dtype=float)
    nonzero = counts > 0
    ring_mean[nonzero] = sum_vals[nonzero] / counts[nonzero]

    # Center ring magnitude.
    ring0_mag = abs(ring_mean[0])
    if ring0_mag < tiny_abs:
        # If the ring-averaged center magnitude is essentially zero but we had
        # a non-zero pixel center, we still treat the central magnitude as the
        # reference.
        ring0_mag = abs(center_val)

    if ring0_mag < tiny_abs:
        # No meaningful signal for defining a lobe.
        return 0.0

    # Threshold for considering a ring to carry significant contrast.
    mag_threshold = zero_level_fraction * ring0_mag

    # Find the smallest radius where the sign flips relative to the center.
    r_boundary_index = None
    for k in range(1, max_ring + 1):
        v = ring_mean[k]
        if abs(v) < mag_threshold:
            # Very small average; treat as effectively zero but not as a sign flip.
            continue
        current_sign = 1.0 if v >= 0.0 else -1.0
        if current_sign != center_sign:
            r_boundary_index = k
            break

    if r_boundary_index is not None:
        # Place the boundary slightly inside the ring where the sign flips.
        r_boundary = float(r_boundary_index) - 0.5
        if r_boundary < 0.0:
            r_boundary = 0.0
        return r_boundary

    # Fallback: no sign flip detected. Use the largest radius where the
    # ring-averaged magnitude is above threshold.
    significant_indices = np.where(np.abs(ring_mean) >= mag_threshold)[0]
    if significant_indices.size == 0:
        return 0.0

    k_max = int(significant_indices[-1])
    r_boundary = float(k_max) + 0.5
    return max(r_boundary, 0.0)


def generate_central_lobe_mask(
    contrast_image: np.ndarray,
    center_yx: tuple[int, int],
) -> np.ndarray:
    """
    Generate a binary central-lobe mask for a particle using the sign-flip
    definition of the central lobe, operating in the final image grid.

    The mask is defined as:

        mask(y, x) = 1  if  r(y, x) <= r_boundary
                      0  otherwise,

    where r_boundary is determined by the radius at which the ring-averaged
    contrast first flips sign relative to the contrast at the particle center,
    or by a small-amplitude fallback if no sign flip occurs.

    Edge cases:
        - If the contrast image contains no meaningful signal near the center,
          the returned mask is all zeros.
        - If the particle is near the frame boundary, the radial computation
          naturally truncates at the edge; the resulting mask is clipped by the
          image boundaries, which is consistent with the finite camera FOV.

    Args:
        contrast_image (np.ndarray):
            2D float array representing the per-particle contrast image at the
            final resolution. This should be derived from the iPSF/contrast
            generation step for that particle only (as in rendering.py).
        center_yx (tuple[int, int]):
            (cy, cx) integer pixel indices of the particle center in the
            contrast image.

    Returns:
        np.ndarray: uint8 binary mask with values 0 or 255 and the same shape
        as contrast_image.
    """
    img = np.asarray(contrast_image, dtype=float)
    if img.ndim != 2:
        raise ValueError("contrast_image must be a 2D array.")

    H, W = img.shape
    cy, cx = center_yx
    cy = int(np.clip(cy, 0, H - 1))
    cx = int(np.clip(cx, 0, W - 1))

    # Compute central-lobe radius.
    r_boundary = _compute_central_lobe_radius_pixels(img, (cy, cx))
    if r_boundary <= 0.0:
        # No detectable central lobe; return all zeros.
        return np.zeros_like(img, dtype=np.uint8)

    yy, xx = np.indices((H, W))
    dy = yy - cy
    dx = xx - cx
    r_float = np.sqrt(dx * dx + dy * dy)

    mask_bool = r_float <= r_boundary
    mask = mask_bool.astype(np.uint8) * 255
    return mask


def save_mask(
    mask: np.ndarray,
    base_mask_directory: str,
    particle_index: int,
    frame_index: int,
) -> None:
    """
    Save a single-particle mask image to disk using the established directory
    and filename conventions.

    Directory structure:
        base_mask_directory/
            particle_1/
                frame_0000.png
                frame_0001.png
                ...
            particle_2/
                ...

    Args:
        mask (np.ndarray): The binary mask image to save (uint8, 0 or 255).
        base_mask_directory (str): Root directory for all particle masks.
        particle_index (int): Zero-based particle index.
        frame_index (int): Zero-based frame index.
    """
    particle_dir = os.path.join(base_mask_directory, f"particle_{particle_index + 1}")
    os.makedirs(particle_dir, exist_ok=True)

    filename = os.path.join(particle_dir, f"frame_{frame_index:04d}.png")
    cv2.imwrite(filename, mask)


def generate_and_save_mask_for_particle(
    contrast_image: np.ndarray,
    params: dict,
    particle_index: int,
    frame_index: int,
    center_nm: tuple[float, float] | None = None,
) -> None:
    """
    High-level helper that generates and saves a central-lobe mask for a single
    particle in a single frame.

    This function now implements the PSF-based central-lobe definition in the
    Code Design Document while keeping the external interface (directories,
    filenames) unchanged.

    The mask is defined entirely from the per-particle contrast image derived
    from the particle's iPSF and reference field for that frame, not from the
    noisy final video. The central lobe is detected via ring-averaged sign
    analysis of the contrast around the particle center.

    Args:
        contrast_image (np.ndarray):
            2D contrast image for the particle at the final image resolution.
            This should be the particle-only contrast as generated in
            rendering.generate_video_and_masks (e.g., contrast_final_normalized).
        params (dict):
            Global simulation parameter dictionary (PARAMS). Must contain
            "mask_output_directory" and "pixel_size_nm". The legacy
            "mask_threshold" parameter is no longer used by this function.
        particle_index (int):
            Zero-based index of the particle.
        frame_index (int):
            Zero-based index of the frame.
        center_nm (tuple[float, float] | None):
            Optional (x_nm, y_nm) position of the particle center in nanometers
            at this frame. If provided, it is converted into pixel coordinates
            to define the center for the central-lobe computation. If None,
            the image center is used as a fallback.

    Notes:
        - The legacy magnitude thresholding behavior is preserved in
          generate_binary_mask but is not used here.
        - If the mapped center lies outside the image due to an extreme or
          noisy position, it is clamped to the nearest valid pixel.
    """
    base_mask_dir = params["mask_output_directory"]

    H, W = contrast_image.shape
    # Determine center in pixel coordinates (final resolution).
    if center_nm is not None:
        x_nm, y_nm = center_nm
        pixel_size_nm = float(params["pixel_size_nm"])
        # Map physical coordinates (x_nm, y_nm) to pixel indices. The same
        # mapping is used in rendering when placing PSFs on the canvas.
        cx = int(round(x_nm / pixel_size_nm))
        cy = int(round(y_nm / pixel_size_nm))
    else:
        # Fallback: use the image center.
        cy = H // 2
        cx = W // 2

    cy = int(np.clip(cy, 0, H - 1))
    cx = int(np.clip(cx, 0, W - 1))

    mask = generate_central_lobe_mask(contrast_image, (cy, cx))
    save_mask(mask, base_mask_dir, particle_index, frame_index)