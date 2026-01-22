import os
import warnings

import numpy as np
import cv2
from scipy.stats import norm

# Import simulation components from other files
from config import PARAMS
from materials import resolve_particle_refractive_indices
from trajectory import simulate_trajectories, stokes_einstein_diffusion_coefficient
from optics import compute_ipsf_stack
from rendering import generate_video_and_masks
from postprocessing import apply_background_subtraction, save_video

# Suppress RankWarning from numpy's polyfit, which can occur in Mie scattering calculations
warnings.filterwarnings("ignore", category=np.RankWarning)

# Hard upper bound on automatically estimated z-stack full range (in nm) to
# avoid pathological configurations creating extremely expensive iPSF stacks.
# This helper is no longer used to drive the main simulation pipeline, but is
# retained for potential offline analysis or future tooling.
_MAX_AUTO_Z_STACK_RANGE_NM = 200000.0


def _estimate_required_z_stack_full_range_nm(
    diffusion_coefficient_m2_s: float,
    fps: float,
    duration_seconds: float,
    coverage_probability: float,
) -> float:
    """
    Estimate the full iPSF z-stack range (in nm) required for a single
    Brownian particle so that its z-position stays within the stack for the
    entire video with approximately the requested coverage_probability.

    NOTE:
        This function is kept for potential offline analysis, but the runtime
        simulation no longer uses any global z-stack estimator to determine
        the actual PSF z-ranges. The production pipeline now derives z-ranges
        per particle type directly from the realized Brownian trajectories.

    Model:
        - 1D z-motion is Brownian with diffusion coefficient D:
              Z_T ~ N(0, sigma_T^2), sigma_T = sqrt(2 * D * T)
        - There are N = fps * duration_seconds frames.
        - We conservatively enforce:
              N * P(|Z_T| > L) <= 1 - coverage_probability
          which, using Gaussian tails, gives:
              L = sigma_T * z_score
          where:
              z_score = Phi^{-1}(1 - (1 - coverage_probability) / (2N))

    The returned value is the full range 2L expressed in nanometers. This
    helper does not clamp to _MAX_AUTO_Z_STACK_RANGE_NM; the caller is
    responsible for any global caps.

    Args:
        diffusion_coefficient_m2_s: Brownian diffusion coefficient D (m^2/s).
        fps: Frames per second of the simulation (float > 0).
        duration_seconds: Total simulation time in seconds (float > 0).
        coverage_probability: Target probability in (0, 1) that the particle
            stays within the z-stack for the entire video.

    Returns:
        float: Estimated full z-stack range in nanometers (>= 0). Returns 0.0
            if inputs are degenerate (non-positive D, fps, or duration).
    """
    D = float(diffusion_coefficient_m2_s)
    fps = float(fps)
    duration_seconds = float(duration_seconds)
    coverage_probability = float(coverage_probability)

    if D <= 0.0 or fps <= 0.0 or duration_seconds <= 0.0:
        return 0.0

    # Number of frames used in the union bound.
    num_frames = max(int(fps * duration_seconds), 1)

    # Clamp coverage_probability into a numerically safe open interval (0, 1).
    if coverage_probability <= 0.0:
        coverage_probability = 1e-6
    if coverage_probability >= 1.0:
        coverage_probability = 1.0 - 1e-9

    # Total probability mass we allow for "ever leaving" the z-stack.
    total_tail_prob = 1.0 - coverage_probability

    # Conservative per-frame bound: P(|Z_T| > L) <= total_tail_prob / num_frames.
    per_frame_tail_prob = total_tail_prob / float(num_frames)
    if per_frame_tail_prob <= 0.0:
        per_frame_tail_prob = 1e-12

    # For a symmetric Gaussian, P(|Z_T| > L) = 2 * (1 - Phi(L / sigma_T)).
    # So 1 - Phi(L / sigma_T) = per_frame_tail_prob / 2.
    single_side_tail_prob = per_frame_tail_prob / 2.0
    target_cdf = 1.0 - single_side_tail_prob

    # Guard against numeric edge cases in norm.ppf.
    target_cdf = min(max(target_cdf, 1e-12), 1.0 - 1e-12)
    z_score = norm.ppf(target_cdf)

    if not np.isfinite(z_score) or z_score <= 0.0:
        return 0.0

    # One-dimensional standard deviation along z after the full duration.
    sigma_z_m = np.sqrt(2.0 * D * duration_seconds)

    # Half-range in meters, then convert to nanometers and double for full range.
    half_range_nm = float(z_score * sigma_z_m * 1e9)
    full_range_nm = 2.0 * half_range_nm

    return max(full_range_nm, 0.0)


def _estimate_global_z_stack_range_nm(params: dict, diameters_nm) -> float:
    """
    Legacy helper for estimating a single global iPSF z-stack full range.

    IMPORTANT:
        The main simulation pipeline no longer uses a global z-stack range to
        drive PSF computation. Instead, per-particle-type z-ranges are derived
        directly from the realized Brownian trajectories (see run_simulation).

        This function is retained for potential offline analysis or tooling,
        but its output is not used anywhere in the production rendering
        pipeline.
    """
    num_particles = int(params["num_particles"])
    if len(diameters_nm) != num_particles:
        raise ValueError(
            "Length of params['particle_diameters_nm'] "
            f"({len(diameters_nm)}) must match params['num_particles'] ({num_particles})."
        )

    fps = float(params["fps"])
    duration_seconds = float(params["duration_seconds"])
    temperature_K = float(params["temperature_K"])
    viscosity_Pa_s = float(params["viscosity_Pa_s"])

    if fps <= 0.0 or duration_seconds <= 0.0:
        fallback = float(params.get("z_stack_range_nm", 30500.0))
        return fallback

    coverage_probability = float(params.get("z_stack_coverage_probability", 0.9999))
    if coverage_probability <= 0.0:
        coverage_probability = 1e-6
    if coverage_probability >= 1.0:
        coverage_probability = 1.0 - 1e-9

    max_D = 0.0
    max_D_diameter_nm = None

    for d_nm in diameters_nm:
        d_nm = float(d_nm)
        D_m2_s = stokes_einstein_diffusion_coefficient(
            d_nm,
            temperature_K,
            viscosity_Pa_s,
        )
        if D_m2_s > max_D:
            max_D = D_m2_s
            max_D_diameter_nm = d_nm

    if max_D <= 0.0 or max_D_diameter_nm is None:
        fallback = float(params.get("z_stack_range_nm", 30500.0))
        return fallback

    auto_range_nm = _estimate_required_z_stack_full_range_nm(
        diffusion_coefficient_m2_s=max_D,
        fps=fps,
        duration_seconds=duration_seconds,
        coverage_probability=coverage_probability,
    )

    if auto_range_nm <= 0.0:
        fallback = float(params.get("z_stack_range_nm", 30500.0))
        return fallback

    if auto_range_nm > _MAX_AUTO_Z_STACK_RANGE_NM:
        auto_range_nm = _MAX_AUTO_Z_STACK_RANGE_NM

    return float(auto_range_nm)


def run_simulation(params: dict) -> None:
    """
    Run the complete iSCAT simulation and video generation pipeline for a given
    parameter dictionary.

    Updated architecture (fundamental change):
        - The PSF iPSF Z-stacks are no longer based on a single global
          z_stack_range_nm.
        - Instead, for each unique particle type (diameter, complex index),
          we:
              1) Simulate trajectories for all particles.
              2) Collect the z-positions of all particles of that type.
              3) Compute type-specific z_min/z_max from those trajectories.
              4) Expand that range with a safety factor and a minimum span.
              5) Build a type-specific z grid and compute the iPSF stack only
                 over that grid.
        - Each particle’s interpolator therefore has its own z-range matching
          the realized Brownian motion of its type, plus a margin. This avoids
          both wasted computation and truncation-induced visibility loss.

    From the user's perspective, the simulation behavior (video outputs,
    particle visibility) remains the same or improves: particles do not
    disappear due to PSF z-range truncation, and there is no longer a hidden
    global z-range coupling between different particle types.
    """
    # --- Setup Output Directories ---
    if params["mask_generation_enabled"]:
        base_mask_dir = params["mask_output_directory"]
        print(f"Checking for mask output directories at {base_mask_dir}...")
        os.makedirs(base_mask_dir, exist_ok=True)
        for i in range(params["num_particles"]):
            particle_mask_dir = os.path.join(base_mask_dir, f"particle_{i+1}")
            os.makedirs(particle_mask_dir, exist_ok=True)

    output_dir = os.path.dirname(params["output_filename"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Resolve per-particle refractive indices from materials/overrides ---
    particle_refractive_indices = resolve_particle_refractive_indices(params)
    diameters_nm = params["particle_diameters_nm"]

    # NOTE: We intentionally no longer override params["z_stack_range_nm"]
    # here with any automatic global estimator. The only remaining role of
    # z_stack_range_nm is to define the initial Z-distribution in the Brownian
    # trajectory simulation (see trajectory.simulate_trajectories). The PSF
    # z-ranges are now derived from the realized trajectories per particle
    # type (see below).

    # --- Step 1: Simulate particle movement ---
    trajectories_nm = simulate_trajectories(params)

    # --- Step 2: Compute unique iPSF stacks with per-type trajectory-based Z-ranges ---
    #
    # Particle type key: (diameter_nm, n_real, n_imag), exactly as used
    # previously for deduplication. For each type:
    #   - Identify which particle indices share this type.
    #   - From trajectories_nm[type_indices, :, 2], compute z_min and z_max.
    #   - Expand this range with a safety factor and a minimum span.
    #   - Build a type-specific z grid and compute the iPSF stack only on
    #     that grid.
    #
    # This ensures that each type’s iPSF stack covers exactly the Z-range it
    # actually visits (plus margin), and different types are not forced to
    # share a single global axial extent.
    print("Pre-computing unique particle iPSF stacks with trajectory-based Z-ranges...")
    num_particles = params["num_particles"]

    # Map type key -> list of particle indices
    type_to_indices = {}
    for i in range(num_particles):
        n_complex = particle_refractive_indices[i]
        key = (
            diameters_nm[i],
            float(n_complex.real),
            float(n_complex.imag),
        )
        type_to_indices.setdefault(key, []).append(i)

    # Safety parameters for per-type z-range expansion.
    z_step_nm = float(params["z_stack_step_nm"])
    if z_step_nm <= 0.0:
        raise ValueError("PARAMS['z_stack_step_nm'] must be positive.")

    # Use the configured z_stack_range_nm as a *scale* to define a reasonable
    # minimum span for PSF stacks, but do not use it as a global range. This
    # avoids collapsing stacks for nearly constant trajectories while keeping
    # visibility robust.
    fallback_global_z_range_nm = float(params.get("z_stack_range_nm", 30500.0))
    SAFETY_FACTOR = 1.1
    MIN_HALF_SPAN = max(4.0 * z_step_nm, 0.1 * fallback_global_z_range_nm)

    unique_particles = {}

    for type_key, indices in type_to_indices.items():
        diam_nm_type, n_real, n_imag = type_key
        indices_array = np.asarray(indices, dtype=int)

        # Extract all z positions for this type: shape (num_type_particles, num_frames)
        z_positions_type = trajectories_nm[indices_array, :, 2]
        z_min_realized = float(np.min(z_positions_type))
        z_max_realized = float(np.max(z_positions_type))

        # Compute center and half-span of the realized range.
        z_center = 0.5 * (z_min_realized + z_max_realized)
        z_half_span = 0.5 * (z_max_realized - z_min_realized)

        # Expand with a safety factor and enforce a minimum half-span. This
        # ensures that even if a type barely moves in Z (or starts at a fixed
        # Z plane), the iPSF stack still spans a physically reasonable depth
        # around that plane, and that rare outlier steps just outside the
        # observed extrema are covered.
        z_half_span_safe = max(z_half_span * SAFETY_FACTOR, MIN_HALF_SPAN)
        z_min_safe = z_center - z_half_span_safe
        z_max_safe = z_center + z_half_span_safe

        # Construct the per-type Z grid.
        # We include the endpoint by adding +z_step_nm to the upper bound.
        z_values_type = np.arange(z_min_safe, z_max_safe + z_step_nm, z_step_nm)

        print(
            "  Particle type (diameter = %.1f nm, n = %.4f + %.4fi): "
            "z_min_realized = %.1f nm, z_max_realized = %.1f nm, "
            "expanded to [%.1f, %.1f] nm with %d slices."
            % (
                float(diam_nm_type),
                float(n_real),
                float(n_imag),
                z_min_realized,
                z_max_realized,
                z_min_safe,
                z_max_safe,
                int(z_values_type.size),
            )
        )

        n_complex_type = complex(n_real, n_imag)
        unique_particles[type_key] = compute_ipsf_stack(
            params,
            diam_nm_type,
            n_complex_type,
            z_values_type,
        )

    # Assign the correct pre-computed iPSF interpolator to each particle.
    ipsf_interpolators = [
        unique_particles[
            (
                diameters_nm[i],
                float(particle_refractive_indices[i].real),
                float(particle_refractive_indices[i].imag),
            )
        ]
        for i in range(num_particles)
    ]

    # --- Step 3: Generate raw video frames and masks ---
    raw_signal_frames, raw_reference_frames = generate_video_and_masks(
        params,
        trajectories_nm,
        ipsf_interpolators,
    )

    # --- Step 4: Process frames for final video ---
    final_frames = apply_background_subtraction(
        raw_signal_frames,
        raw_reference_frames,
        params,
    )

    if not final_frames:
        print("Video generation failed or produced no frames. Exiting.")
        return

    # --- Step 5: Save the final video ---
    img_size = (params["image_size_pixels"], params["image_size_pixels"])
    save_video(params["output_filename"], final_frames, params["fps"], img_size)


def main():
    """
    Script entry point: run the simulation using the global config.PARAMS.

    This preserves the behavior of the original implementation so that running
    this file as a script still performs a single simulation configured by
    config.PARAMS, with the enhancement that iPSF z-stacks are now sized per
    particle type based on the realized trajectories rather than a single
    global z-range.
    """
    run_simulation(PARAMS)


if __name__ == '__main__':
    main()