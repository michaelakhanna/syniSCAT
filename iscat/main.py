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
# This is chosen to stay at or below the "like 200000 nm" scale you explicitly
# wanted to avoid.
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
    Estimate a single global iPSF z-stack full range (in nm) for the current
    simulation, based on Brownian motion statistics and the desired coverage
    probability.

    Strategy:
        - For each particle diameter, compute its diffusion coefficient D via
          the Stokes–Einstein equation.
        - Identify the most mobile particle (largest D).
        - Use _estimate_required_z_stack_full_range_nm for that particle,
          with the configured fps, duration_seconds, and
          z_stack_coverage_probability, to obtain a required full z-stack
          range.
        - Clamp the result to _MAX_AUTO_Z_STACK_RANGE_NM to avoid extreme
          computational cost.
        - If anything goes wrong (degenerate parameters), fall back to the
          existing PARAMS["z_stack_range_nm"].

    This function also logs the final chosen z-stack range and the particle
    that determined it.

    Args:
        params: Simulation parameter dictionary.
        diameters_nm: Sequence of per-particle diameters in nanometers.

    Returns:
        float: Global z-stack full range in nanometers to be written into
            params["z_stack_range_nm"].
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
        # Degenerate timing: fall back immediately.
        fallback = float(params.get("z_stack_range_nm", 30500.0))
        print(
            "Automatic z-stack range estimation skipped due to non-positive fps "
            "or duration; using existing PARAMS['z_stack_range_nm'] "
            f"={fallback:.1f} nm."
        )
        return fallback

    coverage_probability = float(params.get("z_stack_coverage_probability", 0.9999))
    # Clamp into a reasonable open interval for numerical stability.
    if coverage_probability <= 0.0:
        coverage_probability = 1e-6
    if coverage_probability >= 1.0:
        coverage_probability = 1.0 - 1e-9

    # Find the most mobile particle (largest D).
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
        # Degenerate diffusion: fall back to existing configuration.
        fallback = float(params.get("z_stack_range_nm", 30500.0))
        print(
            "Automatic z-stack range estimation encountered non-positive diffusion "
            "coefficients; using existing PARAMS['z_stack_range_nm'] "
            f"={fallback:.1f} nm."
        )
        return fallback

    # Compute required full range for the most mobile particle.
    auto_range_nm = _estimate_required_z_stack_full_range_nm(
        diffusion_coefficient_m2_s=max_D,
        fps=fps,
        duration_seconds=duration_seconds,
        coverage_probability=coverage_probability,
    )

    if auto_range_nm <= 0.0:
        fallback = float(params.get("z_stack_range_nm", 30500.0))
        print(
            "Automatic z-stack range estimation returned a non-positive value; "
            "using existing PARAMS['z_stack_range_nm'] "
            f"={fallback:.1f} nm instead."
        )
        return fallback

    # Clamp to the global maximum allowed range to avoid extreme compute.
    if auto_range_nm > _MAX_AUTO_Z_STACK_RANGE_NM:
        print(
            f"Automatic z-stack range {auto_range_nm:.1f} nm exceeds the hard cap "
            f"{_MAX_AUTO_Z_STACK_RANGE_NM:.1f} nm; clamping to the cap. "
            "Consider lowering 'z_stack_coverage_probability' or shortening the "
            "video if this occurs frequently."
        )
        auto_range_nm = _MAX_AUTO_Z_STACK_RANGE_NM

    print("Estimated global iPSF z-stack range based on Brownian motion:")
    print(f"  coverage probability target : {coverage_probability:.6f}")
    print(f"  most mobile particle diameter: {max_D_diameter_nm:.1f} nm")
    print(f"  resulting z_stack_range_nm   : {auto_range_nm:.1f} nm")

    return float(auto_range_nm)


def run_simulation(params: dict) -> None:
    """
    Run the complete iSCAT simulation and video generation pipeline for a given
    parameter dictionary.

    This function is the core programmatic entry point for the simulation. It
    performs:

        1. Output directory preparation (video and masks).
        2. Per-particle refractive index resolution from materials/overrides.
        3. Automatic estimation of a global iPSF z-stack range based on
           Brownian diffusion statistics and z_stack_coverage_probability.
        4. 3D Brownian motion trajectory simulation.
        5. Pre-computation of unique iPSF Z-stacks for each particle type.
        6. Frame-by-frame rendering of signal/reference frames and masks.
        7. Background subtraction, normalization, and final .mp4 encoding.

    When called as:

        run_simulation(PARAMS)

    it reproduces the behavior of the original `main()` function, with the
    enhancement that the z-stack range is no longer manually tuned but is
    derived from a physically motivated probability model.

    Args:
        params (dict): Simulation parameter dictionary. Typically a dictionary
            following the structure of config.PARAMS, possibly with overrides
            applied (e.g., for presets or randomized generation).
    """
    # --- Setup Output Directories ---
    # Ensure the output directories for the video and masks exist.
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
    # This combines:
    #   - params["particle_materials"] (if provided) -> material-based lookup, and
    #   - params["particle_refractive_indices"] (if provided) -> explicit overrides.
    #
    # The result is a single complex refractive index per particle, stored back
    # into params["particle_refractive_indices"] as a numpy array. This ensures
    # subsequent optics code sees a consistent, resolved value regardless of how
    # the user specified the particle properties.
    particle_refractive_indices = resolve_particle_refractive_indices(params)
    diameters_nm = params["particle_diameters_nm"]

    # --- Automatic estimation of global z-stack range ---
    # Instead of manually specifying z_stack_range_nm, estimate a single global
    # iPSF z-stack range from Brownian motion statistics and the desired
    # coverage probability. This ensures that smaller (more mobile) particles
    # get a sufficiently wide z-stack while avoiding excessively large stacks.
    try:
        auto_z_stack_range_nm = _estimate_global_z_stack_range_nm(params, diameters_nm)
        params["z_stack_range_nm"] = auto_z_stack_range_nm
    except Exception as exc:
        # Fall back gracefully to the existing configuration if anything goes wrong.
        warnings.warn(
            f"Automatic z-stack range estimation failed ({exc!r}); "
            "falling back to existing PARAMS['z_stack_range_nm'] value.",
            RuntimeWarning,
        )
        auto_z_stack_range_nm = float(params.get("z_stack_range_nm", 30500.0))
        params["z_stack_range_nm"] = auto_z_stack_range_nm
        print(
            f"Using fallback z_stack_range_nm={auto_z_stack_range_nm:.1f} nm "
            "for both trajectories and iPSF computation."
        )

    # --- Step 1: Simulate particle movement ---
    # This generates the 3D coordinates for each particle over time.
    trajectories_nm = simulate_trajectories(params)

    # --- Step 2: Compute unique iPSF stacks ---
    # To save computation time, only compute the iPSF once for each unique type
    # of particle. A unique particle type is defined by its diameter and complex
    # refractive index (n + i k) within the medium.
    unique_particles = {}
    print("Pre-computing unique particle iPSF stacks...")
    num_particles = params["num_particles"]

    for i in range(num_particles):
        n_complex = particle_refractive_indices[i]
        key = (
            diameters_nm[i],
            float(n_complex.real),
            float(n_complex.imag),
        )
        if key not in unique_particles:
            unique_particles[key] = compute_ipsf_stack(
                params,
                diameters_nm[i],
                n_complex,
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
    # This is the main rendering loop that generates the raw 16-bit data.
    raw_signal_frames, raw_reference_frames = generate_video_and_masks(
        params,
        trajectories_nm,
        ipsf_interpolators,
    )

    # --- Step 4: Process frames for final video ---
    # This performs background subtraction and normalization to 8-bit.
    final_frames = apply_background_subtraction(
        raw_signal_frames,
        raw_reference_frames,
        params,
    )

    if not final_frames:
        print("Video generation failed or produced no frames. Exiting.")
        return

    # --- Step 5: Save the final video ---
    # Encodes the processed frames into an .mp4 file.
    img_size = (params["image_size_pixels"], params["image_size_pixels"])
    save_video(params["output_filename"], final_frames, params["fps"], img_size)


def main():
    """
    Script entry point: run the simulation using the global config.PARAMS.

    This preserves the behavior of the original implementation so that running
    this file as a script still performs a single simulation configured by
    config.PARAMS, with the enhancement of automatic z-stack range estimation.
    """
    run_simulation(PARAMS)


if __name__ == '__main__':
    main()