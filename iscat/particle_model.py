from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from optics import IPSFZInterpolator


@dataclass(frozen=True)
class ParticleType:
    """
    Describes an optical "particle type" in the simulation.

    In the current implementation, a particle type corresponds to a single
    spherical particle characterized by:
        - Its physical diameter in nanometers.
        - Its complex refractive index (n + i k).
        - A precomputed iPSF Z-interpolator defined on a type-specific z-grid.

    Multiple particle instances that share the same (diameter, refractive index)
    pair reference the same ParticleType object and therefore share the same
    iPSF Z-stack and interpolator. This deduplication logic already existed in
    main.run_simulation; this class makes it explicit and extensible.

    Future extensions (non-spherical / composite particles) can add additional
    fields here (e.g., rigid sub-particle geometry, shape kind), but from the
    perspective of the current spherical pipeline, those extensions will be
    additive and will not change behavior.
    """
    diameter_nm: float
    refractive_index: complex
    ipsf_interpolator: IPSFZInterpolator

    @property
    def type_key(self) -> Tuple[float, float, float]:
        """
        Return a tuple that uniquely identifies this particle type within the
        current simulation:

            (diameter_nm, n.real, n.imag)

        This matches the key used in main.run_simulation when grouping
        particles by type.
        """
        n = self.refractive_index
        return (self.diameter_nm, float(n.real), float(n.imag))


@dataclass
class ParticleInstance:
    """
    Represents a single particle instance in the simulation.

    Each instance:
        - References exactly one ParticleType (which defines its optical
          behavior and iPSF interpolator).
        - Stores its full 3D trajectory in nanometers over all frames.
        - Stores its per-particle signal multiplier (scalar amplitude factor).

    This is a purely structural abstraction for now: the rest of the code
    still uses the legacy arrays (trajectories_nm and ipsf_interpolators)
    to render the video. Subsequent refactors can switch rendering and
    trackability logic over to using ParticleInstance directly without
    changing how trajectories or iPSF stacks are produced.
    """
    index: int
    particle_type: ParticleType
    trajectory_nm: np.ndarray
    signal_multiplier: float


def build_particle_types_and_instances(
    params: dict,
    trajectories_nm: np.ndarray,
    particle_refractive_indices: np.ndarray,
    ipsf_interpolators_by_type: Dict[Tuple[float, float, float], IPSFZInterpolator],
) -> Tuple[Dict[Tuple[float, float, float], ParticleType], List[ParticleInstance]]:
    """
    Construct ParticleType and ParticleInstance objects for the current
    simulation run.

    This helper centralizes the mapping from per-particle scalar parameters
    (diameter, refractive index, signal multiplier) and trajectories into
    structured objects, using the existing per-type iPSF interpolators that
    were computed in main.run_simulation.

    It does not change any behavior of the simulation; the returned objects
    are an additional representation that can be adopted by downstream
    components (e.g., rendering) in future refactors.

    Args:
        params (dict):
            Global parameter dictionary (PARAMS) for this simulation.
            Must contain:
                - "num_particles"
                - "particle_diameters_nm"
                - "particle_signal_multipliers"
        trajectories_nm (np.ndarray):
            Particle trajectories with shape (num_particles, num_frames, 3),
            as returned by trajectory.simulate_trajectories.
        particle_refractive_indices (np.ndarray):
            Complex refractive indices for each particle, shape (num_particles,).
        ipsf_interpolators_by_type (dict):
            Mapping from type_key = (diameter_nm, n_real, n_imag) to the
            IPSFZInterpolator computed for that type in main.run_simulation.

    Returns:
        tuple:
            - A dictionary mapping type_key -> ParticleType.
            - A list of ParticleInstance objects of length num_particles.

    Raises:
        ValueError: If the lengths or shapes of the inputs are inconsistent
            with PARAMS["num_particles"].
    """
    num_particles = int(params["num_particles"])

    diameters_nm = params["particle_diameters_nm"]
    if len(diameters_nm) != num_particles:
        raise ValueError(
            "Length of PARAMS['particle_diameters_nm'] "
            f"({len(diameters_nm)}) must match PARAMS['num_particles'] ({num_particles})."
        )

    signal_multipliers = params.get("particle_signal_multipliers", None)
    if signal_multipliers is None or len(signal_multipliers) != num_particles:
        raise ValueError(
            "PARAMS['particle_signal_multipliers'] must be provided and have "
            f"length equal to PARAMS['num_particles'] ({num_particles})."
        )

    if trajectories_nm.shape[0] != num_particles or trajectories_nm.shape[2] != 3:
        raise ValueError(
            "trajectories_nm must have shape (num_particles, num_frames, 3). "
            f"Got {trajectories_nm.shape} for num_particles={num_particles}."
        )

    if particle_refractive_indices.shape[0] != num_particles:
        raise ValueError(
            "particle_refractive_indices must have length num_particles. "
            f"Got {particle_refractive_indices.shape[0]} for num_particles={num_particles}."
        )

    # Build ParticleType objects from the provided per-type interpolators.
    particle_types: Dict[Tuple[float, float, float], ParticleType] = {}
    for type_key, interpolator in ipsf_interpolators_by_type.items():
        diam_nm, n_real, n_imag = type_key
        n_complex = complex(n_real, n_imag)
        particle_types[type_key] = ParticleType(
            diameter_nm=float(diam_nm),
            refractive_index=n_complex,
            ipsf_interpolator=interpolator,
        )

    # Build ParticleInstance objects, one per particle, referencing the
    # appropriate ParticleType and its trajectory.
    instances: List[ParticleInstance] = []
    for i in range(num_particles):
        n_complex = particle_refractive_indices[i]
        type_key = (
            float(diameters_nm[i]),
            float(n_complex.real),
            float(n_complex.imag),
        )

        try:
            ptype = particle_types[type_key]
        except KeyError as exc:
            raise KeyError(
                "No ParticleType found for particle index {i} with type_key "
                f"{type_key}. This indicates a mismatch between the keys used "
                "to build ipsf_interpolators_by_type and the per-particle "
                "diameter/refractive_index arrays."
            ) from exc

        instance = ParticleInstance(
            index=i,
            particle_type=ptype,
            trajectory_nm=trajectories_nm[i],
            signal_multiplier=float(signal_multipliers[i]),
        )
        instances.append(instance)

    return particle_types, instances