import numpy as np
from scipy.special import jn, yn
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from tqdm import tqdm


class IPSFZInterpolator:
    """
    Lightweight 1D interpolator over a precomputed iPSF Z-stack.

    The underlying data is a 3D array of shape (num_z, height, width) containing
    the complex-valued interferometric PSF for each discrete z position in
    z_values_nm. This class performs linear interpolation along z and returns
    the corresponding 2D complex field.

    This replaces the use of scipy.interpolate.RegularGridInterpolator for the
    specific case of a 1D grid (z) with a full 2D field stored at each grid
    point. It keeps the intended behavior: fast lookup of a 2D iPSF slice at an
    arbitrary z, and zero field outside the precomputed z-range.
    """

    def __init__(self, z_values_nm, ipsf_stack_complex):
        """
        Args:
            z_values_nm (array-like): 1D array of z positions (in nm) at which
                the iPSF has been precomputed.
            ipsf_stack_complex (np.ndarray): 3D complex array with shape
                (len(z_values_nm), height, width). The first axis corresponds
                to the z positions.
        """
        self.z_values = np.asarray(z_values_nm, dtype=float)
        if self.z_values.ndim != 1 or self.z_values.size == 0:
            raise ValueError("z_values_nm must be a non-empty 1D array.")

        self.ipsf_stack = np.asarray(ipsf_stack_complex, dtype=np.complex128)
        if self.ipsf_stack.shape[0] != self.z_values.size:
            raise ValueError(
                "First dimension of ipsf_stack_complex must match the length "
                "of z_values_nm."
            )

        self.z_min = float(self.z_values[0])
        self.z_max = float(self.z_values[-1])

        if self.z_values.size > 1:
            # The z grid is constructed with np.arange, so we assume uniform spacing.
            self.dz = float(self.z_values[1] - self.z_values[0])
        else:
            # Degenerate case: only a single z-slice; no interpolation possible.
            self.dz = 1.0

    def __call__(self, z_nm):
        """
        Linearly interpolate the iPSF stack along z.

        Args:
            z_nm (float or array-like): Axial position(s) in nanometers.

        Returns:
            np.ndarray:
                - If z_nm is a scalar, returns a 2D complex array of shape
                  (height, width) for that z position.
                - If z_nm is array-like with shape (N,), returns a 3D complex
                  array of shape (N, height, width), where each slice along the
                  first axis corresponds to one input z.
        """
        z = np.asarray(z_nm, dtype=float)

        # Scalar input: return a single 2D iPSF slice.
        if z.ndim == 0:
            return self._interp_single(float(z))

        # Vector input: interpolate each z independently.
        z_flat = z.ravel()
        out = np.empty((z_flat.size,) + self.ipsf_stack.shape[1:], dtype=np.complex128)
        for idx, z_val in enumerate(z_flat):
            out[idx] = self._interp_single(float(z_val))

        # Reshape back to match the input z shape, with PSF dimensions appended.
        new_shape = z.shape + self.ipsf_stack.shape[1:]
        return out.reshape(new_shape)

    def _interp_single(self, z_val):
        """
        Interpolate for a single scalar z position.

        For z values outside the precomputed range, this mimics the behavior of
        a RegularGridInterpolator with bounds_error=False and fill_value=0 by
        returning an all-zero field.
        """
        # Outside the precomputed z-range: no signal from the particle.
        if z_val < self.z_min or z_val > self.z_max:
            return np.zeros_like(self.ipsf_stack[0])

        # If only a single z-slice exists, always return that slice.
        if self.z_values.size == 1:
            return self.ipsf_stack[0]

        # Compute fractional index position within the z grid.
        rel_pos = (z_val - self.z_min) / self.dz
        lower_index = int(np.floor(rel_pos))

        # Clamp to valid indices.
        if lower_index >= self.z_values.size - 1:
            return self.ipsf_stack[-1]
        if lower_index < 0:
            return self.ipsf_stack[0]

        alpha = rel_pos - lower_index  # fractional distance between slices
        lower_slice = self.ipsf_stack[lower_index]
        upper_slice = self.ipsf_stack[lower_index + 1]
        return (1.0 - alpha) * lower_slice + alpha * upper_slice


def mie_an_bn(m, x):
    """
    Calculates Mie scattering coefficients a_n and b_n.

    Args:
        m (complex): The complex refractive index ratio (particle/medium).
        x (float): The size parameter (2*pi*r/lambda).
    """
    nmax = int(np.ceil(x + 4 * x**(1/3) + 2))
    n = np.arange(1, nmax + 1)

    # Riccati-Bessel functions
    psi_n_x = np.sqrt(0.5 * np.pi * x) * jn(n + 0.5, x)
    psi_n_mx = np.sqrt(0.5 * np.pi * m * x) * jn(n + 0.5, m * x)
    chi_n_x = -np.sqrt(0.5 * np.pi * x) * yn(n + 0.5, x)

    # Derivatives of Riccati-Bessel functions
    psi_prime_n_x = (
        np.sqrt(0.5 * np.pi / x) * jn(n + 0.5, x) / 2
        + np.sqrt(0.5 * np.pi * x)
        * (jn(n - 1 + 0.5, x) - (n + 1) / x * jn(n + 0.5, x))
    )
    psi_prime_n_mx = (
        np.sqrt(0.5 * np.pi / (m * x)) * jn(n + 0.5, m * x) / 2
        + np.sqrt(0.5 * np.pi * m * x)
        * (jn(n - 1 + 0.5, m * x) - (n + 1) / (m * x) * jn(n + 0.5, m * x))
    )

    xi_n_x = psi_n_x + 1j * chi_n_x
    xi_prime_n_x = (
        psi_prime_n_x
        - 1j * np.sqrt(0.5 * np.pi / x) * yn(n + 0.5, x) / 2
        - 1j
        * np.sqrt(0.5 * np.pi * x)
        * (yn(n - 1 + 0.5, x) - (n + 1) / x * yn(n + 0.5, x))
    )

    # Mie coefficients calculation
    a_n = (
        (m**2 * psi_n_mx * psi_prime_n_x - psi_n_x * psi_prime_n_mx)
        / (m**2 * psi_n_mx * xi_prime_n_x - xi_n_x * psi_prime_n_mx)
    )
    b_n = (
        (psi_n_mx * psi_prime_n_x - psi_n_x * psi_prime_n_mx)
        / (psi_n_mx * xi_prime_n_x - xi_n_x * psi_prime_n_mx)
    )

    return a_n, b_n


def mie_S1_S2(m, x, mu):
    """
    Calculates Mie scattering amplitude functions S1 and S2.

    Args:
        m (complex): complex refractive index ratio
        x (float): size parameter
        mu (float): cos(theta) where theta is the scattering angle
    """
    nmax = int(np.ceil(x + 4 * x**(1 / 3) + 2))
    a_n, b_n = mie_an_bn(m, x)

    S1 = 0j
    S2 = 0j
    pi_n = np.zeros(nmax + 2)
    tau_n = np.zeros(nmax + 2)
    pi_n[1] = 1.0

    # Summation over n for S1 and S2
    for n in range(1, nmax + 1):
        if n > 1:
            pi_n[n] = ((2 * n - 1) / (n - 1)) * mu * pi_n[n - 1] - (n / (n - 1)) * pi_n[n - 2]

        tau_n[n] = n * mu * pi_n[n] - (n + 1) * pi_n[n - 1]

        factor = (2 * n + 1) / (n * (n + 1))
        S1 += factor * (a_n[n - 1] * pi_n[n] + b_n[n - 1] * tau_n[n])
        S2 += factor * (a_n[n - 1] * tau_n[n] + b_n[n - 1] * pi_n[n])

    return S1, S2


def compute_ipsf_stack(params, particle_diameter_nm, particle_refractive_index):
    """
    Computes a complex 3D vectorial interferometric Point Spread Function (iPSF)
    stack using the Debye-Born integral, calculated via FFT for efficiency.

    For each discrete z position in the stack, a full 2D complex field is
    computed. The function returns an interpolator that provides the iPSF at
    arbitrary z positions within (or outside) the precomputed range.

    Args:
        params (dict): The main simulation parameter dictionary.
        particle_diameter_nm (float): The diameter of the particle for this iPSF.
        particle_refractive_index (complex): The complex refractive index of
            the particle.

    Returns:
        IPSFZInterpolator: An interpolator object that can return the complex
            2D iPSF for any given z-position within the stack's range. Outside
            the range, it returns a zero field.
    """
    # --- Setup k-space coordinates and optical parameters ---
    os_factor = params["psf_oversampling_factor"]
    pupil_samples = params["pupil_samples"]
    psf_size_nm = params["image_size_pixels"] * params["pixel_size_nm"]
    n_medium = params["refractive_index_medium"]
    wavelength_medium_nm = params["wavelength_nm"] / n_medium
    k_medium = 2 * np.pi / wavelength_medium_nm

    dk = (2 * np.pi / psf_size_nm) * os_factor
    kx = np.arange(-pupil_samples // 2, pupil_samples // 2) * dk
    ky = np.arange(-pupil_samples // 2, pupil_samples // 2) * dk
    Kx, Ky = np.meshgrid(kx, ky)
    K_sq = Kx**2 + Ky**2

    # --- Define the pupil aperture and coordinates ---
    sin_theta = np.sqrt(K_sq) / k_medium
    max_sin_theta = params["numerical_aperture"] / n_medium
    aperture_mask = (sin_theta <= max_sin_theta).astype(float)

    cos_theta = np.zeros_like(sin_theta)
    valid_mask = sin_theta <= 1
    cos_theta[valid_mask] = np.sqrt(1 - sin_theta[valid_mask] ** 2)

    # --- Calculate Mie scattering amplitudes across the pupil ---
    m = particle_refractive_index / n_medium
    radius_nm = particle_diameter_nm / 2
    x = 2 * np.pi * radius_nm / wavelength_medium_nm
    mu = np.zeros_like(cos_theta)
    mu[valid_mask] = cos_theta[valid_mask]
    S1_vec, S2_vec = np.vectorize(mie_S1_S2)(m, x, mu)

    # --- Define aberration and apodization functions ---
    z_values = np.arange(
        -params["z_stack_range_nm"] / 2,
        params["z_stack_range_nm"] / 2 + 1,
        params["z_stack_step_nm"],
    )
    rho = sin_theta / max_sin_theta
    zernike_spherical = np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    spherical_phase = params["spherical_aberration_strength"] * zernike_spherical * 2 * np.pi
    apodization = np.exp(-params["apodization_factor"] * (rho**2))

    print(f"Computing iPSF stack for {particle_diameter_nm} nm particle...")
    ipsf_stack_complex = np.zeros((len(z_values), pupil_samples, pupil_samples), dtype=np.complex128)

    # --- Compute the iPSF for each Z-slice ---
    for i, z in enumerate(tqdm(z_values)):
        defocus_phase = k_medium * z * cos_theta
        aberration_phase = defocus_phase + spherical_phase

        # This simulates the complex, random aberrations of a real lens system.
        aberration_phase += (
            np.random.rand(pupil_samples, pupil_samples) - 0.5
        ) * params["random_aberration_strength"] * 2 * np.pi

        pupil_function = (
            -1j * wavelength_medium_nm
        ) * aperture_mask * apodization * S2_vec * np.exp(1j * aberration_phase)

        # The Amplitude Spread Function (ASF) is the Fourier transform of the pupil function.
        asf = fftshift(ifft2(ifftshift(pupil_function)))
        ipsf_stack_complex[i, :, :] = asf

    # Create a custom interpolator for fast lookups later.
    interpolator = IPSFZInterpolator(z_values, ipsf_stack_complex)

    print("iPSF stack computation complete.")
    return interpolator