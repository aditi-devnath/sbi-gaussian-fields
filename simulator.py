# simulator.py
# Generates 2D isotropic Gaussian random fields with a Lorentzian power spectrum
# P(k) = A / (1 + (k/k0)^2)
# A is the amplitude parameter recovered throughout this project.

import numpy as np


def lorentzian_power_spectrum(k, A, k0):
    return A / (1.0 + (k / k0) ** 2)


def make_k_grid(N):
    # fftfreq returns frequencies in [-0.5, 0.5], multiply by N to get wavenumbers
    freqs = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(freqs, freqs)
    k = np.sqrt(kx**2 + ky**2)
    return kx, ky, k


def generate_field(N, A, k0, seed=None):
    # Generates one realisation of the field in Fourier space.
    # Each mode is drawn as a complex Gaussian with variance P(k).
    # Hermitian symmetry (delta_k[-i,-j] = conj(delta_k[i,j])) is enforced
    # so that the inverse FFT returns a real-valued field.

    rng = np.random.default_rng(seed)
    _, _, k_grid = make_k_grid(N)
    delta_k = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            k_val = k_grid[i, j]
            Pk = lorentzian_power_spectrum(k_val, A, k0)

            if (i < N // 2) or (i == N // 2 and j <= N // 2):
                re = rng.normal(0.0, np.sqrt(Pk / 2.0))
                im = rng.normal(0.0, np.sqrt(Pk / 2.0))
                delta_k[i, j] = re + 1j * im
                i_c, j_c = (-i) % N, (-j) % N
                delta_k[i_c, j_c] = np.conj(delta_k[i, j])

    delta_k[0, 0] = 0.0  # zero DC mode -> zero-mean field
    return np.fft.ifft2(delta_k).real


def estimate_power_spectrum(delta, N, n_bins=None):
    # Estimates P(k) by azimuthal averaging of |delta_k|^2 in radial bins.
    # Averaging in annuli is valid because the field is isotropic.
    # Returns bin centres, mean power per bin, and mode counts.
    # Mode counts are needed for the Whittle likelihood variance.

    if n_bins is None:
        n_bins = N // 2

    _, _, k_grid = make_k_grid(N)
    k_flat = k_grid.flatten()

    Pk_2D = np.abs(np.fft.fft2(delta)) ** 2
    Pk_flat = Pk_2D.flatten()

    k_bins = np.linspace(0, k_flat.max(), n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    Pk_est = np.zeros(n_bins)
    Nmodes = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i + 1])
        if mask.any():
            Pk_est[i] = Pk_flat[mask].mean()
            Nmodes[i] = mask.sum()

    # Drop DC bin - zeroed during field generation
    return k_centers[1:], Pk_est[1:], Nmodes[1:]


def summary_statistic(delta, N, n_bins=30):
    # Compresses the NxN field to a 1D vector of binned power spectrum values.
    # This is the data compression step for inference - raw pixels are not used.
    # For a Gaussian field, P(k) is a sufficient statistic for A.
    # n_bins+1 passed to estimate_power_spectrum because the DC bin gets dropped
    _, Pk_est, _ = estimate_power_spectrum(delta, N, n_bins=n_bins + 1)
    return Pk_est
