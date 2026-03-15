# inference.py
# Two methods for recovering amplitude A from a field realisation.
#
# 1. Whittle likelihood evaluated on a grid over A
# 2. SNPE-C via the sbi package (normalising flow)
#
# The Whittle result serves as an analytical benchmark.
# Agreement between the two methods validates the neural approach.
#
# References:
#   Cranmer, Brehmer & Louppe (2020) - The frontier of simulation-based inference
#   Tejero-Cantero et al. (2020) - sbi: A toolkit for simulation-based inference
#   Whittle (1953) - Estimation and information in stationary time series

import numpy as np
import torch
from sbi import utils as utils
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.user_input_checks import check_sbi_inputs, process_prior, process_simulator

from simulator import lorentzian_power_spectrum, generate_field, summary_statistic


# ------------------------------------------------------------------
# Whittle likelihood
# ------------------------------------------------------------------

def whittle_log_likelihood(Pk_data, Nmodes, k_centers, A_test, k0):
    # Whittle approximation: for a Gaussian field the binned power spectrum
    # is approximately chi-squared with 2*Nmodes degrees of freedom.
    # In the many-modes limit this gives a Gaussian with mean P_model
    # and variance 2*P_model^2 / Nmodes.
    # The log(var) term accounts for the fact that variance itself depends on A.

    P_model = lorentzian_power_spectrum(k_centers, A_test, k0)
    var = 2.0 * P_model**2 / Nmodes
    return -0.5 * np.sum((Pk_data - P_model)**2 / var + np.log(var))


def grid_posterior(Pk_data, Nmodes, k_centers, k0, A_min=0.1, A_max=3.0, n_grid=300):
    # Evaluates the likelihood at n_grid values of A and converts to a posterior.
    # Flat prior assumed, so posterior is proportional to likelihood.
    # Subtracting the max before exponentiating avoids numerical overflow.

    A_grid = np.linspace(A_min, A_max, n_grid)

    loglike = np.zeros(n_grid)
    for idx, A_test in enumerate(A_grid):
        loglike[idx] = whittle_log_likelihood(Pk_data, Nmodes, k_centers, A_test, k0)

    posterior = np.exp(loglike - loglike.max())
    return A_grid, posterior, loglike


def map_estimate(A_grid, posterior):
    return A_grid[np.argmax(posterior)]


def credible_interval(A_grid, posterior, level=0.68):
    # Integrates the posterior numerically to find the credible interval bounds.
    norm = np.trapezoid(posterior, A_grid)
    cdf = np.cumsum(posterior) * (A_grid[1] - A_grid[0]) / norm
    lo_idx = np.searchsorted(cdf, (1 - level) / 2)
    hi_idx = np.searchsorted(cdf, 1 - (1 - level) / 2)
    return A_grid[lo_idx], A_grid[hi_idx]


# ------------------------------------------------------------------
# SNPE-C via sbi
# ------------------------------------------------------------------

def build_prior(A_min=0.1, A_max=3.0):
    # Uniform prior over A as a torch distribution - required format for sbi
    return utils.BoxUniform(
        low=torch.tensor([A_min], dtype=torch.float32),
        high=torch.tensor([A_max], dtype=torch.float32),
    )


def make_simulator(N, k0, n_bins):
    # Wraps the forward model for sbi.
    # sbi expects: theta (torch tensor) -> x (torch tensor)

    rng = np.random.default_rng()

    def simulator(theta):
        A = float(theta[0])
        field = generate_field(N, max(A, 1e-3), k0, seed=int(rng.integers(1_000_000)))
        x = summary_statistic(field, N, n_bins=n_bins)
        return torch.tensor(x, dtype=torch.float32)

    return simulator


class SNPEInference:
    # Wraps the SNPE-C workflow: simulate -> train -> infer.
    #
    # SNPE-C trains a normalising flow to directly learn p(A|x) from simulated
    # (parameter, summary) pairs. Unlike the Whittle approach it makes no
    # assumptions about the likelihood form, so it would generalise to problems
    # where no analytical likelihood exists (e.g. field-level inference,
    # non-Gaussian statistics).
    #
    # Amortised: training cost is paid once, inference on any new observation
    # is then a single forward pass through the network.

    def __init__(self, N, k0, n_bins=30, A_min=0.1, A_max=3.0):
        self.N = N
        self.k0 = k0
        self.n_bins = n_bins
        self.prior = build_prior(A_min, A_max)
        self.simulator_fn = make_simulator(N, k0, n_bins)
        self.posterior = None

    def run(self, n_sims=2000, verbose=True):
        # Validate inputs, draw training simulations, train the flow.

        prior, _, prior_returns_numpy = process_prior(self.prior)
        simulator = process_simulator(self.simulator_fn, prior, prior_returns_numpy)
        check_sbi_inputs(simulator, prior)

        inference = SNPE(prior=prior)

        if verbose:
            print(f"  Running {n_sims} simulations...")

        theta, x = simulate_for_sbi(
            simulator, prior,
            num_simulations=n_sims,
            simulation_batch_size=1,
        )

        # print(f"  theta shape: {theta.shape}, x shape: {x.shape}")

        if verbose:
            print("  Training normalising flow...")

        inference.append_simulations(theta, x)
        density_estimator = inference.train(show_train_summary=verbose)
        self.posterior = inference.build_posterior(density_estimator)

    def sample(self, x_obs, n_samples=5000):
        if isinstance(x_obs, np.ndarray):
            x_obs = torch.tensor(x_obs, dtype=torch.float32)
        return self.posterior.sample((n_samples,), x=x_obs, show_progress_bars=False)

    def log_prob(self, A_values, x_obs):
        # Evaluates posterior density on a grid - used for plotting
        if isinstance(x_obs, np.ndarray):
            x_obs = torch.tensor(x_obs, dtype=torch.float32)
        theta_grid = torch.tensor(A_values[:, None], dtype=torch.float32)
        return self.posterior.log_prob(theta_grid, x=x_obs).detach().numpy()
