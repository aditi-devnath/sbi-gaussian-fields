# run_demo.py
# Runs all three inference methods end to end and saves comparison plots.
#
# Steps:
#   1. Generate a field realisation and check the power spectrum
#   2. Whittle likelihood posterior over a grid of A values
#   3. Monte Carlo coverage test (200 realisations)
#   4. SNPE-C via sbi
#   5. Posterior comparison: Whittle vs SNPE
#   6. Posterior predictive check

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from simulator import (
    generate_field,
    estimate_power_spectrum,
    summary_statistic,
    lorentzian_power_spectrum,
)
from inference import (
    grid_posterior,
    map_estimate,
    credible_interval,
    SNPEInference,
)

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SEED   = 42
N      = 128
A_TRUE = 1.0
K0     = 30.0
N_BINS = 30


# ------------------------------------------------------------------
# 1. Generate field and check power spectrum
# ------------------------------------------------------------------

field = generate_field(N, A_TRUE, K0, seed=SEED)
k_centers, Pk_est, Nmodes = estimate_power_spectrum(field, N, n_bins=N_BINS + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im = axes[0].imshow(field, cmap="RdBu_r", origin="lower")
axes[0].set_title(f"Simulated 2D Gaussian Field  (A={A_TRUE}, k0={K0})", fontsize=11)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
plt.colorbar(im, ax=axes[0], label="delta")

P_theory = lorentzian_power_spectrum(k_centers, A_TRUE, K0)
axes[1].plot(k_centers, Pk_est,   lw=1.5, color="#2c7bb6", label="Estimated P(k)")
axes[1].plot(k_centers, P_theory, lw=2.0, color="#d7191c", ls="--", label="True P(k)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("P(k)")
axes[1].set_title("Binned Power Spectrum", fontsize=11)
axes[1].legend()

fig.tight_layout()
fig.savefig("plots/01_field_and_spectrum.png")
plt.close(fig)
print("step 1 done")


# ------------------------------------------------------------------
# 2. Whittle likelihood posterior
# ------------------------------------------------------------------

A_grid, post_w, _ = grid_posterior(
    Pk_est, Nmodes, k_centers, K0, A_min=0.3, A_max=2.0, n_grid=300
)

A_map      = map_estimate(A_grid, post_w)
A_lo, A_hi = credible_interval(A_grid, post_w, level=0.68)
print(f"whittle MAP: {A_map:.3f}  68% CI: [{A_lo:.3f}, {A_hi:.3f}]")

norm_w = np.trapezoid(post_w, A_grid)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(A_grid, post_w / norm_w, lw=2, color="#2c7bb6", label="Whittle posterior")
ax.axvline(A_TRUE, color="#d7191c", lw=1.8, ls="--", label=f"True A={A_TRUE}")
ax.axvline(A_map,  color="#1a9641", lw=1.5, ls=":",  label=f"MAP={A_map:.3f}")
ax.fill_between(A_grid, post_w / norm_w,
                where=(A_grid >= A_lo) & (A_grid <= A_hi),
                alpha=0.25, color="#2c7bb6", label="68% CI")
ax.set_xlabel("A")
ax.set_ylabel("p(A | data)")
ax.set_title("Whittle Likelihood Posterior", fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig("plots/02_whittle_posterior.png")
plt.close(fig)
print("step 2 done")


# ------------------------------------------------------------------
# 3. Monte Carlo coverage (200 realisations)
# ------------------------------------------------------------------

rng_mc = np.random.default_rng(SEED + 1)
A_mc   = []

for _ in range(200):
    f          = generate_field(N, A_TRUE, K0, seed=rng_mc.integers(1_000_000))
    k_c, Pk_b, _ = estimate_power_spectrum(f, N, n_bins=N_BINS + 1)
    fk         = 1.0 / (1.0 + (k_c / K0) ** 2)
    A_mc.append(np.sum(Pk_b * fk) / np.sum(fk ** 2))

A_mc = np.array(A_mc)
print(f"MC mean: {A_mc.mean():.4f}  std: {A_mc.std():.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(A_mc, bins=20, color="#2c7bb6", edgecolor="white", alpha=0.85)
ax.axvline(A_TRUE,       color="#d7191c", lw=2,   ls="--", label=f"True A={A_TRUE}")
ax.axvline(A_mc.mean(),  color="#1a9641", lw=1.8, ls=":",  label=f"Mean={A_mc.mean():.3f}")
ax.set_xlabel("Recovered A")
ax.set_ylabel("Count")
ax.set_title("Monte Carlo Coverage", fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig("plots/03_mc_recovery.png")
plt.close(fig)
print("step 3 done")


# ------------------------------------------------------------------
# 4. SNPE-C
# ------------------------------------------------------------------

x_obs   = summary_statistic(field, N, n_bins=N_BINS)
x_obs_t = torch.tensor(x_obs, dtype=torch.float32)

snpe = SNPEInference(N=N, k0=K0, n_bins=N_BINS, A_min=0.3, A_max=2.0)
snpe.run(n_sims=2000, verbose=True)

samples_snpe = snpe.sample(x_obs_t, n_samples=5000).numpy().flatten()
print(f"SNPE mean: {samples_snpe.mean():.3f}  std: {samples_snpe.std():.3f}")

A_eval = np.linspace(0.3, 2.0, 300)
lp     = snpe.log_prob(A_eval, x_obs_t)
prob_s = np.exp(lp - lp.max())
prob_s /= np.trapezoid(prob_s, A_eval)

print("step 4 done")


# ------------------------------------------------------------------
# 5. Posterior comparison
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(A_grid, post_w / norm_w,
        lw=2.5, color="#2c7bb6",
        label=f"Whittle  (MAP={A_map:.3f})")
ax.plot(A_eval, prob_s,
        lw=2.5, color="#fd8d3c", ls="-.",
        label=f"SNPE-C  (mean={samples_snpe.mean():.3f})")
ax.axvline(A_TRUE, color="#d7191c", lw=2, ls="--", label=f"True A={A_TRUE}")

ax.set_xlabel("A", fontsize=12)
ax.set_ylabel("p(A | data)", fontsize=12)
ax.set_title("Posterior Comparison: Whittle vs SNPE-C", fontsize=12)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("plots/04_posterior_comparison.png")
plt.close(fig)
print("step 5 done")


# ------------------------------------------------------------------
# 6. Posterior predictive check
# ------------------------------------------------------------------

rng_ppc = np.random.default_rng(SEED + 99)
A_ppc   = rng_ppc.choice(samples_snpe, size=40, replace=False)

fig, ax = plt.subplots(figsize=(8, 4.5))

for A_s in A_ppc:
    f_s        = generate_field(N, max(float(A_s), 0.01), K0,
                                seed=int(rng_ppc.integers(1_000_000)))
    k_s, Pk_s, _ = estimate_power_spectrum(f_s, N, n_bins=N_BINS + 1)
    ax.plot(k_s, Pk_s, color="#fd8d3c", alpha=0.18, lw=0.9)

ax.plot(k_centers, Pk_est,
        lw=2.2, color="#2c7bb6", label="Observed P(k)", zorder=5)
ax.plot(k_centers, lorentzian_power_spectrum(k_centers, A_TRUE, K0),
        "k--", lw=1.5, label="True P(k)", zorder=5)
ax.set_xlabel("k")
ax.set_ylabel("P(k)")
ax.set_title("Posterior Predictive Check\norange = simulations from SNPE posterior", fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig("plots/05_posterior_predictive.png")
plt.close(fig)
print("step 6 done")


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

print("\n--- results ---")
print(f"true A                : {A_TRUE}")
print(f"MC least-squares      : {A_mc.mean():.3f} +/- {A_mc.std():.3f}")
print(f"Whittle MAP  (68% CI) : {A_map:.3f}  [{A_lo:.3f}, {A_hi:.3f}]")
print(f"SNPE mean +/- std     : {samples_snpe.mean():.3f} +/- {samples_snpe.std():.3f}")
