# inference_scratch.py
# First attempt at NPE implemented from scratch in NumPy, before using the sbi package.
# Wrote this to understand the core idea before letting a library handle it.
#
# The network output is (mean, log_sigma), so the posterior approximation is Gaussian.
# This is the main limitation compared to inference.py where the normalising flow
# can represent any posterior shape.
# Works reasonably well here because the true posterior is roughly Gaussian.

import numpy as np
from simulator import generate_field, summary_statistic


class MLP:
    # Minimal MLP in pure NumPy with ReLU activations and He initialisation.

    def __init__(self, layer_sizes, learning_rate=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            W = rng.normal(0, np.sqrt(2.0 / fan_in), (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)

        self.lr = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        activations = [X]
        zs = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            zs.append(z)
            if i < len(self.weights) - 1:
                activations.append(self.relu(z))
            else:
                activations.append(z)
        return activations, zs

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def train_step(self, X, y):
        activations, zs = self.forward(X)
        y_hat = activations[-1]
        loss = np.mean((y_hat - y) ** 2)

        delta = 2 * (y_hat - y) / len(y)
        grads_W = []
        grads_b = []

        for i in reversed(range(len(self.weights))):
            grads_W.insert(0, activations[i].T @ delta)
            grads_b.insert(0, delta.sum(axis=0))
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_grad(zs[i - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

        return loss


class NeuralPosteriorEstimator:
    # Same simulate -> train -> infer workflow as SNPEInference in inference.py
    # but with a simpler Gaussian output head instead of a normalising flow.

    def __init__(self, N, k0, A_prior=(0.1, 3.0), n_bins=30, hidden=(64, 64), lr=3e-3):
        self.N = N
        self.k0 = k0
        self.A_prior = A_prior
        self.n_bins = n_bins
        self.net = MLP([n_bins] + list(hidden) + [2], lr)
        self.x_mean = None
        self.x_std = None
        self.A_mean = None
        self.A_std = None

    def simulate(self, n_sims, seed=None):
        rng = np.random.default_rng(seed)
        A_min, A_max = self.A_prior
        A_samples = rng.uniform(A_min, A_max, n_sims)

        X = []
        for i in range(n_sims):
            A = A_samples[i]
            field = generate_field(self.N, A, self.k0, seed=int(rng.integers(1e9)))
            x = summary_statistic(field, self.N, n_bins=self.n_bins)
            X.append(x)

        return A_samples, np.array(X)

    def train(self, A_samples, X, n_epochs=200, batch_size=64, verbose=True):
        # Normalise inputs and targets before training
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.x_mean) / self.x_std

        self.A_mean = A_samples.mean()
        self.A_std = A_samples.std() + 1e-8
        A_norm = (A_samples - self.A_mean) / self.A_std

        n = len(A_samples)
        rng = np.random.default_rng(42)

        for epoch in range(n_epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                X_b = X_norm[batch_idx]
                y_b = np.stack([A_norm[batch_idx], np.zeros(len(batch_idx))], axis=1)
                loss = self.net.train_step(X_b, y_b)
                epoch_loss += loss
                n_batches += 1

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}  loss={epoch_loss/n_batches:.4f}")

    def infer(self, x_obs):
        x_norm = (x_obs - self.x_mean) / self.x_std
        out = self.net.predict(x_norm[None, :])
        mu_A = out[0, 0] * self.A_std + self.A_mean
        sigma_A = np.exp(out[0, 1]) * self.A_std
        return float(mu_A), float(sigma_A)

    def posterior_samples(self, x_obs, n_samples=2000, seed=None):
        rng = np.random.default_rng(seed)
        mu, sigma = self.infer(x_obs)
        return rng.normal(mu, sigma, n_samples)
