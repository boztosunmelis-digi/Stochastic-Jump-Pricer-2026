"""Bates stochastic volatility jump-diffusion model simulation engine."""
import os
import numpy as np
from dotenv import load_dotenv

# Security: Load parameters from the private .env file
load_dotenv()


class BatesModelEngine:
    """Simulation engine for the Bates stochastic volatility
    jump-diffusion model."""

    def __init__(self):
        # Primary Parameters fetched from .env (Security Layer)
        self.s0 = float(os.getenv("S0", "100.0"))
        self.r = float(os.getenv("RISK_FREE_RATE", "0.05"))
        self.kappa = float(
            os.getenv("KAPPA", "2.0")
        )  # Speed of mean reversion
        self.theta = float(
            os.getenv("THETA", "0.04")
        )  # Long-run average volatility
        self.sigma_v = float(
            os.getenv("SIGMA_V", "0.3")
        )  # Volatility of volatility
        self.rho = float(os.getenv("RHO", "-0.7"))  # Correlation
        self.lamb = float(
            os.getenv("LAMBDA_JUMP", "0.1")
        )  # Jump intensity
        self.mu_j = float(os.getenv("MU_JUMP", "-0.05"))  # Mean jump size
        self.delta_j = float(
            os.getenv("DELTA_JUMP", "0.1")
        )  # Jump volatility

    def get_params(self):
        """Return current model parameters as a dictionary."""
        return {
            "S0": self.s0, "r": self.r, "kappa": self.kappa,
            "theta": self.theta, "sigma_v": self.sigma_v,
            "rho": self.rho, "lamb": self.lamb,
            "mu_j": self.mu_j, "delta_j": self.delta_j,
        }

    def simulate_path(self, t_years=1.75, steps=504):
        """
        Single-path Monte Carlo Simulation for Q1 2026 - Q4 2027.
        All random draws are pre-generated in bulk for performance.
        """
        dt = t_years / steps
        # Expected jump size correction (k_bar)
        k_bar = np.exp(self.mu_j + 0.5 * self.delta_j**2) - 1

        # Pre-generate all random numbers at once (avoids per-step overhead)
        z1 = np.random.standard_normal(steps)
        z2 = (
            self.rho * z1
            + np.sqrt(1 - self.rho**2) * np.random.standard_normal(steps)
        )
        n_jumps_arr = np.random.poisson(self.lamb * dt, steps)

        prices = np.zeros(steps + 1)
        variance = np.zeros(steps + 1)
        prices[0] = self.s0
        variance[0] = self.theta  # Start at long-run mean

        for t in range(1, steps + 1):
            # Merton Jump Component
            nj = n_jumps_arr[t - 1]
            jump = (
                np.sum(np.random.normal(self.mu_j, self.delta_j, nj))
                if nj > 0 else 0.0
            )

            # Heston Variance Process (full-truncation for numerical stability)
            v_prev = max(variance[t - 1], 0.0)
            variance[t] = np.maximum(
                v_prev + self.kappa * (self.theta - v_prev) * dt
                + self.sigma_v * np.sqrt(v_prev * dt) * z2[t - 1],
                1e-6
            )

            # Bates Price Process
            drift = (self.r - self.lamb * k_bar - 0.5 * v_prev) * dt
            diffusion = np.sqrt(v_prev * dt) * z1[t - 1]
            prices[t] = prices[t - 1] * np.exp(drift + diffusion + jump)

        return prices

    def simulate_paths(self, n_paths=50, t_years=1.75, steps=504):
        """
        Batch Monte Carlo: generate n_paths simultaneously using vectorised
        NumPy operations across the path dimension.

        All random draws are pre-allocated as (steps, n_paths) arrays so that
        each time-step update operates on the entire path ensemble at once,
        replacing the outer Python loop over individual paths with a single
        vectorised step.

        Returns
        -------
        np.ndarray, shape (n_paths, steps + 1)
        """
        dt = t_years / steps
        k_bar = np.exp(self.mu_j + 0.5 * self.delta_j**2) - 1

        # Pre-generate all random draws: shape (steps, n_paths)
        z1 = np.random.standard_normal((steps, n_paths))
        z2 = (
            self.rho * z1
            + np.sqrt(1 - self.rho**2) * np.random.standard_normal((steps, n_paths))
        )
        # Poisson jump counts per (step, path)
        n_jumps = np.random.poisson(self.lamb * dt, (steps, n_paths))

        prices = np.zeros((steps + 1, n_paths))
        variance = np.zeros((steps + 1, n_paths))
        prices[0] = self.s0
        variance[0] = self.theta

        for t in range(1, steps + 1):
            nj = n_jumps[t - 1]                  # (n_paths,)
            v_prev = np.maximum(variance[t - 1], 0.0)

            # Jump component: sum of nj i.i.d. Normal(mu_j, delta_j)
            # = Normal(nj * mu_j, sqrt(nj) * delta_j) — exact result, not approx
            jumps = np.zeros(n_paths)
            jump_mask = nj > 0
            if jump_mask.any():
                nj_active = nj[jump_mask].astype(float)
                jumps[jump_mask] = np.random.normal(
                    self.mu_j * nj_active,
                    self.delta_j * np.sqrt(nj_active)
                )

            # Heston Variance Process (vectorised across all paths)
            variance[t] = np.maximum(
                v_prev + self.kappa * (self.theta - v_prev) * dt
                + self.sigma_v * np.sqrt(v_prev * dt) * z2[t - 1],
                1e-6
            )

            # Bates Price Process (vectorised across all paths)
            drift = (self.r - self.lamb * k_bar - 0.5 * v_prev) * dt
            diffusion = np.sqrt(v_prev * dt) * z1[t - 1]
            prices[t] = prices[t - 1] * np.exp(drift + diffusion + jumps)

        return prices.T  # → (n_paths, steps + 1)
