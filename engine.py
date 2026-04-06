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
        Monte Carlo Simulation for Q1 2026 - Q4 2027 (approx 1.75 years)
        """
        dt = t_years / steps
        # Expected jump size correction (k_bar)
        k_bar = np.exp(self.mu_j + 0.5 * self.delta_j**2) - 1

        # Initialize paths
        prices = np.zeros(steps + 1)
        variance = np.zeros(steps + 1)
        prices[0] = self.s0
        variance[0] = self.theta  # Start at long-run mean

        for t in range(1, steps + 1):
            # Correlated Brownian Motions
            z1 = np.random.standard_normal()
            z2 = (
                self.rho * z1
                + np.sqrt(1 - self.rho**2) * np.random.standard_normal()
            )

            # Merton Jump Component
            n_jumps = np.random.poisson(self.lamb * dt)
            jumps = (
                np.sum(np.random.normal(self.mu_j, self.delta_j, n_jumps))
                if n_jumps > 0 else 0
            )

            # Heston Variance Process (Reflected at zero for stability)
            variance[t] = np.maximum(
                variance[t-1] + self.kappa * (self.theta - variance[t-1]) * dt
                + self.sigma_v * np.sqrt(variance[t-1]) * np.sqrt(dt) * z2,
                1e-6
            )

            # Bates Price Process
            drift = (self.r - self.lamb * k_bar - 0.5 * variance[t-1]) * dt
            diffusion = np.sqrt(variance[t-1] * dt) * z1
            prices[t] = prices[t-1] * np.exp(drift + diffusion + jumps)

        return prices
