import numpy as np
import os
from dotenv import load_dotenv

# Security: Load parameters from the private .env file
load_dotenv()

class BatesModelEngine:
    def __init__(self):
        # Primary Parameters fetched from .env (Security Layer)
        self.S0 = float(os.getenv("S0", 100.0))
        self.r = float(os.getenv("RISK_FREE_RATE", 0.05))
        self.kappa = float(os.getenv("KAPPA", 2.0))      # Speed of mean reversion
        self.theta = float(os.getenv("THETA", 0.04))    # Long-run average volatility
        self.sigma_v = float(os.getenv("SIGMA_V", 0.3)) # Volatility of volatility
        self.rho = float(os.getenv("RHO", -0.7))        # Correlation
        self.lamb = float(os.getenv("LAMBDA_JUMP", 0.1)) # Jump intensity
        self.mu_j = float(os.getenv("MU_JUMP", -0.05))   # Mean jump size
        self.delta_j = float(os.getenv("DELTA_JUMP", 0.1)) # Jump volatility

    def simulate_path(self, T=1.75, steps=504):
        """
        Monte Carlo Simulation for Q1 2026 - Q4 2027 (approx 1.75 years)
        """
        dt = T / steps
        # Expected jump size correction (k_bar)
        k_bar = np.exp(self.mu_j + 0.5 * self.delta_j**2) - 1
        
        # Initialize paths
        S = np.zeros(steps + 1)
        V = np.zeros(steps + 1)
        S[0] = self.S0
        V[0] = self.theta # Start at long-run mean
        
        for t in range(1, steps + 1):
            # Correlated Brownian Motions
            Z1 = np.random.standard_normal()
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal()
            
            # Merton Jump Component
            N = np.random.poisson(self.lamb * dt)
            Jumps = np.sum(np.random.normal(self.mu_j, self.delta_j, N)) if N > 0 else 0
            
            # Heston Variance Process (Reflected at zero for stability)
            V[t] = np.maximum(V[t-1] + self.kappa * (self.theta - V[t-1]) * dt + 
                              self.sigma_v * np.sqrt(V[t-1]) * np.sqrt(dt) * Z2, 1e-6)
            
            # Bates Price Process
            drift = (self.r - self.lamb * k_bar - 0.5 * V[t-1]) * dt
            diffusion = np.sqrt(V[t-1] * dt) * Z1
            S[t] = S[t-1] * np.exp(drift + diffusion + Jumps)
            
        return S