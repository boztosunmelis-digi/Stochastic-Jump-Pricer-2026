"""Calibration module for the Bates stochastic volatility
jump-diffusion model."""
import numpy as np
import yfinance as yf
from scipy.integrate import quad
from scipy.optimize import minimize


class BatesCalibrator:
    """Calibrates the Bates model to real market option prices via IFT."""
    def __init__(self, symbol="SPY", expiry="2026-12-18"):
        self.symbol = symbol
        self.expiry = expiry

        # Pull real 2026 market data
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(expiry)

        # We take a slice of strikes to keep the optimisation snappy
        self.strikes = chain.calls['strike'].values[10:30]
        self.market_prices = chain.calls['lastPrice'].values[10:30]
        self.s0 = tk.fast_info['lastPrice']

    def bates_char_func(self, u, t_years, params):
        """Bates (1996) Characteristic Function."""
        kappa, theta, sig_v, rho, lamb, mu_j, del_j = params
        r = 0.05

        # Heston Component
        gamma = np.sqrt(
            (kappa - 1j*rho*sig_v*u)**2
            + (sig_v**2)*(u**2 + 1j*u)
        )
        exp_gt = np.exp(gamma * t_years)
        ratio = (
            (kappa - 1j*rho*sig_v*u - gamma)
            / (kappa - 1j*rho*sig_v*u + gamma)
        )
        d_num = kappa - 1j*rho*sig_v*u - gamma
        d_val = (
            d_num / sig_v**2
            * ((1 - exp_gt) / (1 - ratio * exp_gt))
        )
        c_val = (
            (kappa * theta) / (sig_v**2)
            * (
                d_num * t_years
                - 2 * np.log((1 - ratio * exp_gt) / (1 - ratio))
            )
        )

        # Merton Jump Component
        jump = t_years * lamb * (
            np.exp(1j*u*mu_j - 0.5*del_j**2 * u**2) - 1
        )

        return np.exp(
            c_val + d_val * theta + jump
            + 1j*u*np.log(self.s0 * np.exp(r*t_years))
        )

    def _integrand(self, u, k_strike, t_years, params):
        """Integrand for the inverse Fourier transform pricing."""
        return np.real(
            np.exp(-1j * u * np.log(k_strike))
            * self.bates_char_func(u - 1j, t_years, params)
            / (1j * u * self.bates_char_func(-1j, t_years, params))
        )

    def price_option(self, k_strike, t_years, params):
        """Semi-analytical pricing via Inverse Fourier Transform."""
        # Integration from 0 to 100 is usually sufficient for convergence
        integral, _ = quad(self._integrand, 0, 100,
                           args=(k_strike, t_years, params))
        return self.s0 - (np.sqrt(self.s0 * k_strike) / np.pi) * integral

    def objective_function(self, params):
        """Compute MSE between market prices and Fourier-integrated prices."""
        errors = []
        t_years = 1.0  # Time to expiry in years
        for i, k_strike in enumerate(self.strikes):
            try:
                model_price = self.price_option(k_strike, t_years, params)
                errors.append((model_price - self.market_prices[i])**2)
            except ValueError:  # Penalty for non-converging math
                errors.append(1e6)
        return np.mean(errors)

    def fit(self):
        """Run L-BFGS-B optimisation with physical bounds."""
        # Initial guess: [kappa, theta, sig_v, rho, lamb, mu_j, del_j]
        init_guess = [2.0, 0.04, 0.3, -0.7, 0.1, -0.05, 0.1]

        # Bounds ensure the model stays mathematically sound
        bounds = [
            (0.1, 5.0), (0.01, 0.2), (0.1, 1.0),
            (-0.95, 0.0), (0.0, 1.0), (-0.5, 0.0), (0.01, 0.5)
        ]

        res = minimize(
            self.objective_function,
            init_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        return res.x
