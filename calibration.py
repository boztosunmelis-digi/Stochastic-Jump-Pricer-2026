"""Calibration module for the Bates stochastic volatility
jump-diffusion model."""
import numpy as np
import yfinance as yf
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution


class BatesCalibrator:
    """Calibrates the Bates model to real market option prices via IFT."""
    def __init__(self, symbol="SPY", expiry="2026-12-18"):
        self.symbol = symbol
        self.expiry = expiry
        self.r = 0.05

        try:
            # Attempt to pull real 2026 market data
            tk = yf.Ticker(symbol)
            chain = tk.option_chain(expiry)

            # Success: Use real market data
            self.strikes = chain.calls['strike'].values[10:30]
            self.market_prices = chain.calls['lastPrice'].values[10:30]
            self.s0 = tk.fast_info['lastPrice']
            print(f"Successfully fetched live data for {symbol}")

        except Exception as e:  # pylint: disable=broad-except
            # Fallback: Yahoo is rate-limiting us or the IP is blocked
            print(
                f"Rate Limit/Data Error: {e}. "
                "Switching to Synthetic 2026 Smile."
            )
            self.s0 = 100.0
            self.strikes = np.linspace(80, 120, 20)
            self.market_prices = [
                max(self.s0 - k, 0) + (100 - k)**2 / 400 + 1.5
                for k in self.strikes
            ]

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
        """Dual-pass calibration: Global search followed by
        local refinement."""
        # Standardised 2026-2027 Equity Bounds
        bounds = [
            (0.1, 4.0),    # kappa
            (0.01, 0.3),   # theta
            (0.1, 0.8),    # sig_v
            (-0.99, -0.1),  # rho (Forced negative for SPY/AAPL)
            (0.0, 0.8),    # lamb
            (-0.4, 0.0),   # mu_j
            (0.01, 0.4)    # del_j
        ]

        # Pass 1: Global "Scout" (Prevents getting stuck at 0.00 correlation)
        global_res = differential_evolution(
            self.objective_function,
            bounds,
            popsize=5,  # Efficiency over brute force
            tol=0.1
        )

        # Pass 2: Local "Sniper" (Refines the result for high precision)
        res = minimize(
            self.objective_function,
            global_res.x,
            bounds=bounds,
            method='L-BFGS-B',
            options={'ftol': 1e-6}  # Increase precision
        )
        return res.x
