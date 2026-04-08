"""Calibration module for the Bates model."""
import numpy as np
import yfinance as yf
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution


class BatesCalibrator:
    """Calibrates Bates model parameters via IFT."""

    def __init__(self, symbol, expiry="2026-12-18"):
        self.symbol = symbol
        self.expiry = expiry
        self.r = 0.05
        self.spot = 550.0  # Default, overwritten by live data
        self.strikes = np.linspace(500, 600, 20)
        self.market_prices = np.linspace(60, 10, 20)
        try:
            tk = yf.Ticker(symbol)
            self.spot = float(tk.fast_info['lastPrice'])
            chain = tk.option_chain(expiry)
            calls = chain.calls

            # --- THE SMIRK FIX: CENTER STRIKES ON SPOT ---
            idx = (np.abs(calls['strike'] - self.spot)).argmin()
            buffer = 15
            start_idx = max(0, idx - buffer)
            end_idx = min(len(calls), idx + buffer)

            self.strikes = calls['strike'].values[start_idx:end_idx]
            self.market_prices = calls['lastPrice'].values[start_idx:end_idx]
            print(f"Bates Engine: {symbol} at ${self.spot:.2f}")
        except Exception:  # pylint: disable=broad-except
            self.spot = 550.0
            self.strikes = np.linspace(500, 600, 20)
            self.market_prices = np.linspace(60, 10, 20)

    def bates_char_func(self, u, t_years, params):
        """Bates Characteristic Function."""
        kappa, theta, sig_v, rho, lamb, mu_j, del_j = params
        r = self.r
        gamma = np.sqrt(
            (kappa - 1j*rho*sig_v*u)**2 + (sig_v**2)*(u**2 + 1j*u)
        )
        exp_gt = np.exp(gamma * t_years)
        ratio = (
            (kappa - 1j*rho*sig_v*u - gamma) /
            (kappa - 1j*rho*sig_v*u + gamma)
        )
        d_num = kappa - 1j*rho*sig_v*u - gamma
        d_val = (
            d_num / sig_v**2 * ((1 - exp_gt) / (1 - ratio * exp_gt))
        )
        c_val = (
            (kappa * theta) / (sig_v**2) *
            (d_num * t_years - 2 * np.log((1 - ratio * exp_gt) / (1 - ratio)))
        )
        jump = t_years * lamb * (
            np.exp(1j*u*mu_j - 0.5*del_j**2 * u**2) - 1
        )
        return np.exp(
            c_val + d_val * theta + jump + 1j*u *
            np.log(self.spot * np.exp(r*t_years))
        )

    def _integrand(self, u, k_strike, t_years, params):
        """Integrand for the inverse Fourier transform."""
        cf_u = self.bates_char_func(u - 1j, t_years, params)
        cf_den = self.bates_char_func(-1j, t_years, params)
        return np.real(
            np.exp(-1j * u * np.log(k_strike)) * cf_u / (1j * u * cf_den)
        )

    def price_option(self, k_strike, t_years, params):
        """Analytical pricing via Fourier Transform."""
        integral, _ = quad(
            self._integrand, 0, 60, args=(k_strike, t_years, params)
        )
        return self.spot - (np.sqrt(self.spot * k_strike) / np.pi) * integral

    def objective_function(self, params):
        """MSE between market and model prices."""
        errors = []
        for i, k_strike in enumerate(self.strikes):
            try:
                m_price = self.price_option(k_strike, 1.0, params)
                errors.append((m_price - self.market_prices[i])**2)
            except (ValueError, ZeroDivisionError):
                errors.append(1e6)
        return np.mean(errors)

    def fit(self):
        """Dual-pass calibration logic."""
        bounds = [
            (0.1, 4.0), (0.01, 0.3), (0.1, 0.8), (-0.95, -0.1),
            (0.0, 0.8), (-0.4, 0.0), (0.01, 0.4)
        ]
        g_res = differential_evolution(
            self.objective_function, bounds, popsize=3, tol=0.1
        )
        res = minimize(
            self.objective_function, g_res.x, bounds=bounds,
            method='L-BFGS-B', options={'ftol': 1e-4}
        )
        return res.x
