import numpy as np
from scipy.optimize import minimize
from engine import BatesModelEngine

class BatesCalibrator:
    """
    Fits the Bates Model to market-observed option prices.
    Reference: Fadugba & Okunlola (2014)
    """
    def __init__(self, market_prices, strikes, expiry):
        self.market_prices = market_prices
        self.strikes = strikes
        self.expiry = expiry
        self.engine = BatesModelEngine()

    def objective_function(self, params):
        # Unpack parameters for the optimizer
        kappa, theta, sigma_v, rho, lamb = params
        
        # Temporarily update engine parameters for testing
        self.engine.kappa = kappa
        self.engine.theta = theta
        self.engine.sigma_v = sigma_v
        self.engine.rho = rho
        self.engine.lamb = lamb
        
        errors = []
        for i, K in enumerate(self.strikes):
            # Monte Carlo pricing for calibration (Stable but slow)
            # In production, use Fourier Inversion for speed
            paths = [self.engine.simulate_path(T=self.expiry)[-1] for _ in range(300)]
            model_price = np.mean(np.maximum(np.array(paths) - K, 0))
            errors.append((model_price - self.market_prices[i])**2)
            
        return np.sum(errors)

    def fit(self):
        # Initial guesses: [kappa, theta, sigma_v, rho, lamb]
        init_guess = [2.0, 0.04, 0.3, -0.7, 0.1]
        bounds = [(0.1, 5.0), (0.01, 0.2), (0.1, 1.0), (-0.95, 0.0), (0.0, 1.0)]
        
        print("Starting calibration... this may take a moment.")
        res = minimize(self.objective_function, init_guess, bounds=bounds, method='L-BFGS-B')
        return res.x

if __name__ == "__main__":
    # Example usage for your README/Portfolio
    print("Bates Calibrator Initialized.")