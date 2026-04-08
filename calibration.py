"""Calibration module for the Bates model."""
import numpy as np
import yfinance as yf
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
# Note: ThreadPoolExecutor is no longer needed — the FFT prices all strikes
# simultaneously in one vectorised call, making the per-strike thread pool
# approach redundant.

# Single source of truth for the calibration expiry date
EXPIRY_DATE = "2026-12-18"

# FFT grid constants (Carr-Madan 1999)
# N must be a power of 2; 4096 gives a dense enough log-strike grid for
# ATM-to-wing interpolation while keeping the FFT negligibly fast.
_FFT_N = 4096
_FFT_ETA = 0.25   # integration-variable step size


class BatesCalibrator:
    """Calibrates Bates model parameters via IFT."""

    def __init__(self, symbol, expiry=EXPIRY_DATE,
                 calls_df=None, prefetched_spot=None):
        """
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        expiry : str
            Option expiry date string (YYYY-MM-DD).
        calls_df : pd.DataFrame, optional
            Pre-fetched calls chain passed in from the dashboard's
            @st.cache_data layer (ttl=3600).  When supplied together with
            prefetched_spot, the constructor makes **zero** yfinance calls,
            which fully protects the 3-D vol surface from rate-limit
            exhaustion on rapid re-calibrations.
        prefetched_spot : float, optional
            Spot price already retrieved from the dashboard's cached data.
        """
        self.symbol = symbol
        self.expiry = expiry
        self.r = 0.05
        self.spot = 550.0  # Default, overwritten by live data
        self.strikes = np.linspace(500, 600, 20)
        self.market_prices = np.linspace(60, 10, 20)

        if calls_df is not None and prefetched_spot is not None:
            # Use data already fetched by the dashboard cache — no yfinance call.
            # yfinance live streaming still occurs; it is just mediated through
            # Streamlit's cache so one HTTP round-trip covers the whole session.
            self.spot = prefetched_spot
            self._process_chain(calls_df)
        else:
            try:
                tk = yf.Ticker(symbol)
                self.spot = float(tk.fast_info['lastPrice'])
                chain = tk.option_chain(expiry)
                self._process_chain(chain.calls)
                print(f"Bates Engine: {symbol} at ${self.spot:.2f}")
            except Exception:  # pylint: disable=broad-except
                self.spot = 550.0
                self.strikes = np.linspace(500, 600, 20)
                self.market_prices = np.linspace(60, 10, 20)

    def _process_chain(self, calls):
        """Extract ATM-centred strike window from a calls DataFrame."""
        # --- THE SMIRK FIX: CENTER STRIKES ON SPOT ---
        idx = (np.abs(calls['strike'] - self.spot)).argmin()
        buffer = 15
        start_idx = max(0, idx - buffer)
        end_idx = min(len(calls), idx + buffer)
        self.strikes = calls['strike'].values[start_idx:end_idx]
        self.market_prices = calls['lastPrice'].values[start_idx:end_idx]

    def bates_char_func(self, u, t_years, params):
        """
        Bates Characteristic Function.

        Accepts scalar or NumPy array `u` (including complex-valued arrays).
        All operations broadcast element-wise, so passing the full FFT
        frequency grid as a single array evaluates all N points in one
        vectorised call — no Python loop required.
        """
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

    # ------------------------------------------------------------------
    # FFT pricing  (Carr-Madan 1999)
    # ------------------------------------------------------------------
    def price_options_fft(self, t_years, params,
                          alpha=1.5, N=_FFT_N, eta=_FFT_ETA):
        """
        Price the full strike grid in O(N log N) using the Carr-Madan (1999)
        FFT method.

        Algorithm
        ---------
        1. Build the integration-variable grid  u_j = j * η,  j = 0 … N-1.
        2. Evaluate the Bates characteristic function at  u - (α+1)i  for
           the entire grid **in one vectorised NumPy call** (no Python loop).
        3. Form the Carr-Madan modified payoff transform ψ(u).
        4. Apply Simpson-rule weights for improved quadrature accuracy.
        5. Execute a single `np.fft.fft` to obtain call prices at N
           log-strikes simultaneously.
        6. Interpolate onto the actual market log-strikes.

        Parameters
        ----------
        alpha : float
            Damping exponent (1.5 is the standard choice; must satisfy
            α > 0 and α(α+1) < ∞ under the risk-neutral measure).
        N : int
            FFT grid size — must be a power of 2.
        eta : float
            Integration-variable step size η.  The log-strike spacing
            λ = 2π / (N η) is determined automatically.

        Returns
        -------
        np.ndarray
            Model call prices interpolated at self.strikes, floored at 0.
        """
        lam = 2.0 * np.pi / (N * eta)   # log-strike grid spacing
        b   = N * lam / 2.0              # grid half-width (centres around ATM)

        # Integration grid and log-strike grid
        j       = np.arange(N)
        u       = j * eta
        k_log   = -b + lam * j

        # ── Step 1: vectorised characteristic function evaluation ─────────
        # The entire N-point frequency grid is passed as one complex array;
        # bates_char_func broadcasts element-wise so no Python loop is needed.
        cf_vals = self.bates_char_func(u - (alpha + 1) * 1j, t_years, params)

        # ── Step 2: Carr-Madan modified payoff transform ──────────────────
        # Denominator is non-zero for all u when α > 0.
        denom = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u
        psi   = np.exp(-self.r * t_years) * cf_vals / denom

        # ── Step 3: Simpson's rule weights for O(η⁴) quadrature error ─────
        simpson = (eta / 3.0) * (3.0 + (-1.0)**j - (j == 0).astype(float))

        # ── Step 4: single FFT ────────────────────────────────────────────
        fft_input  = np.exp(1j * b * u) * psi * simpson
        fft_output = np.real(np.fft.fft(fft_input))

        # ── Step 5: recover and interpolate call prices ───────────────────
        call_prices = (np.exp(-alpha * k_log) / np.pi) * fft_output
        log_strikes_mkt = np.log(self.strikes)
        model_prices = np.interp(log_strikes_mkt, k_log, call_prices)

        return np.maximum(model_prices, 0.0)

    # ------------------------------------------------------------------
    # Single-strike quad pricer (retained for diagnostics / unit tests)
    # ------------------------------------------------------------------
    def price_option(self, k_strike, t_years, params, alpha=1.5):
        """
        Single-strike call price via scalar Carr-Madan quadrature.

        Uses the same modified payoff transform ψ(u) as price_options_fft
        so both methods are numerically consistent.  Retained for spot-
        checks and unit tests; calibration always uses the FFT path.

        Note
        ----
        The original formula (S0 - √(S0K)/π · ∫ …) evaluated only the
        Π₁ term of the Heston two-integral decomposition and omitted the
        discounted Π₂ leg, producing prices larger than the spot.  This
        scalar version corrects that by using the full Carr-Madan integrand.
        """
        k = np.log(k_strike)

        def _integrand_scalar(u):
            cf = self.bates_char_func(u - (alpha + 1) * 1j, t_years, params)
            denom = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u
            psi = np.exp(-self.r * t_years) * cf / denom
            return np.real(np.exp(-1j * u * k) * psi)

        integral, _ = quad(_integrand_scalar, 0, 500)
        return max(np.exp(-alpha * k) / np.pi * integral, 0.0)

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------
    def objective_function(self, params):
        """
        MSE between market and model prices.

        Prices **all strikes in one FFT call** (O(N log N)) rather than
        running a per-strike quad integration loop.  This is the primary
        speed-up for the L-BFGS-B and differential_evolution passes.
        """
        try:
            model_prices = self.price_options_fft(1.0, params)
            sq_errors = (model_prices - self.market_prices) ** 2
            return float(np.mean(sq_errors))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return 1e6

    def fit(self):
        """Dual-pass calibration logic."""
        bounds = [
            (0.1, 4.0), (0.01, 0.3), (0.1, 0.8), (-0.95, -0.1),
            (0.05, 0.8), (-0.4, -0.01), (0.01, 0.4)
        ]
        g_res = differential_evolution(
            self.objective_function, bounds, popsize=3, tol=0.1
        )
        res = minimize(
            self.objective_function, g_res.x, bounds=bounds,
            method='L-BFGS-B', options={'ftol': 1e-4}
        )
        return res.x
