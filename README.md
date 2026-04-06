![Status](https://img.shields.io/badge/status-complete-success)
![Python](https://img.shields.io/badge/python-3.14-blue)
![Math](https://img.shields.io/badge/model-Bates%20(1996)-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Security](https://img.shields.io/badge/security-Bandit%20Scanned-yellow)

# Stochastic-Jump-Pricer-2026: Bates Model Core

**Live Dashboard**: [View 2026-2027 Interactive Simulation](https://stochastic-jump-pricer-2026-7xksu8bhonxziqjrvgwbas.streamlit.app/)

## Introduction
A full-stack financial engineering suite implementing the **Bates (1996) Model**. This project captures complex asset dynamics for the volatility window between **Q1 of 2026 and Q4 of 2027** by integrating **Heston Stochastic Volatility** with **Merton's Jump-Diffusion** via high-speed Fourier Inversion.

## Methodology: The Dual-Engine Approach
To achieve institutional-grade performance, the engine bifurcates the pricing and simulation tasks:

* **Calibration via Fourier Inversion**: We utilise **Numerical Quadrature Integration** of the characteristic function for model calibration. Unlike Monte Carlo, this semi-analytical approach provides near-instantaneous pricing, allowing the **L-BFGS-B optimiser** to converge on market parameters ($\kappa, \sigma_v, \rho$) in milliseconds rather than minutes.
* **Projection via Monte Carlo**: While Fourier methods are superior for pricing, they do not reveal the "path-dependency" of the asset. We maintain a **Monte Carlo engine** for the 2026-2027 projections to visualise the **discontinuous jumps** and **stochastic volatility clustering** that characterise real-world market shocks.

## Mathematical Architecture ($D + J$)
The Bates model evaluates European option prices by treating the characteristic function ($\phi$) as a product of diffusion and jump components:

$$\phi_{X_{t}}^{Bates}(z) = \phi_{X_{t}}^{Heston}(z) \times \exp\left(t\lambda\left(\exp\left(-\frac{\delta^{2}z^{2}}{2} + i\mu z\right) - 1\right)\right)$$

### 1. High-Speed Calibration (Fourier Inversion)
This engine utilise **Numerical Quadrature Integration** of the characteristic function. This allows for near-instantaneous calibration of parameters ($\kappa, \theta, \sigma_v, \rho, \lambda$) against live market smiles.

### 2. Live Market Integration
The system fetches real-time 2026–2027 option chains via the **yfinance API**, allowing users to observe the calibrated "Volatility Smirk" for assets including **SPY, AAPL, TSLA, and XOM**.

## Technical Stack
* **Engine**: Python (NumPy, SciPy) utilising **Inverse Fourier Transforms** for precise pricing and **Euler-Maruyama discretisation** for path-dependent simulations.
* **Data**: `yfinance` API for real-time 2026–2027 option chain retrieval and market smile extraction.
* **Frontend**: Streamlit & Plotly for interactive 3D Volatility Surface visualisation.
* **Optimisation**: High-speed **L-BFGS-B** routine for multi-parameter model fitting against live market data.

## Security Posture
> **Note**: This repository utilises automated dependency monitoring and Bandit-based security linting. All model parameters are isolated in a non-tracked environment to ensure the integrity of the 2026-2027 simulation data.

---
*Reference: Fadugba, S. E. & Okunlola, J. T. (2014). On the Combination of Merton and Heston Models in the Theory of Option Pricing. International Journal of Applied Science and Mathematics.*

![Status](https://img.shields.io/badge/status-complete-success) ![License](https://img.shields.io/badge/license-MIT-green)