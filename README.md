![Status](https://img.shields.io/badge/status-complete-success)
![Python](https://img.shields.io/badge/python-3.14-blue)
![Math](https://img.shields.io/badge/model-Bates%20(1996)-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Security](https://img.shields.io/badge/security-Bandit%20Scanned-yellow)

# Stochastic-Jump-Pricer-2026: Bates Model Core

**Live Dashboard**: [View 2026-2027 Interactive Simulation]
https://stochastic-jump-pricer-2026-7xksu8bhonxziqjrvgwbas.streamlit.app/

## Introduction
A full-stack financial engineering suite implementing the **Bates (1996) Model**. This project captures complex asset dynamics for the volatility window between **Q1 of 2026 and Q4 of 2027** by integrating **Heston Stochastic Volatility** with **Merton's Jump-Diffusion** via high-speed Fourier Inversion.

## Overview
This engine simulates and prices assets by capturing two critical features of financial returns:

1. **Mean-Reverting Volatility**: Utilising the Heston model to ensure volatility evolves in a stochastic, mean-reverting fashion.
2. **Discontinuous Jumps**: Implementing Merton’s jump model to account for sudden outliers and "fat tails" driven by public information shocks.

## Mathematical Architecture ($D + J$)
The Bates model evaluates European option prices by treating the characteristic function ($\phi$) as a product of diffusion and jump components:

$$\phi_{X_{t}}^{Bates}(z) = \phi_{X_{t}}^{Heston}(z) \times \exp\left(t\lambda\left(\exp\left(-\frac{\delta^{2}z^{2}}{2} + i\mu z\right) - 1\right)\right)$$

### 1. High-Speed Calibration (Fourier Inversion)
Unlike traditional Monte Carlo methods, this engine utilises **Numerical Quadrature Integration** of the characteristic function. This allows for near-instantaneous calibration of parameters ($\kappa, \theta, \sigma_v, \rho, \lambda$) against live market smiles.

### 2. Live Market Integration
The system fetches real-time 2026-2027 option chains via the **yfinance API**, allowing users to observe the calibrated "Volatility Smirk" for assets including **SPY, AAPL, TSLA, and XOM**.

## Technical Stack
* **Engine**: Python (NumPy, SciPy) using **Inverse Fourier Transforms** for pricing and **Euler-Maruyama** for path projection.
* **Data**: `yfinance` API for live 2026 market data retrieval.
* **Frontend**: Streamlit & Plotly for interactive 3D Volatility Surface visualisation.
* **Optimisation**: L-BFGS-B routine for multi-parameter model fitting.

## Security Posture
> **Note**: This repository utilises automated dependency monitoring and Bandit-based security linting. All model parameters are isolated in a non-tracked environment to ensure the integrity of the 2026-2027 simulation data.

---
*Reference: Fadugba, S. E. & Okunlola, J. T. (2014). On the Combination of Merton and Heston Models in the Theory of Option Pricing. International Journal of Applied Science and Mathematics.*

![Status](https://img.shields.io/badge/status-complete-success) ![License](https://img.shields.io/badge/license-MIT-green)