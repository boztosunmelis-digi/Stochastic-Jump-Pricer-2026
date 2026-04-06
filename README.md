# Stochastic-Jump-Pricer-2026: Bates Model Core

A full-stack financial simulation implementing the **Bates (1996) Model**, which captures asset dynamics for the **Q1 2026 - Q4 2027** window by combining **Heston Stochastic Volatility** with **Merton's Jump-Diffusion** [6, 16].

## Overview
This project simulates asset price paths while capturing two critical features of financial returns:

1. **Mean-Reverting Volatility**: Using the Heston model to ensure volatility evolves in a stochastic, mean-reverting fashion [15, 67].
2. **Occasional Outliers**: Using Merton's jump model to account for sudden, discontinuous trajectories in asset prices caused by public information [15, 48, 49].

## Mathematical Architecture ($D + J$)
The Bates model evaluates European option prices by treating the characteristic function as a product of diffusion and jump components [18, 179]:

$$\phi_{X_{t}}^{Bates}(z) = \phi_{X_{t}}^{Heston}(z) \times \exp\left(t\lambda\left(\exp\left(-\frac{\delta^{2}z^{2}}{2} + i\mu z\right) - 1\right)\right)$$

### 1. Heston Diffusion Component ($D$)
Handles the "leverage effect" and time-varying volatility through a correlated Wiener process [70, 124]:

$$dv_{t} = \kappa(\theta - v_{t})dt + \sigma\sqrt{v_{t}}dW_{t}^{2}$$

* **$\kappa$**: Speed of mean reversion [94].
* **$\theta$**: Long-run average volatility [94].

### 2. Merton Jump Component ($J$)
Adds mass to the distribution tails to account for fatter tails than the standard Black-Scholes model [44, 95]:

$$dS_{t} = (r - \lambda k)S_{t}dt + \sqrt{v_{t}}S_{t}dW_{t}^{1} + dZ_{t}$$

* **$\lambda$**: Jump intensity [201].
* **$Z_t$**: Compound Poisson process with log-normal jump sizes [53].

## Technical Stack & Security
* **Backend**: Python (NumPy, SciPy) implementing the Euler-Maruyama discretization.
* **Frontend**: Streamlit & Plotly for interactive 3D Volatility Surface visualization.
* **Security**: Strict `.env` parameter isolation to prevent hardcoding of sensitive market data.
* **Infrastructure**: Automated GitHub Actions for continuous security linting.

---
*Reference: Fadugba, S. E. & Okunlola, J. T. (2014). On the Combination of Merton and Heston Models in the Theory of Option Pricing. International Journal of Applied Science and Mathematics.* [3, 5, 52]