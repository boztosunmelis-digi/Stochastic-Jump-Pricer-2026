# Bates-Core: Stochastic Volatility & Jump Simulation (2026-2027)

[cite_start]A full-stack financial simulation implementing the **Bates (1996) Model**, which combines the **Heston Stochastic Volatility** model with **Merton's Jump-Diffusion**[cite: 6, 16].

## Overview
This project simulates asset price paths for the timeframe **Q1 2026 - Q4 2027**. It captures two critical features of asset returns:
1. [cite_start]**Mean-Reverting Volatility**: Using the Heston model to ensure volatility stays bounded[cite: 15, 67].
2. [cite_start]**Occasional Outliers**: Using Merton's jump model to account for sudden, discontinuous price changes[cite: 15, 49].

## Technical Stack
- **Backend**: Python (NumPy, SciPy) for Monte Carlo engines.
- **Frontend**: Streamlit & Plotly for 3D Implied Volatility Surface visualization.
- **Security**: `.env` parameter isolation for zero-hardcode compliance.

## Key Equations
The characteristic function is the product of Heston and Merton components:
[cite_start]$$\varphi_{X_{t}}^{Bates}(z) = \varphi_{X_{t}}^{Heston}(z) \cdot \varphi_{X_{t}}^{Merton}(z)$$ [cite: 179]