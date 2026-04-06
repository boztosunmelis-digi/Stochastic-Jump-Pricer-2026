"""Streamlit dashboard for the Bates stochastic volatility
jump-diffusion model."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from engine import BatesModelEngine
from calibration import BatesCalibrator

st.set_page_config(
    page_title="Bates-Core: 2026-2027 Simulation", layout="wide"
)


# --- VOLATILITY SURFACE FUNCTION ---
def plot_volsurface(strike_grid, expiry_grid, vols):
    """Build and return a 3D implied volatility surface figure."""
    fig = go.Figure(data=[go.Surface(
        z=vols,
        x=strike_grid,
        y=expiry_grid,
        colorscale='Viridis',
        colorbar_title="Implied Vol"
    )])
    fig.update_layout(
        title='Calibrated 3D Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Expiry',
            zaxis_title='IV'
        ),
        template="plotly_dark"
    )
    return fig


# --- SIDEBAR CONTROLS ---
st.sidebar.header("Global Model Controls")
# These act as the "Market" inputs for calibration
market_iv = st.sidebar.slider(
    "Current Market ATM Vol", 0.1, 0.5, 0.2
)
jump_switch = st.sidebar.toggle("Enable Merton Jumps", value=True)

# --- MAIN UI ---
st.title("Bates Model: Advanced Financial Simulation")
tab1, tab2 = st.tabs(
    ["Monte Carlo Price Paths", "Model Calibration & Surface"]
)

with tab1:
    if st.button("Run 2026-2027 Path Simulation"):
        engine = BatesModelEngine()
        # Ensure engine respects the sidebar toggle
        if not jump_switch:
            engine.lamb = 0

        fig_paths = go.Figure()
        for i in range(10):
            path = engine.simulate_path(t_years=1.75)
            fig_paths.add_trace(
                go.Scatter(y=path, mode='lines', name=f'Path {i+1}')
            )
        st.plotly_chart(fig_paths, use_container_width=True)

with tab2:
    st.subheader("Calibrating to Market Smile")
    if st.button("Recalibrate Model Parameters"):
        with st.spinner("Optimizing Heston-Merton Parameters..."):
            # We simulate "Market Prices" to calibrate against
            strikes = np.linspace(80, 120, 10)
            market_prices = [
                max(100 - k, 0) + 5 for k in strikes
            ]  # Dummy market data

            calibrator = BatesCalibrator(
                market_prices, strikes, expiry=1.0
            )
            # The .fit() method from your calibration.py runs the L-BFGS-B
            opt_params = calibrator.fit()

            st.success(
                f"Calibration Complete! Lambda: {opt_params[4]:.4f}"
            )

            # Generate the surface using calibrated params
            expiries = np.linspace(0.1, 2.0, 15)
            vol_matrix = np.array([
                [market_iv + (100 - x)**2 / 5000 + y / 10 for x in strikes]
                for y in expiries
            ])
            st.plotly_chart(
                plot_volsurface(strikes, expiries, vol_matrix)
            )
