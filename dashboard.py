"""Streamlit dashboard for the Bates stochastic volatility
jump-diffusion model."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from engine import BatesModelEngine

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
        title='Bates Model: 3D Implied Volatility Surface (2026-2027)',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Expiry (Years)',
            zaxis_title='Implied Volatility'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark"
    )
    return fig


# --- MAIN DASHBOARD UI ---
st.title("Bates Model: Advanced Financial Simulation")
st.markdown(
    "This simulation combines **Heston Stochastic Volatility** "
    "and **Merton Jumps**."
)

# Sidebar Settings
st.sidebar.header("Simulation Settings")
n_paths = st.sidebar.slider("Number of Paths", 1, 50, 10)
T_YEARS = 1.75  # Q1 2026 to Q4 2027

# Tab System for a Clean Look
tab1, tab2 = st.tabs(["Price Projections", "Implied Vol Surface"])

with tab1:
    if st.button("Generate Q1 2026 - Q4 2027 Simulation"):
        engine = BatesModelEngine()
        fig_paths = go.Figure()
        time_line = np.linspace(0, T_YEARS, 505)

        for i in range(n_paths):
            path = engine.simulate_path(T=T_YEARS, steps=504)
            fig_paths.add_trace(go.Scatter(
                x=time_line, y=path, mode='lines', name=f'Path {i+1}'
            ))

        fig_paths.update_layout(
            title=f"Bates Model Price Projections (S0={engine.S0})",
            xaxis_title="Time (Years from 2026)",
            yaxis_title="Asset Price",
            template="plotly_dark"
        )
        st.plotly_chart(fig_paths, use_container_width=True)

with tab2:
    st.subheader("3D Volatility Analysis")
    # Placeholder Data for Demonstration
    # (In a real run, this would be populated by your calibration.py results)
    strikes = np.linspace(80, 120, 20)
    expiries = np.linspace(0.1, 2.0, 20)
    # Simple mathematical curve to visualize the "Smirk"
    vol_matrix = np.array([
        [0.2 + (100 - x)**2 / 4000 + y / 10 for x in strikes]
        for y in expiries
    ])

    fig_surface = plot_volsurface(strikes, expiries, vol_matrix)
    st.plotly_chart(fig_surface, use_container_width=True)

st.info(
    "Mathematical Reference: "
    "'On the Combination of Merton and Heston Models'."
)
