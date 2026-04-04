import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from engine import BatesModelEngine

st.set_page_config(page_title="Bates-Core: 2026-2027 Simulation", layout="wide")

st.title(" Bates Model: Advanced Financial Simulation")
st.markdown("This simulation combines **Heston Stochastic Volatility** and **Merton Jumps**[cite: 10, 46].")

# Sidebar for Simulation Parameters (Security: No Hardcoding)
st.sidebar.header("Simulation Settings")
n_paths = st.sidebar.slider("Number of Paths", 1, 50, 10)
T_years = 1.75  # Q1 2026 to Q4 2027

if st.button("Generate Q1 2026 - Q4 2027 Simulation"):
    engine = BatesModelEngine()
    
    # Generate multiple paths
    fig = go.Figure()
    time_line = np.linspace(0, T_years, 505)
    
    for i in range(n_paths):
        path = engine.simulate_path(T=T_years, steps=504)
        fig.add_trace(go.Scatter(x=time_line, y=path, mode='lines', name=f'Path {i+1}'))
    
    fig.update_layout(
        title=f"Bates Model Price Projections (S0={engine.S0})",
        xaxis_title="Time (Years from 2026)",
        yaxis_title="Asset Price",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Reference Note for the Repo
    st.info("Mathematical Reference: 'On the Combination of Merton and Heston Models'[cite: 3, 4].")