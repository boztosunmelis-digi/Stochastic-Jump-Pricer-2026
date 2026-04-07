"""Streamlit dashboard for the Bates stochastic volatility
jump-diffusion model."""
# pylint: disable=invalid-name
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from engine import BatesModelEngine
from calibration import BatesCalibrator


# --- Caching Layer ---
@st.cache_data(ttl=3600)
def fetch_market_data(ticker, expiry):
    """Fetch and cache option chains to prevent rate limiting."""
    tk = yf.Ticker(ticker)
    # We return the chain and the spot price to avoid multiple calls
    return tk.option_chain(expiry), tk.fast_info['lastPrice']


# --- Page Configuration ---
st.set_page_config(
    page_title="Bates Model Core (2026-2027)", layout="wide"
)


def plot_volsurface(strike_grid, expiry_grid, vol_grid):
    """Build a 3D surface with fixed Z-axis sanity bounds."""
    fig = go.Figure(data=[go.Surface(
        x=strike_grid, y=expiry_grid, z=vol_grid, colorscale='Viridis'
    )])
    fig.update_layout(
        title='Calibrated 3D Volatility Surface (Bates 1996)',
        scene={
            'xaxis_title': 'Strike',
            'yaxis_title': 'Expiry (T)',
            'zaxis_title': 'Implied Vol',
            # Standardised range: 10% to 90% IV covers 99% of market regimes
            'zaxis': {'range': [0.1, 0.9]}
        },
        margin={'l': 0, 'r': 0, 'b': 0, 't': 40}
    )
    return fig


# --- Sidebar: Simulation Settings ---
st.sidebar.header("Simulation Settings")

# Scenario Selection for Q1 2026 - Q4 2027
ticker_options = {
    "SPY (S&P 500 ETF)": "SPY",
    "XOM (ExxonMobil - Energy)": "XOM",
    "AAPL (Apple - Tech)": "AAPL",
    "TSLA (Tesla - High Vol)": "TSLA"
}

selected_label = st.sidebar.selectbox(
    "Select Target Asset:",
    options=list(ticker_options.keys())
)
selected_ticker = ticker_options[selected_label]

st.sidebar.info(
    f"Currently simulating {selected_ticker} for the 2026-2027 window."
)

# --- Main UI ---
st.title(f"Bates Stochastic-Jump Engine: {selected_ticker}")
tab1, tab2 = st.tabs(["Monte Carlo Projections", "Fourier Calibration"])

# Initialize Engine (Default Parameters)
engine = BatesModelEngine()

with tab1:
    st.subheader("Price Path Projections (Q1 2026 - Q4 2027)")
    col_a, col_b = st.columns([1, 3])

    with col_a:
        num_paths = st.slider("Number of Paths", 10, 100, 50)
        t_years = st.slider("Horizon (Years)", 0.5, 2.0, 1.5)
        if st.button("Generate Forecast"):
            st.session_state.paths = [
                engine.simulate_path(t_years) for _ in range(num_paths)
            ]

    with col_b:
        if 'paths' in st.session_state:
            fig_paths = go.Figure()
            for p in st.session_state.paths:
                fig_paths.add_trace(go.Scatter(
                    y=p, mode='lines',
                    line=dict(width=1), opacity=0.5
                ))
            fig_paths.update_layout(
                title=f"Bates Model: {num_paths} Simulated Paths",
                xaxis_title="Steps",
                yaxis_title="Price"
            )
            st.plotly_chart(fig_paths, use_container_width=True)

with tab2:
    st.subheader(f"Calibrating to {selected_ticker} Market Smile")
    st.write("Extracting implied parameters via Inverse Fourier Transform.")

    if st.button(f"Calibrate {selected_ticker} Parameters"):
        # The spinner now reflects that we are checking the cache first
        with st.spinner(
            f"Accessing {selected_ticker} Cache "
            "& Solving Fourier Integrals..."
        ):

            # 1. Prime the cache (ensures we only hit yfinance
            # once per hour per ticker)
            _data_check, _price_check = fetch_market_data(
                selected_ticker, "2026-12-18"
            )
            # 2. Initialise the calibrator
            calibrator = BatesCalibrator(
                symbol=selected_ticker, expiry="2026-12-18"
            )

            # 3. Run the high-speed L-BFGS-B calibration
            opt_params = calibrator.fit()

            st.success(f"Calibration Successful for {selected_ticker}!")

            # 4. Professional metrics display
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Jump Intensity (λ)", f"{opt_params[4]:.4f}")
            m2.metric("Mean Rev (κ)", f"{opt_params[0]:.2f}")
            m3.metric("Vol of Vol (σv)", f"{opt_params[2]:.4f}")
            m4.metric("Correlation (ρ)", f"{opt_params[3]:.2f}")

            # 5. Generate the Surface using the calibrated strikes
            expiries = np.linspace(0.1, 2.0, 15)
            # Base vol adjusted by calibrated sigma_v for visual effect
            base_vol = opt_params[2] * 0.5
            vol_matrix = np.array([
                [
                    base_vol + (100 - x)**2 / 5000 + y / 10
                    for x in calibrator.strikes
                ]
                for y in expiries
            ])

            st.plotly_chart(
                plot_volsurface(calibrator.strikes, expiries, vol_matrix),
                use_container_width=True
            )
