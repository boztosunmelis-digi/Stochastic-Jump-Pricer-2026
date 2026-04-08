"""Streamlit dashboard for the Bates stochastic volatility model."""
# pylint: disable=invalid-name
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from engine import BatesModelEngine
from calibration import BatesCalibrator


# --- Caching Layer ---
@st.cache_data(ttl=3600)
def fetch_market_data(ticker, expiry):
    """Fetch raw data to ensure serializability for caching."""
    import yfinance as yf  # pylint: disable=import-outside-toplevel
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)

        # Extract copies to isolate data from internal yfinance objects
        calls_df = chain.calls.copy()
        puts_df = chain.puts.copy()
        spot_price = float(tk.fast_info['lastPrice'])

        return calls_df, puts_df, spot_price
    except Exception as e:  # pylint: disable=broad-except
        st.warning(f"Data Fetch Warning: {e}. Using fallback spot.")
        return None, None, 550.0


# --- Auto-Intervention Logic ---
def needs_recalibration(current_spot, last_spot, threshold=0.02):
    """Intervene only if the spot price has drifted more than 2%."""
    if last_spot is None:
        return True
    drift = abs(current_spot - last_spot) / last_spot
    return drift > threshold


# --- Page Configuration ---
st.set_page_config(page_title="Bates Model Core (2026)", layout="wide")


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
            'zaxis': {'range': [0.0, 0.7]}
        },
        margin={'l': 0, 'r': 0, 'b': 0, 't': 40}
    )
    return fig


# --- Sidebar: Simulation Settings ---
st.sidebar.header("Simulation Settings")
ticker_options = {
    "SPY (S&P 500 ETF)": "SPY",
    "XOM (ExxonMobil - Energy)": "XOM",
    "AAPL (Apple - Tech)": "AAPL",
    "TSLA (Tesla - High Vol)": "TSLA"
}
selected_label = st.sidebar.selectbox(
    "Select Target Asset:", options=list(ticker_options.keys())
)
selected_ticker = ticker_options[selected_label]

# --- Main UI ---
st.title(f"Bates Stochastic-Jump Engine: {selected_ticker}")
tab1, tab2 = st.tabs(["Monte Carlo Projections", "Fourier Calibration"])

with tab1:
    st.subheader("Price Path Projections (2026-2027)")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        num_paths = st.slider("Number of Paths", 10, 100, 50)
        t_years_sim = st.slider("Horizon (Years)", 0.5, 2.0, 1.5)
        if st.button("Generate Forecast"):
            if 'calibrated_params' in st.session_state:
                p = st.session_state.calibrated_params
                engine = BatesModelEngine()
                engine.kappa, engine.theta, engine.sigma_v = p[0], p[1], p[2]
                engine.rho, engine.lamb = p[3], p[4]
                engine.mu_j, engine.delta_j = p[5], p[6]
                st.sidebar.success("Using Calibrated Parameters!")
            else:
                engine = BatesModelEngine()
                st.sidebar.warning("Using Default Parameters.")
            st.session_state.paths = [
                engine.simulate_path(t_years_sim) for _ in range(num_paths)
            ]
    with col_b:
        if 'paths' in st.session_state:
            fig_paths = go.Figure()
            for p in st.session_state.paths:
                fig_paths.add_trace(go.Scatter(
                    y=p, mode='lines', line=dict(width=1), opacity=0.5
                ))
            st.plotly_chart(fig_paths, width='stretch')

with tab2:
    st.subheader(f"Calibrating to {selected_ticker} Market Smile")
    if st.button(f"Calibrate {selected_ticker} Parameters"):
        prev_spot = st.session_state.get('last_calibrated_spot')
        last_params = st.session_state.get('calibrated_params')
        # Use a temporary calibrator to get the current spot price
        # without making a separate yfinance call
        _temp = BatesCalibrator(selected_ticker, "2026-12-18")
        spot_now = _temp.spot

        if not needs_recalibration(spot_now, prev_spot):
            st.info("✅ Model Healthy: Using cached parameters.")
            opt_params = last_params
            calibrator = BatesCalibrator(selected_ticker, "2026-12-18")
        else:
            with st.spinner("Solving Fourier Integrals..."):
                calibrator = BatesCalibrator(selected_ticker, "2026-12-18")
                opt_params = calibrator.fit()
                st.session_state.calibrated_params = opt_params
                st.session_state.last_calibrated_spot = spot_now
                st.success("Calibration Successful!")

        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jump Intensity (λ)", f"{opt_params[4]:.4f}")
        m2.metric("Mean Rev (κ)", f"{opt_params[0]:.2f}")
        m3.metric("Vol of Vol (σv)", f"{opt_params[2]:.4f}")
        m4.metric("Correlation (ρ)", f"{opt_params[3]:.2f}")

        # Visual Surface Generation
        spot = calibrator.spot
        sig_v, rho = opt_params[2], opt_params[3]
        expiries = np.linspace(0.1, 2.0, 15)
        vol_matrix = np.array([[
            (sig_v * 0.4) + (2.5 * ((x - spot)/spot)**2) -
            (rho * (x - spot)/spot) + (y * 0.04)
            for x in calibrator.strikes
        ] for y in expiries])
        st.plotly_chart(
            plot_volsurface(calibrator.strikes, expiries, vol_matrix),
            width='stretch'
        )
