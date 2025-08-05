import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import csv

st.set_page_config(page_title="Bayesian Dividend Simulator", layout="centered")
st.title("📈 Bayesian Dividend Simulator")

st.markdown("""
This app simulates future **dividend income** using Bayesian modeling and user-defined priors. All dividends are **reinvested monthly**.
""")

# --- Input Controls ---
with st.sidebar:
    st.header("Simulation Settings")
    initial_shares = st.number_input("Initial number of shares", 1, 100000, value=100)
    initial_price = st.number_input("Initial share price ($)", 0.1, 1000.0, value=10.0)
    years = st.slider("Years to simulate", 1, 50, value=10)
    n_scenarios = st.number_input("Number of scenarios", 100, 10000, value=1000, step=100)
    tax_rate = st.slider("Dividend Tax Rate (%)", 0.0, 50.0, value=15.0)
    inflation_rate = st.slider("Inflation Rate (%)", 0.0, 10.0, value=2.5)
    cred_mass = st.slider("Credibility Interval", 0.80, 0.99, value=0.95)
    output_mode = st.radio("Output frequency", ["yearly", "monthly"], index=0)

    st.header("Distribution Priors")
    dist_options = ["normal", "lognormal"]

    def dist_inputs(label, default_mean, default_std):
        dist = st.selectbox(f"{label} distribution", dist_options, key=label)
        mean = st.number_input(f"{label} mean", value=default_mean, key=f"{label}_mean")
        std = st.number_input(f"{label} stddev", value=default_std, key=f"{label}_std")
        return dist, mean, std

    yield_dist, yield_mean, yield_std = dist_inputs("Dividend Yield (%)", 6.0, 1.0)
    div_growth_dist, div_growth_mean, div_growth_std = dist_inputs("Dividend Growth (%)", 2.0, 0.5)
    price_growth_dist, price_growth_mean, price_growth_std = dist_inputs("Price Growth (%)", 3.0, 1.0)

# --- Distribution Functions ---
def get_dist(dist, mean, std):
    if dist == "lognormal":
        mu = np.log(mean) - 0.5 * std**2
        return lambda: np.random.lognormal(mean=mu, sigma=std)
    else:
        return lambda: np.random.normal(loc=mean, scale=std)

def compute_hpdi(samples, cred_mass=0.95):
    sorted_samples = np.sort(samples)
    ci_idx = int(np.floor(cred_mass * len(sorted_samples)))
    n_cis = len(sorted_samples) - ci_idx
    interval_width = sorted_samples[ci_idx:] - sorted_samples[:n_cis]
    min_idx = np.argmin(interval_width)
    return sorted_samples[min_idx], sorted_samples[min_idx + ci_idx]

def simulate_scenarios(n_scenarios, years, initial_shares, initial_price,
                        yield_sampler, div_growth_sampler, price_growth_sampler,
                        inflation_rate, tax_rate, output_mode):
    months = years * 12
    steps = months if output_mode == "monthly" else years
    result = np.zeros((n_scenarios, steps))

    for sim in range(n_scenarios):
        shares = initial_shares
        price = initial_price
        annual_yield = yield_sampler()
        annual_div_growth = div_growth_sampler()
        annual_price_growth = price_growth_sampler()

        m_yield = annual_yield / 12 / 100
        m_div_growth = (1 + annual_div_growth / 100) ** (1 / 12) - 1
        m_price_growth = (1 + annual_price_growth / 100) ** (1 / 12) - 1

        total = 0
        for month in range(1, months + 1):
            div_per_share = price * m_yield
            gross_div = shares * div_per_share
            net_div = gross_div * (1 - tax_rate / 100)
            shares += net_div / price
            real_div = net_div / ((1 + inflation_rate / 100) ** (month / 12))

            if output_mode == "monthly":
                result[sim, month - 1] = real_div
            else:
                total += real_div
                if month % 12 == 0:
                    result[sim, month // 12 - 1] = total
                    total = 0
    return result

def summarize(data, cred_mass):
    mean = np.mean(data, axis=0)
    low, high = [], []
    for i in range(data.shape[1]):
        l, h = compute_hpdi(data[:, i], cred_mass)
        low.append(l)
        high.append(h)
    return mean, low, high

def plot_with_hpdi(x, mean, low, high, label, color):
    plt.fill_between(x, low, high, color=color, alpha=0.3, label=f"{label} {int(cred_mass*100)}% HPDI")
    plt.plot(x, mean, label=f"{label} Mean", color=color)

# --- Run Simulation ---
if st.button("Run Simulation"):
    st.info("Running simulations...")
    y_sampler = get_dist(yield_dist, yield_mean, yield_std)
    dg_sampler = get_dist(div_growth_dist, div_growth_mean, div_growth_std)
    pg_sampler = get_dist(price_growth_dist, price_growth_mean, price_growth_std)

    sim = simulate_scenarios(
        n_scenarios, years, initial_shares, initial_price,
        y_sampler, dg_sampler, pg_sampler,
        inflation_rate, tax_rate, output_mode
    )

    x = np.arange(1, sim.shape[1] + 1)
    unit = "Month" if output_mode == "monthly" else "Year"
    mean, low, high = summarize(sim, cred_mass)
    cum = np.cumsum(sim, axis=1)
    cum_mean, cum_low, cum_high = summarize(cum, cred_mass)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_with_hpdi(x, mean, low, high, f"{unit}ly Dividend", "blue")
    plt.title(f"{unit}-wise Real Dividend Income")
    plt.xlabel(unit)
    plt.ylabel("Dividends ($, real)")
    plt.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    plot_with_hpdi(x, cum_mean, cum_low, cum_high, f"Cumulative {unit}s", "green")
    plt.title(f"Cumulative Real Dividends")
    plt.xlabel(unit)
    plt.ylabel("Cumulative Dividends ($, real)")
    plt.legend()
    st.pyplot(fig2)

    # Export CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([unit, "Mean", "HPDI Low", "HPDI High"])
    for i in range(len(mean)):
        writer.writerow([i + 1, mean[i], low[i], high[i]])
    b64 = base64.b64encode(output.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dividends_{output_mode}.csv">📥 Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
