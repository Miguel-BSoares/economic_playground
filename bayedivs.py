import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
import csv

# Function to draw samples from prior distribution
def draw_samples(distribution, mean, std, n_samples):
    if distribution == "lognormal":
        sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
        mu = np.log(mean) - 0.5 * sigma ** 2
        return lognorm(s=sigma, scale=np.exp(mu)).rvs(n_samples)
    else:  # normal
        return norm(loc=mean, scale=std).rvs(n_samples)

# Function to compute HPDI
def compute_hpdi(data, cred_mass=0.95):
    sorted_data = np.sort(data)
    ci_idx = int(np.floor(cred_mass * len(sorted_data)))
    n_cis = len(sorted_data) - ci_idx
    ci_width = sorted_data[ci_idx:] - sorted_data[:n_cis]
    min_idx = np.argmin(ci_width)
    hdi_min = sorted_data[min_idx]
    hdi_max = sorted_data[min_idx + ci_idx]
    return hdi_min, hdi_max

# Function to summarize simulations
def summarize_scenarios(sim_array, cred_mass=0.95):
    mean_vals = np.mean(sim_array, axis=0)
    hpdi_lows = []
    hpdi_highs = []
    for i in range(sim_array.shape[1]):
        low, high = compute_hpdi(sim_array[:, i], cred_mass)
        hpdi_lows.append(low)
        hpdi_highs.append(high)
    return mean_vals, np.array(hpdi_lows), np.array(hpdi_highs)

# Function to simulate dividend reinvestment
def simulate_dividends(
    years, n_scenarios, priors, inflation, reinvestment_freq, share_sets
):
    periods_per_year = 12 if reinvestment_freq == "Monthly" else 4
    total_periods = years * periods_per_year
    inflation_rate = inflation / 100 / periods_per_year

    sim_results = np.zeros((n_scenarios, total_periods))

    for scenario in range(n_scenarios):
        total_shares = 0
        scenario_dividends = np.zeros(total_periods)

        for share_set in share_sets:
            shares = share_set["shares"]
            price = share_set["price"]

            yield_samples = draw_samples(priors["yield_dist"], priors["yield_mean"], priors["yield_std"], total_periods)
            price_growth_samples = draw_samples(priors["price_growth_dist"], priors["price_growth_mean"], priors["price_growth_std"], total_periods)
            div_growth_samples = draw_samples(priors["div_growth_dist"], priors["div_growth_mean"], priors["div_growth_std"], total_periods)

            shares_owned = shares
            current_price = price
            current_dividend = (priors["yield_mean"] / 100) * current_price / periods_per_year

            for t in range(total_periods):
                gross_dividend = shares_owned * current_dividend
                net_dividend = gross_dividend * (1 - priors["tax"] / 100)
                real_dividend = net_dividend / ((1 + inflation_rate) ** t)
                scenario_dividends[t] += real_dividend

                shares_purchased = net_dividend / current_price
                shares_owned += shares_purchased

                current_dividend *= (1 + div_growth_samples[t] / 100 / periods_per_year)
                current_price *= (1 + price_growth_samples[t] / 100 / periods_per_year)

        sim_results[scenario, :] = scenario_dividends

    return sim_results

# Streamlit UI
st.title("Bayesian Dividend Simulator")

# Share sets
st.header("📊 Share Inputs (up to 10 sets)")
num_sets = st.slider("Number of share sets", 1, 10, value=1)
share_sets = []
for i in range(num_sets):
    with st.expander(f"Share Set {i + 1}"):
        shares = st.number_input(f"Initial number of shares (Set {i + 1})", min_value=1, value=100)
        price = st.number_input(f"Initial share price (Set {i + 1})", min_value=0.01, value=100.0)
        share_sets.append({"shares": shares, "price": price})

st.header("🎯 Simulation Settings")
years = st.slider("Years to simulate", 1, 50, value=20)
n_scenarios = st.number_input("Number of scenarios", min_value=100, value=1000)
reinvestment_freq = st.radio("Reinvestment frequency", ["Monthly", "Quarterly"])
output_freq = st.radio("Output frequency", ["Monthly", "Yearly"])
inflation = st.number_input("Inflation rate (annual %, integer)", value=2, step=1)

st.header("📈 Priors")
prior_type = st.radio("Distribution type", ["normal", "lognormal"])

priors = {
    "yield_mean": st.number_input("Dividend yield mean (%)", value=4.0),
    "yield_std": st.number_input("Dividend yield std dev (%)", value=1.0),
    "yield_dist": prior_type,
    "div_growth_mean": st.number_input("Dividend growth mean (%)", value=5.0),
    "div_growth_std": st.number_input("Dividend growth std dev (%)", value=1.5),
    "div_growth_dist": prior_type,
    "price_growth_mean": st.number_input("Price growth mean (%)", value=6.0),
    "price_growth_std": st.number_input("Price growth std dev (%)", value=2.0),
    "price_growth_dist": prior_type,
    "tax": st.slider("Dividend tax withholding (%)", 0, 50, value=15),
}

if st.button("Run Simulation"):
    sim_results = simulate_dividends(years, n_scenarios, priors, inflation, reinvestment_freq, share_sets)
    cred_mass = 0.95

    # Adjust output resolution
    if output_freq == "Yearly":
        periods_per_year = 12 if reinvestment_freq == "Monthly" else 4
        sim_results = sim_results.reshape(n_scenarios, years, periods_per_year).sum(axis=2)
        x = np.arange(1, years + 1)
        unit = "Year"
    else:
        x = np.arange(1, sim_results.shape[1] + 1)
        unit = "Month"

    mean_vals, hpdi_low, hpdi_high = summarize_scenarios(sim_results, cred_mass)

    # Cumulative totals
    cumulative_results = np.cumsum(sim_results, axis=1)
    cum_mean, cum_hpdi_low, cum_hpdi_high = summarize_scenarios(cumulative_results, cred_mass)

    # Plot regular dividends
    plt.figure()
    plt.fill_between(x, hpdi_low, hpdi_high, color='lightblue', alpha=0.3, label=f"{int(cred_mass * 100)}% HPDI")
    plt.plot(x, mean_vals, label="Mean", color='blue')
    plt.xlabel(unit)
    plt.ylabel("Dividends ($, real)")
    plt.title("Real Dividend Income")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Plot cumulative dividends
    plt.figure()
    plt.fill_between(x, cum_hpdi_low, cum_hpdi_high, color='lightgreen', alpha=0.3, label=f"{int(cred_mass * 100)}% HPDI")
    plt.plot(x, cum_mean, color='green', label="Cumulative Mean", linewidth=2)
    plt.title(f"Cumulative Real Dividends with {int(cred_mass * 100)}% HPDI")
    plt.xlabel(unit)
    plt.ylabel("Cumulative Dividends ($, today's value)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

    # Export to CSV
    import pandas as pd
    df = pd.DataFrame({
        unit: x,
        "Mean": mean_vals,
        "HPDI Low": hpdi_low,
        "HPDI High": hpdi_high,
        "Cumulative Mean": cum_mean,
        "Cumulative HPDI Low": cum_hpdi_low,
        "Cumulative HPDI High": cum_hpdi_high
    })
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "dividend_simulation.csv", "text/csv")
