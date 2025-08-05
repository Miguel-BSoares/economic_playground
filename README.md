### Fintech repository ###
Tool development to simulate financial scenarios

# 1. Bayesian Dividend Simulator

A Streamlit web app that models dividend income and reinvestment using Bayesian priors and high-probability density intervals (HPDI).

## 🚀 Features

- Define priors for:
  - Dividend yield
  - Dividend growth
  - Price growth
- Supports **normal** and **lognormal** distributions
- Handles:
  - Monthly reinvestment
  - Dividend tax withholding
  - Inflation adjustment
- Outputs:
  - Real dividend income (monthly or yearly)
  - Cumulative dividend growth
  - 95% HPDI shaded confidence intervals
- CSV download of results

## 📦 Installation

```bash
pip install streamlit matplotlib numpy
