###  Introduction ###

I have an interest in economy and stock market because I find it interesting how markets evolve also as a response to investor´s sentiment. I have no education in economy but still interested in the topic so in this repo I will pubish some tools I find it usefull myself.

# 1. Bayesian Dividend Simulator (tool)

A Streamlit web app that models dividend income and reinvestment using Bayesian priors and high-probability density intervals (HPDI).

The idea is to be able to model variance in earnings as a function of different scenarios. Bayesian statistics allow to devise expected distributions, but in general this is based on quite simplistic expectations. 

More importantly, it accounts for compound interest rates (and earnings) over monthly or quarterly periods. 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bayedivsv1.streamlit.app)

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



