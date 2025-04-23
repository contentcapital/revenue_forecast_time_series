# Time Series Forecasting with PyMC3

This project demonstrates time series forecasting using PyMC3, a probabilistic programming framework for Bayesian statistical modeling.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `time_series_forecast.py`: Main script for time series forecasting
- `royalty_revenue_data_with_royalty.csv`: Input data file
- `requirements.txt`: Project dependencies

## Usage

1. Make sure your data file is in the correct format (CSV with a date column and a target variable column)
2. Run the forecasting script:
```bash
python time_series_forecast.py
```

## Output

The script will generate several files:
- `original_series.png`: Plot of the original time series
- `posterior_distributions.png`: Posterior distributions of model parameters
- `forecast.png`: Forecast plot with credible intervals
- `forecast_results.csv`: Forecast results in CSV format

## Model Description

The model implements a simple linear trend model with:
- Normal priors for the intercept (alpha) and slope (beta)
- Half-Normal prior for the noise term (sigma)
- Normal likelihood for the observations

The forecast includes:
- Point estimates (mean)
- 95% credible intervals
- 12-period forecast horizon

## Notes

- Adjust the `target_column` variable in the script to match your data's column name
- The model can be modified to include more complex components like seasonality or autoregressive terms
- The forecast horizon (12 periods) can be adjusted by modifying the `future_t` array 