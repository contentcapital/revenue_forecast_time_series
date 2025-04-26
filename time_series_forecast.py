import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import arviz as az
from datetime import datetime


np.random.seed(42)


df = pd.read_csv('royalty_revenue_data_with_royalty.csv')


if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)


target_column = 'revenue'  
y = df[target_column].values


t = np.arange(len(y))


plt.figure(figsize=(12, 6))
plt.plot(t, y)
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Revenue')
plt.savefig('original_series.png')
plt.close()


with pm.Model() as model:
    
    sigma = pm.HalfNormal('sigma', sigma=1)
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1)
    
    trend = alpha + beta * t
    

    likelihood = pm.Normal('y', mu=trend, sigma=sigma, observed=y)
    
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)


az.plot_trace(trace)
plt.savefig('posterior_distributions.png')
plt.close()


with model:
    
    future_t = np.arange(len(y), len(y) + 12)  
    
    
    ppc = pm.sample_posterior_predictive(trace, var_names=['y'])
    
    
    forecast_mean = ppc['y'].mean(axis=0)
    forecast_lower = np.percentile(ppc['y'], 2.5, axis=0)
    forecast_upper = np.percentile(ppc['y'], 97.5, axis=0)


plt.figure(figsize=(12, 6))
plt.plot(t, y, label='Observed')
plt.plot(future_t, forecast_mean, 'r', label='Forecast')
plt.fill_between(future_t, forecast_lower, forecast_upper, alpha=0.2, color='r')
plt.title('Time Series Forecast')
plt.xlabel('Time')
plt.ylabel('Revenue')
plt.legend()
plt.savefig('forecast.png')
plt.close()


print("\nModel Summary:")
print(az.summary(trace))


forecast_df = pd.DataFrame({
    'time': future_t,
    'mean': forecast_mean,
    'lower': forecast_lower,
    'upper': forecast_upper
})
forecast_df.to_csv('forecast_results.csv', index=False) 