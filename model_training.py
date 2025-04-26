import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
from prophet import Prophet
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import joblib

class TimeSeriesModels:
    def __init__(self, data, target_column, train_size=0.8):
        self.data = data
        self.target_column = target_column
        self.train_size = train_size
        self.train_data, self.test_data = self._split_data()
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def _split_data(self):
        split_idx = int(len(self.data) * self.train_size)
        train = self.data[:split_idx]
        test = self.data[split_idx:]
        return train, test
    
    def train_pymc3_model(self):
        with pm.Model() as model:
            sigma = pm.HalfNormal('sigma', sigma=1)
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=1)
            gamma = pm.Normal('gamma', mu=0, sigma=1)
            
            t = np.arange(len(self.train_data))
            
            trend = alpha + beta * t
            seasonality = gamma * np.sin(2 * np.pi * t / 12)
            
            likelihood = pm.Normal('y', 
                                 mu=trend + seasonality, 
                                 sigma=sigma, 
                                 observed=self.train_data[self.target_column])
            
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)
        
        self.models['pymc3'] = {'model': model, 'trace': trace}
        return model, trace
    
    def train_prophet_model(self):
        prophet_data = self.train_data.reset_index()
        prophet_data.columns = ['ds', 'y']
        
        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_data)
        
        self.models['prophet'] = model
        return model
    
    def train_arima_model(self):
        model = auto_arima(self.train_data[self.target_column],
                          seasonal=True,
                          m=12,
                          suppress_warnings=True)
        
        self.models['arima'] = model
        return model
    
    def make_predictions(self):
        with self.models['pymc3']['model']:
            future_t = np.arange(len(self.train_data), 
                               len(self.train_data) + len(self.test_data))
            ppc = pm.sample_posterior_predictive(self.models['pymc3']['trace'],
                                               var_names=['y'])
            self.predictions['pymc3'] = ppc['y'].mean(axis=0)
        
        future = pd.DataFrame({
            'ds': pd.date_range(start=self.train_data.index[-1],
                              periods=len(self.test_data) + 1,
                              freq='M')[1:]
        })
        prophet_forecast = self.models['prophet'].predict(future)
        self.predictions['prophet'] = prophet_forecast['yhat'].values
        
        arima_forecast = self.models['arima'].predict(n_periods=len(self.test_data))
        self.predictions['arima'] = arima_forecast
    
    def calculate_metrics(self):
        for model_name, preds in self.predictions.items():
            self.metrics[model_name] = {
                'MSE': mean_squared_error(self.test_data[self.target_column], preds),
                'MAE': mean_absolute_error(self.test_data[self.target_column], preds),
                'RMSE': np.sqrt(mean_squared_error(self.test_data[self.target_column], preds))
            }
    
    def plot_predictions(self):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data[self.target_column],
            name='Actual',
            line=dict(color='black')
        ))
        
        colors = ['blue', 'red', 'green']
        for (model_name, preds), color in zip(self.predictions.items(), colors):
            fig.add_trace(go.Scatter(
                x=self.test_data.index,
                y=preds,
                name=f'{model_name} Forecast',
                line=dict(color=color, dash='dash')
            ))
        
        fig.update_layout(
            title='Time Series Forecast Comparison',
            xaxis_title='Date',
            yaxis_title=self.target_column,
            height=600
        )
        
        fig.write_html("forecast_comparison.html")
    
    def save_results(self):
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv('model_metrics.csv')
        
        predictions_df = pd.DataFrame(self.predictions, index=self.test_data.index)
        predictions_df['actual'] = self.test_data[self.target_column]
        predictions_df.to_csv('model_predictions.csv')
        
        for model_name, model_data in self.models.items():
            if model_name == 'pymc3':
                az.to_netcdf(model_data['trace'], f'{model_name}_trace.nc')
            else:
                joblib.dump(model_data, f'{model_name}_model.joblib')

def main():
    data = pd.read_csv('preprocessed_data.csv', index_col=0, parse_dates=True)
    
    ts_models = TimeSeriesModels(data, target_column='revenue')
    
    print("Training PyMC3 model...")
    ts_models.train_pymc3_model()
    
    print("Training Prophet model...")
    ts_models.train_prophet_model()
    
    print("Training ARIMA model...")
    ts_models.train_arima_model()
    
    print("Generating predictions...")
    ts_models.make_predictions()
    
    print("Calculating metrics...")
    ts_models.calculate_metrics()
    
    print("Creating visualizations...")
    ts_models.plot_predictions()
    
    print("Saving results...")
    ts_models.save_results()

if __name__ == "__main__":
    main() 