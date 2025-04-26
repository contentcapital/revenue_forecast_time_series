import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import os
import json
from datetime import datetime, timedelta
import warnings
import pmdarima as pm
from prophet import Prophet
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    def __init__(self, data_path, target_column='Total Revenue', date_column='Date'):
       
        self.data_path = data_path
        self.target_column = target_column
        self.date_column = date_column
        self.results_path = 'results'
        self.setup_directories()
        
    def setup_directories(self):
        
        directories = [
            'results/plots/diagnostic',
            'results/plots/decomposition',
            'results/plots/forecasting',
            'results/plots/correlation',
            'results/data/statistics',
            'results/data/models',
            'results/data/forecasts',
            'results/data/feature_importance'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def load_data(self):
        """Load and prepare the time series data"""
        try:
            df = pd.read_csv(self.data_path)
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df.set_index(self.date_column, inplace=True)
            df.sort_index(inplace=True)
            self.data = df
            self.save_data_info()
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
    def save_data_info(self):
        
        stats_dict = self.data[self.target_column].describe().to_dict()
        stats_dict = {k: float(v) for k, v in stats_dict.items()}
        
        
        rolling_stats = {
            '7d_mean': float(self.data[self.target_column].rolling(7).mean().mean()),
            '30d_mean': float(self.data[self.target_column].rolling(30).mean().mean()),
            '7d_std': float(self.data[self.target_column].rolling(7).std().mean()),
            '30d_std': float(self.data[self.target_column].rolling(30).std().mean())
        }
        
        info = {
            'time_range': {
                'start': self.data.index.min().strftime('%Y-%m-%d'),
                'end': self.data.index.max().strftime('%Y-%m-%d')
            },
            'total_observations': int(len(self.data)),
            'frequency': self.detect_frequency(),
            'missing_values': int(self.data[self.target_column].isnull().sum()),
            'basic_stats': stats_dict,
            'rolling_stats': rolling_stats
        }
        
        with open(f'{self.results_path}/data/statistics/data_info.json', 'w') as f:
            json.dump(info, f, indent=4)
            
    def detect_frequency(self):
       
        freq = pd.infer_freq(self.data.index)
        return str(freq) if freq else 'Irregular'
        
    def analyze_correlations(self):
      
     
        revenue_cols = [col for col in self.data.columns if 'Revenue' in col]
        corr_matrix = self.data[revenue_cols].corr()
        
      
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Revenue Stream Correlations')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/plots/correlation/revenue_correlations.png')
        plt.close()
        
        
        corr_dict = corr_matrix.to_dict()
        with open(f'{self.results_path}/data/statistics/correlations.json', 'w') as f:
            json.dump(corr_dict, f, indent=4)
            
    def analyze_platform_contribution(self):
        
        revenue_cols = [col for col in self.data.columns if 'Revenue' in col and col != self.target_column]
        
        
        contrib = {}
        for col in revenue_cols:
            contrib[col] = {
                'mean_contribution': float(self.data[col].mean() / self.data[self.target_column].mean() * 100),
                'total_revenue': float(self.data[col].sum())
            }
        
        plt.figure(figsize=(10, 10))
        plt.pie([v['mean_contribution'] for v in contrib.values()], 
                labels=[k.replace(' Revenue', '') for k in contrib.keys()],
                autopct='%1.1f%%')
        plt.title('Platform Revenue Contribution')
        plt.savefig(f'{self.results_path}/plots/correlation/platform_contribution.png')
        plt.close()
        
      
        with open(f'{self.results_path}/data/statistics/platform_contribution.json', 'w') as f:
            json.dump(contrib, f, indent=4)
            
    def analyze_seasonality(self):
        
       
        decomposition = seasonal_decompose(self.data[self.target_column], period=7)  # Weekly seasonality
        
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/plots/decomposition/seasonal_decomposition.png')
        plt.close()
        
      
        daily_pattern = self.data.groupby(self.data.index.dayofweek)[self.target_column].mean()
        monthly_pattern = self.data.groupby(self.data.index.month)[self.target_column].mean()
        
       
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        daily_pattern.plot(kind='bar', ax=ax1)
        ax1.set_title('Average Revenue by Day of Week')
        ax1.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        monthly_pattern.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Revenue by Month')
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/plots/decomposition/temporal_patterns.png')
        plt.close()
        
      
        seasonal_stats = {
            'daily_pattern': daily_pattern.to_dict(),
            'monthly_pattern': monthly_pattern.to_dict(),
            'seasonal_strength': self.calculate_seasonal_strength(decomposition)
        }
        
        with open(f'{self.results_path}/data/statistics/seasonal_analysis.json', 'w') as f:
            json.dump(seasonal_stats, f, indent=4)
            
    def calculate_seasonal_strength(self, decomposition):
      
        total_var = np.var(decomposition.observed)
        seasonal_var = np.var(decomposition.seasonal)
        return seasonal_var / total_var if total_var != 0 else 0
        
    def optimize_arima_parameters(self, train_data):
       
        model = pm.auto_arima(train_data,
                            start_p=0, start_q=0,
                            max_p=5, max_q=5,
                            m=7, 
                            seasonal=True,
                            d=None, D=None,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
        return model
        
    def fit_prophet_model(self, train_data):
        
        df_prophet = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
       
        model = Prophet(yearly_seasonality=True,
                       weekly_seasonality=True,
                       daily_seasonality=False,
                       changepoint_prior_scale=0.05)
        model.fit(df_prophet)
        return model
        
    def fit_models(self):
     
        
        tscv = TimeSeriesSplit(n_splits=5)
        models_cv_scores = {
            'holt_winters': {'mae': [], 'rmse': [], 'r2': []},
            'arima': {'mae': [], 'rmse': [], 'r2': []},
            'prophet': {'mae': [], 'rmse': [], 'r2': []}
        }
        
        for train_idx, test_idx in tscv.split(self.data):
            train = self.data.iloc[train_idx]
            test = self.data.iloc[test_idx]
            
           
            hw_model = ExponentialSmoothing(
                train[self.target_column],
                seasonal_periods=7,
                trend='add',
                seasonal='add'
            ).fit()
            
           
            arima_model = self.optimize_arima_parameters(train[self.target_column])
            
         
            prophet_model = self.fit_prophet_model(train[self.target_column])
            
         
            hw_pred = hw_model.forecast(len(test))
            arima_pred = arima_model.predict(n_periods=len(test))
            
            prophet_future = pd.DataFrame({'ds': test.index})
            prophet_pred = prophet_model.predict(prophet_future)['yhat']
            
           
            for name, pred in [('holt_winters', hw_pred), 
                             ('arima', arima_pred),
                             ('prophet', prophet_pred)]:
                models_cv_scores[name]['mae'].append(
                    mean_absolute_error(test[self.target_column], pred))
                models_cv_scores[name]['rmse'].append(
                    np.sqrt(mean_squared_error(test[self.target_column], pred)))
                models_cv_scores[name]['r2'].append(
                    r2_score(test[self.target_column], pred))
        
       
        final_scores = {}
        for model_name, scores in models_cv_scores.items():
            final_scores[model_name] = {
                metric: float(np.mean(values))
                for metric, values in scores.items()
            }
        
      
        with open(f'{self.results_path}/data/models/model_performance.json', 'w') as f:
            json.dump(final_scores, f, indent=4)
            
      
        self.final_models = {
            'holt_winters': ExponentialSmoothing(
                self.data[self.target_column],
                seasonal_periods=7,
                trend='add',
                seasonal='add'
            ).fit(),
            'arima': self.optimize_arima_parameters(self.data[self.target_column]),
            'prophet': self.fit_prophet_model(self.data[self.target_column])
        }
        
        return final_scores
        
    def generate_forecasts(self, periods=90):
        """Generate and compare forecasts from different models"""
        future_dates = pd.date_range(
            start=self.data.index[-1] + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        forecasts = {}
        
       
        forecasts['holt_winters'] = self.final_models['holt_winters'].forecast(periods)
        forecasts['arima'] = pd.Series(
            self.final_models['arima'].predict(n_periods=periods),
            index=future_dates
        )
        
        prophet_future = pd.DataFrame({'ds': future_dates})
        forecasts['prophet'] = pd.Series(
            self.final_models['prophet'].predict(prophet_future)['yhat'].values,
            index=future_dates
        )
        
       
        plt.figure(figsize=(15, 8))
        plt.plot(self.data.index, self.data[self.target_column], 
                label='Historical', color='black', alpha=0.6)
        
        colors = ['blue', 'red', 'green']
        for (name, forecast), color in zip(forecasts.items(), colors):
            plt.plot(forecast.index, forecast, 
                    label=f'{name.replace("_", " ").title()} Forecast',
                    color=color, linestyle='--')
        
        plt.title('Revenue Forecasts Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.results_path}/plots/forecasting/forecast_comparison.png')
        plt.close()
        
     
        forecasts_dict = {
            name: {
                str(date): float(value)
                for date, value in forecast.items()
            }
            for name, forecast in forecasts.items()
        }
        
        with open(f'{self.results_path}/data/forecasts/future_forecasts.json', 'w') as f:
            json.dump(forecasts_dict, f, indent=4)
            
        return forecasts
        
    def run_complete_analysis(self):
       
        print("Starting comprehensive time series analysis...")
      
        print("Loading data...")
        self.load_data()
      
        print("Analyzing revenue stream correlations...")
        self.analyze_correlations()
    
        print("Analyzing platform contributions...")
        self.analyze_platform_contribution()
       
        print("Analyzing seasonality patterns...")
        self.analyze_seasonality()
        
     
        print("Fitting and evaluating models...")
        model_scores = self.fit_models()
        
      
        print("Generating 90-day forecasts...")
        forecasts = self.generate_forecasts(periods=90)
        
        print("Analysis complete! Results saved in the 'results' directory.")
        return model_scores, forecasts

if __name__ == "__main__":
   
    analyzer = TimeSeriesAnalyzer('royalty_revenue_data.csv')
  
    model_scores, forecasts = analyzer.run_complete_analysis() 