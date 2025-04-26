import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pymc as pm
import arviz as az
import json
import os
from datetime import datetime, timedelta
import traceback

class TimeSeriesForecaster:
    def __init__(self, data_path='results/data/processed_data.csv'):
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.target = 'Total Revenue'
        self.forecast_horizon = 30
        self.results_dir = 'results/models'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def prepare_features(self):
        df = self.data.copy()
        
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df[self.target].shift(lag)
            
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df[self.target].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[self.target].rolling(window=window).std()
            
        df = df.dropna()
        
        X = df.drop(columns=[self.target])
        y = df[self.target]
        
        return X, y
        
    def train_xgboost(self, X, y):
        try:
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)
            
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 1000
            }
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                early_stopping_rounds=10,
                verbose_eval=100
            )
            
            y_pred = model.predict(dtest)
            
            metrics = {
                'MAE': float(mean_absolute_error(y_test, y_pred)),
                'RMSE': float(mean_squared_error(y_test, y_pred, squared=False)),
                'R2': float(r2_score(y_test, y_pred))
            }
            
            model.save_model(f'{self.results_dir}/xgboost_model.json')
            with open(f'{self.results_dir}/xgboost_metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            return model, metrics, y_test, y_pred
            
        except Exception as e:
            print(f"Error in XGBoost model training: {str(e)}")
            traceback.print_exc()
            return None, None, None, None
        
    def train_pymc(self, X, y):
        try:
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_test_scaled = X_scaler.transform(X_test)
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            
            n_features = X_train_scaled.shape[1]
            n_test = len(X_test)
            
            coords = {
                "obs_id": np.arange(len(X_train_scaled)),
                "features": np.arange(n_features),
                "test_points": np.arange(n_test)
            }
            
            with pm.Model(coords=coords) as model:
                alpha = pm.Normal('alpha', mu=0, sigma=1)
                
                beta_scale = pm.HalfNormal('beta_scale', sigma=0.1)
                beta = pm.Normal('beta', mu=0, sigma=beta_scale, dims='features')
                
                nu = pm.Gamma('nu', alpha=3, beta=1)
                sigma = pm.HalfNormal('sigma', sigma=0.1)
                
                mu = alpha + pm.math.dot(X_train_scaled, beta)
                
                y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=y_train_scaled, dims='obs_id')
                
                mu_test = alpha + pm.math.dot(X_test_scaled, beta)
                
                trace = pm.sample(
                    draws=2000,
                    tune=2000,
                    chains=4,
                    cores=4,
                    random_seed=42,
                    target_accept=0.99,
                    max_treedepth=15,
                    return_inferencedata=True
                )
                
                y_pred_scaled = mu_test.eval()
                y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2)
            }
            
            trace.to_netcdf(f'{self.results_dir}/pymc_model.nc')
            with open(f'{self.results_dir}/pymc_metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            diagnostics = {
                'divergences': int(trace.sample_stats.diverging.sum().item()),
                'max_treedepth_hits': int((trace.sample_stats.tree_depth >= 15).sum().item()),
                'effective_n': float(az.ess(trace)['beta'].mean()),
                'r_hat': float(az.rhat(trace)['beta'].mean())
            }
            
            with open(f'{self.results_dir}/pymc_diagnostics.json', 'w') as f:
                json.dump(diagnostics, f)
            
            return model, trace, metrics, y_test, y_pred
            
        except Exception as e:
            print(f"Error in PyMC model training: {str(e)}")
            traceback.print_exc()
            return None, None, None, None, None
        
    def generate_forecasts(self, model, X, y):
        try:
            last_date = pd.to_datetime(y.index[-1])
            
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=self.forecast_horizon,
                freq='D'
            )
            
            future_features = pd.DataFrame(index=future_dates, columns=X.columns)
            
            future_features.loc[:, 'day_of_week'] = future_features.index.dayofweek
            future_features.loc[:, 'day_of_month'] = future_features.index.day
            future_features.loc[:, 'month'] = future_features.index.month
            future_features.loc[:, 'quarter'] = future_features.index.quarter
            future_features.loc[:, 'year'] = future_features.index.year
            
            last_values = y[-30:].values
            for i, lag in enumerate([1, 7, 14, 30]):
                col_name = f'lag_{lag}'
                if lag == 1:
                    future_features.loc[future_features.index[0], col_name] = y.iloc[-1]
                    future_features.loc[future_features.index[1:], col_name] = np.nan
                else:
                    future_features.loc[future_features.index[:min(lag, len(future_features))], col_name] = last_values[-lag:]
                    future_features.loc[future_features.index[lag:], col_name] = np.nan
            
            for col in future_features.columns:
                if future_features[col].isnull().all():
                    future_features.loc[:, col] = X[col].iloc[-1]
            
            scaler = StandardScaler()
            scaler.fit(X)
            future_features_scaled = scaler.transform(future_features)
            
            if isinstance(model, xgb.Booster):
                dmatrix = xgb.DMatrix(future_features_scaled)
                forecasts = model.predict(dmatrix)
                return future_dates, forecasts, None
            else:
                forecasts = model.alpha.eval() + np.dot(future_features_scaled, model.beta.eval())
                std = np.sqrt(model.sigma.eval()**2 + np.sum((future_features_scaled * model.beta.eval())**2))
                return future_dates, forecasts, std
                
        except Exception as e:
            print(f"Error in forecast generation: {str(e)}")
            traceback.print_exc()
            return None, None, None
        
    def plot_comparison(self, y_test, y_pred_xgb, y_pred_pymc, metrics_xgb, metrics_pymc):
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(15, 12))
        
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(y_test.index, y_test, label='Actual', color='black', alpha=0.7, linewidth=2)
        ax1.plot(y_test.index, y_pred_xgb, label='XGBoost', color='#2ecc71', alpha=0.7)
        ax1.plot(y_test.index, y_pred_pymc, label='PyMC', color='#e74c3c', alpha=0.7)
        ax1.set_title('Model Predictions vs Actual', fontsize=12, pad=15)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Revenue')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 1, 2)
        ax2.scatter(y_test, y_pred_xgb - y_test, alpha=0.5, label='XGBoost', color='#2ecc71')
        ax2.scatter(y_test, y_pred_pymc - y_test, alpha=0.5, label='PyMC', color='#e74c3c')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_title('Residuals Plot', fontsize=12, pad=15)
        ax2.set_xlabel('Actual Revenue')
        ax2.set_ylabel('Residuals')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(3, 1, 3)
        metrics = ['MAE', 'RMSE', 'R2']
        xgb_values = [metrics_xgb['MAE'], metrics_xgb['RMSE'], metrics_xgb['R2']]
        pymc_values = [metrics_pymc['MAE'], metrics_pymc['RMSE'], metrics_pymc['R2']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, xgb_values, width, label='XGBoost', color='#2ecc71', alpha=0.7)
        ax3.bar(x + width/2, pymc_values, width, label='PyMC', color='#e74c3c', alpha=0.7)
        
        ax3.set_title('Model Performance Comparison', fontsize=12, pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_forecasts(self, future_dates, xgb_forecasts, pymc_forecasts, pymc_std=None):
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(15, 8))
        
        xgb_forecasts = np.array(xgb_forecasts)
        pymc_forecasts = np.array(pymc_forecasts)
        
        plt.plot(future_dates, xgb_forecasts, label='XGBoost Forecast', color='#2ecc71', alpha=0.7, linewidth=2)
        plt.plot(future_dates, pymc_forecasts, label='PyMC Forecast', color='#e74c3c', alpha=0.7, linewidth=2)
        
        if pymc_std is not None:
            pymc_std = np.array(pymc_std)
            plt.fill_between(
                future_dates,
                pymc_forecasts - 2 * pymc_std,
                pymc_forecasts + 2 * pymc_std,
                color='#e74c3c',
                alpha=0.2,
                label='PyMC 95% CI'
            )
        
        plt.title('30-Day Revenue Forecast with Uncertainty', fontsize=12, pad=15)
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/forecast_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_analysis(self):
        print("\n Starting Time Series Analysis ===\n")
        
        print("1. Preparing features...")
        X, y = self.prepare_features()
        print(f"   - Total samples: {len(X)}")
        print(f"   - Features: {', '.join(X.columns)}\n")
        
        print("2. Training XGBoost model...")
        xgb_model, xgb_metrics, y_test, y_pred_xgb = self.train_xgboost(X, y)
        print("   XGBoost Metrics:")
        for metric, value in xgb_metrics.items():
            print(f"   - {metric.upper()}: {value:.4f}")
        print()
        
        print("3. Training PyMC model...")
        pymc_model, pymc_trace, pymc_metrics, _, y_pred_pymc = self.train_pymc(X, y)
        print("   PyMC Metrics:")
        for metric, value in pymc_metrics.items():
            print(f"   - {metric.upper()}: {value:.4f}")
        print()
        
        print("4. Generating forecasts...")
        future_dates, xgb_forecasts, _ = self.generate_forecasts(xgb_model, X, y)
        _, pymc_forecasts, pymc_std = self.generate_forecasts(pymc_model, X, y)
        print(f"   - Forecast horizon: {self.forecast_horizon} days\n")
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'XGBoost_Forecast': xgb_forecasts,
            'PyMC_Forecast': pymc_forecasts
        })
        if pymc_std is not None:
            forecast_df['PyMC_CI_Lower'] = pymc_forecasts - 2 * pymc_std
            forecast_df['PyMC_CI_Upper'] = pymc_forecasts + 2 * pymc_std
        
        forecast_df.set_index('Date', inplace=True)
        forecast_df.to_csv(f'{self.results_dir}/forecasts.csv')
        
        print("5. Creating visualizations...")
        self.plot_comparison(y_test, y_pred_xgb, y_pred_pymc, xgb_metrics, pymc_metrics)
        self.plot_forecasts(future_dates, xgb_forecasts, pymc_forecasts, pymc_std)
        print("   - Plots saved in:", self.results_dir)
        print("\n=== Analysis Complete ===\n")

if __name__ == "__main__":
    forecaster = TimeSeriesForecaster()
    forecaster.run_analysis() 