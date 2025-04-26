import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

def create_time_series_plots(df, target_column):
   
    try:
      
        plt.style.use('default')
        
      
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df[target_column], linewidth=2)
        plt.title('Time Series of ' + target_column)
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/plots/time_series.png')
        plt.close()
        
     
        plt.figure(figsize=(15, 6))
        df['Month'] = df.index.month
        monthly_data = [df[df['Month'] == i][target_column] for i in range(1, 13)]
        plt.boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.title('Monthly Distribution of ' + target_column)
        plt.ylabel(target_column)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/plots/monthly_boxplot.png')
        plt.close()
        
     
        plt.figure(figsize=(15, 6))
        yearly_mean = df.groupby(df.index.year)[target_column].mean()
        plt.plot(yearly_mean.index, yearly_mean.values, marker='o')
        plt.title('Yearly Trend of ' + target_column)
        plt.xlabel('Year')
        plt.ylabel('Average ' + target_column)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('results/plots/yearly_trend.png')
        plt.close()
        
      
        decomposition = seasonal_decompose(df[target_column], period=12)
        
        plt.figure(figsize=(15, 12))
        plt.subplot(411)
        plt.plot(df.index, decomposition.observed)
        plt.title('Observed')
        plt.grid(True)
        
        plt.subplot(412)
        plt.plot(df.index, decomposition.trend)
        plt.title('Trend')
        plt.grid(True)
        
        plt.subplot(413)
        plt.plot(df.index, decomposition.seasonal)
        plt.title('Seasonal')
        plt.grid(True)
        
        plt.subplot(414)
        plt.plot(df.index, decomposition.resid)
        plt.title('Residual')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/plots/seasonal_decomposition.png')
        plt.close()
        
        return decomposition
        
    except Exception as e:
        print(f"Error in creating plots: {str(e)}")
        raise

def analyze_seasonality(df, target_column):
    """
    Analyze and save seasonality patterns
    """
    try:
        seasonality_metrics = {}
       
        monthly_avg = df.groupby(df.index.month)[target_column].mean()
        monthly_std = df.groupby(df.index.month)[target_column].std()
        
       
        quarterly_avg = df.groupby(df.index.quarter)[target_column].mean()
        quarterly_std = df.groupby(df.index.quarter)[target_column].std()
       
        daily_avg = df.groupby(df.index.dayofweek)[target_column].mean()
        daily_std = df.groupby(df.index.dayofweek)[target_column].std()
        
        seasonality_metrics = {
            'monthly': {
                'average': monthly_avg.to_dict(),
                'std': monthly_std.to_dict()
            },
            'quarterly': {
                'average': quarterly_avg.to_dict(),
                'std': quarterly_std.to_dict()
            },
            'daily': {
                'average': daily_avg.to_dict(),
                'std': daily_std.to_dict()
            }
        }
        
       
        with open('results/data/statistics/seasonality_metrics.json', 'w') as f:
            json.dump(seasonality_metrics, f, indent=4)
        
        return seasonality_metrics
        
    except Exception as e:
        print(f"Error in analyzing seasonality: {str(e)}")
        raise

def calculate_statistics(df, target_column):
    """
    Calculate and save important statistics
    """
    try:
        stats = {
            'basic_stats': df[target_column].describe().to_dict(),
            'rolling_stats': {
                'mean_7d': df[target_column].rolling(7).mean().mean(),
                'mean_30d': df[target_column].rolling(30).mean().mean(),
                'std_7d': df[target_column].rolling(7).std().mean(),
                'std_30d': df[target_column].rolling(30).std().mean()
            }
        }
        
        # Save statistics
        with open('results/data/statistics/statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        return stats
        
    except Exception as e:
        print(f"Error in calculating statistics: {str(e)}")
        raise

def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    try:
      
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
      
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
                
        if df is None:
            raise ValueError("Could not read the file with any of the attempted encodings")
       
        if df.empty:
            raise ValueError("The loaded data is empty")
        
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
          
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(df[first_col])
                df.set_index(first_col, inplace=True)
            except:
                raise ValueError("No suitable date column found for time series analysis")
                
      
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    try:
        
        missing_pct = df.isnull().mean() * 100
        
       
        cols_to_drop = missing_pct[missing_pct > 50].index
        df = df.drop(columns=cols_to_drop)
        
      
        df = df.ffill(limit=3)
       
        df = df.bfill(limit=3)
        
    
        df = df.interpolate(method='time')
     
        df = df.fillna(df.mean())
        
        return df
        
    except Exception as e:
        print(f"Error in handling missing values: {str(e)}")
        raise

def handle_outliers(df, threshold=3):
    """Handle outliers in the dataset using IQR method"""
    try:
        for col in df.select_dtypes(include=[np.number]).columns:
          
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
       
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
         
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
         
            if outliers.any():
                df.loc[outliers, col] = df[col].astype(float).median()
        
        return df
        
    except Exception as e:
        print(f"Error in handling outliers: {str(e)}")
        raise

def check_stationarity(df, column):
    """Check if the time series is stationary using ADF test"""
    try:
        
        result = adfuller(df[column].dropna())
  
        stationarity_results = {
            'is_stationary': result[1] < 0.05,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }
        
        return stationarity_results
        
    except Exception as e:
        print(f"Error in checking stationarity: {str(e)}")
        raise

def make_stationary(df):
    """
    Make the time series stationary using differencing
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df[f'{col}_diff'] = df[col].diff()
            df[f'{col}_diff'] = df[f'{col}_diff'].fillna(0)
        
        return df
    except Exception as e:
        print(f"Error in making data stationary: {str(e)}")
        raise

if __name__ == "__main__":
   
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/data/statistics', exist_ok=True)
  
    df = load_and_preprocess_data('royalty_revenue_data.csv')
   
    target_column = 'Total Revenue'
    platform_columns = [col for col in df.columns if 'Revenue' in col and col != 'Total Revenue']
    
   
    raw_stats = df.describe()
    raw_stats.to_csv('results/data/statistics/raw_data_statistics.csv')

    df = handle_missing_values(df)
    
    missing_stats = df.isnull().sum()
    missing_stats.to_csv('results/data/statistics/missing_value_statistics.csv')
  
    df = handle_outliers(df)
    

    outlier_stats = df.describe()
    outlier_stats.to_csv('results/data/statistics/post_outlier_statistics.csv')
    
   
    stationarity_results = check_stationarity(df, target_column)
    with open('results/data/statistics/stationarity_test.json', 'w') as f:
        json.dump(stationarity_results, f, indent=4)
    
   
    if not stationarity_results['is_stationary']:
        df = make_stationary(df, target_column)
    
  
    create_time_series_plots(df, target_column)
    
   
    seasonality_metrics = analyze_seasonality(df, target_column)
    with open('results/data/statistics/seasonality_metrics.json', 'w') as f:
        json.dump(seasonality_metrics, f, indent=4)
    
   
    total_stats = calculate_statistics(df, target_column)
    
   
    platform_stats = {}
    for platform in platform_columns:
        platform_stats[platform] = {
            'total_revenue': df[platform].sum(),
            'mean_revenue': df[platform].mean(),
            'std_revenue': df[platform].std(),
            'contribution': (df[platform].sum() / df[target_column].sum()) * 100
        }
    
   
    statistics = {
        'total_revenue': total_stats,
        'platform_statistics': platform_stats
    }
    
    with open('results/data/statistics/statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)
    
   
    df.to_csv('results/data/processed_data.csv')
    
    print("Data preprocessing completed successfully!")
    print("Results saved in the 'results' directory.") 