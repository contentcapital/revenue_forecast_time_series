import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_time_features(df):
    try:
        df_features = df.copy()
        
        df_features['year'] = df_features.index.year
        df_features['month'] = df_features.index.month
        df_features['day'] = df_features.index.day
        df_features['dayofweek'] = df_features.index.dayofweek
        df_features['quarter'] = df_features.index.quarter
        df_features['dayofyear'] = df_features.index.dayofyear
        df_features['weekofyear'] = df_features.index.isocalendar().week
        
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month']/12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month']/12)
        df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek']/7)
        df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek']/7)
        
        return df_features
    except Exception as e:
        print(f"Error creating time features: {str(e)}")
        raise

def create_lag_features(df, target_column, lags=[1, 7, 14, 30, 90, 180, 365]):
    try:
        df_lags = df.copy()
        
        for lag in lags:
            df_lags[f'{target_column}_lag_{lag}'] = df_lags[target_column].shift(lag)
        
        return df_lags
    except Exception as e:
        print(f"Error creating lag features: {str(e)}")
        raise

def create_rolling_features(df, target_column, windows=[7, 14, 30, 90]):
    try:
        df_rolling = df.copy()
        
        for window in windows:
            df_rolling[f'{target_column}_rolling_mean_{window}'] = df_rolling[target_column].rolling(window=window).mean()
            df_rolling[f'{target_column}_rolling_std_{window}'] = df_rolling[target_column].rolling(window=window).std()
            df_rolling[f'{target_column}_rolling_min_{window}'] = df_rolling[target_column].rolling(window=window).min()
            df_rolling[f'{target_column}_rolling_max_{window}'] = df_rolling[target_column].rolling(window=window).max()
        
        return df_rolling
    except Exception as e:
        print(f"Error creating rolling features: {str(e)}")
        raise

def create_expanding_features(df, target_column):
    try:
        df_expanding = df.copy()
        
        df_expanding[f'{target_column}_expanding_mean'] = df_expanding[target_column].expanding().mean()
        df_expanding[f'{target_column}_expanding_std'] = df_expanding[target_column].expanding().std()
        df_expanding[f'{target_column}_expanding_min'] = df_expanding[target_column].expanding().min()
        df_expanding[f'{target_column}_expanding_max'] = df_expanding[target_column].expanding().max()
        
        return df_expanding
    except Exception as e:
        print(f"Error creating expanding features: {str(e)}")
        raise

def create_seasonal_features(df, target_column):
    try:
        df_seasonal = df.copy()
        
        monthly_avg = df_seasonal.groupby('month')[target_column].mean()
        df_seasonal['monthly_avg'] = df_seasonal['month'].map(monthly_avg)
        
        weekly_avg = df_seasonal.groupby('dayofweek')[target_column].mean()
        df_seasonal['weekly_avg'] = df_seasonal['dayofweek'].map(weekly_avg)
        
        return df_seasonal
    except Exception as e:
        print(f"Error creating seasonal features: {str(e)}")
        raise

def scale_features(df, exclude_columns=None):
    try:
        if exclude_columns is None:
            exclude_columns = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        scaler = StandardScaler()
        
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        
        return df_scaled, scaler
    except Exception as e:
        print(f"Error scaling features: {str(e)}")
        raise

def engineer_features(df, target_column):
    try:
        df_features = create_time_features(df)
        
        df_features = create_lag_features(df_features, target_column)
        
        df_features = create_rolling_features(df_features, target_column)
        
        df_features = create_expanding_features(df_features, target_column)
        
        df_features = create_seasonal_features(df_features, target_column)
        
        exclude_cols = [target_column, 'year', 'month', 'day', 'dayofweek', 
                       'quarter', 'dayofyear', 'weekofyear']
        df_features, scaler = scale_features(df_features, exclude_columns=exclude_cols)
        
        df_features = df_features.dropna()
        
        return df_features, scaler
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        df = pd.read_csv('preprocessed_data.csv', index_col=0, parse_dates=True)
        
        target_column = 'revenue'
        
        df_features, scaler = engineer_features(df, target_column)
        
        df_features.to_csv('engineered_features.csv')
        print("Feature engineering completed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise 