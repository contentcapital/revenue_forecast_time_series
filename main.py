import os
import pandas as pd
import numpy as np
from data_preprocessing import load_and_preprocess_data, analyze_time_series
from model_training import TimeSeriesModels
import warnings
warnings.filterwarnings('ignore')

def run_pipeline():
    print("Starting time series forecasting pipeline...")
    
    print("\nStep 1: Data Preprocessing")
    
    df = load_and_preprocess_data('royalty_revenue_data_with_royalty.csv')
    print("Data preprocessing completed.")
    
    print("\nStep 2: Time Series Analysis")
    
    target_column = 'revenue'
    decomposition = analyze_time_series(df, target_column)
    print("Time series analysis completed. Check time_series_analysis.html for results.")
    
    print("\nStep 3: Model Training and Comparison")
   
    ts_models = TimeSeriesModels(df, target_column=target_column)
    
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
    
    print("\nStep 4: Results Summary")
    print("----------------------")
    metrics_df = pd.read_csv('model_metrics.csv', index_col=0)
    print("\nModel Performance Metrics:")
    print(metrics_df)
    
    print("\nPipeline completed successfully!")
    print("Check the following files for results:")
    print("- time_series_analysis.html: Time series decomposition and analysis")
    print("- forecast_comparison.html: Model predictions comparison")
    print("- model_metrics.csv: Performance metrics for all models")
    print("- model_predictions.csv: Detailed predictions from all models")
    print("- *_model.joblib: Saved model files")
    print("- pymc3_trace.nc: PyMC3 model trace")

if __name__ == "__main__":
    run_pipeline() 