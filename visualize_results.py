import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class ResultsVisualizer:
    def __init__(self, data_path='results/data/processed_data.csv'):
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.results_dir = 'results/presentation_plots'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.platform_contrib = None
        self.seasonal_analysis = None
        self.load_analysis_results()
        
    def load_analysis_results(self):
        try:
            with open('results/data/platform_contrib.json', 'r') as f:
                self.platform_contrib = json.load(f)
        except FileNotFoundError:
            pass
            
        try:
            with open('results/data/seasonality_metrics.json', 'r') as f:
                self.seasonal_analysis = json.load(f)
        except FileNotFoundError:
            pass
            
    def plot_revenue_overview(self):
        plt.figure(figsize=(15, 8))
        plt.plot(self.data.index, self.data['Total Revenue'], color='#3498db', linewidth=2)
        plt.title('Total Revenue Over Time', fontsize=14, pad=15)
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/revenue_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_platform_distribution(self):
        if self.platform_contrib is None:
            return
            
        platforms = list(self.platform_contrib.keys())
        revenues = [self.platform_contrib[p]['total_revenue'] for p in platforms]
        
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=self.data[platforms], palette='viridis')
        plt.title('Platform Revenue Distribution', fontsize=14, pad=15)
        plt.xlabel('Platform')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/platform_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_platform_growth(self):
        if self.platform_contrib is None:
            return
            
        platforms = list(self.platform_contrib.keys())
        growth_rates = [self.platform_contrib[p]['growth_rate'] for p in platforms]
        
        plt.figure(figsize=(12, 6))
        plt.bar(platforms, growth_rates, color='#2ecc71', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.title('Platform Growth Rates', fontsize=14, pad=15)
        plt.xlabel('Platform')
        plt.ylabel('Growth Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/platform_growth.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_platform_performance_table(self):
        if self.platform_contrib is None:
            return
            
        platforms = list(self.platform_contrib.keys())
        metrics = ['total_revenue', 'contribution', 'relative_performance']
        data = []
        
        for platform in platforms:
            row = [platform]
            for metric in metrics:
                value = self.platform_contrib[platform][metric]
                if metric == 'contribution':
                    value = f"{value:.1%}"
                elif metric == 'relative_performance':
                    value = f"{value:.2f}"
                else:
                    value = f"${value:,.2f}"
                row.append(value)
            data.append(row)
            
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Platform', 'Total Revenue', 'Contribution', 'Relative Performance'],
                      fill_color='#3498db',
                      align='left'),
            cells=dict(values=list(zip(*data)),
                     fill_color='white',
                     align='left'))
        ])
        
        fig.update_layout(
            title='Platform Performance Metrics',
            title_x=0.5,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        fig.write_image(f'{self.results_dir}/platform_performance_table.png')
        
    def plot_daily_pattern(self):
        if self.seasonal_analysis is None:
            return
            
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = self.seasonal_analysis['daily']['average']
        daily_std = self.seasonal_analysis['daily']['std']
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(days, daily_avg, yerr=daily_std, fmt='o-', color='#e74c3c', 
                    capsize=5, capthick=2, linewidth=2)
        plt.title('Average Daily Revenue Pattern', fontsize=14, pad=15)
        plt.xlabel('Day of Week')
        plt.ylabel('Revenue')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/daily_pattern.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_monthly_pattern(self):
        if self.seasonal_analysis is None:
            return
            
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_avg = self.seasonal_analysis['monthly']['average']
        monthly_std = self.seasonal_analysis['monthly']['std']
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(months, monthly_avg, yerr=monthly_std, fmt='o-', color='#9b59b6', 
                    capsize=5, capthick=2, linewidth=2)
        plt.title('Average Monthly Revenue Pattern', fontsize=14, pad=15)
        plt.xlabel('Month')
        plt.ylabel('Revenue')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/monthly_pattern.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_seasonal_strength(self):
        if self.seasonal_analysis is None:
            return
            
        seasonal_strength = self.seasonal_analysis['seasonal_strength']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=seasonal_strength,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Seasonal Strength"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 0.3], 'color': "#e74c3c"},
                    {'range': [0.3, 0.7], 'color': "#f1c40f"},
                    {'range': [0.7, 1], 'color': "#2ecc71"}
                ]
            }
        ))
        
        fig.update_layout(
            title='Seasonal Strength Analysis',
            title_x=0.5,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        fig.write_image(f'{self.results_dir}/seasonal_strength.png')
        
    def create_dashboard(self):
        self.plot_revenue_overview()
        self.plot_platform_distribution()
        self.plot_platform_growth()
        self.plot_platform_performance_table()
        self.plot_daily_pattern()
        self.plot_monthly_pattern()
        self.plot_seasonal_strength()
        
if __name__ == "__main__":
    visualizer = ResultsVisualizer()
    visualizer.create_dashboard() 