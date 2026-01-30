"""
Chart generation for reports and dashboard
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ChartGenerator:
    """Generate visualization charts"""
    
    def __init__(self, backtest_runner, output_dir='reports/charts'):
        """
        Args:
            backtest_runner: BacktestRunner object
            output_dir: Output directory for charts
        """
        self.runner = backtest_runner
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_all_charts(self):
        """Generate all charts"""
        print("\nGenerating charts...")
        
        charts = {}
        
        # Equity curve
        print("  - Equity curve vs benchmark")
        charts['equity_curve'] = self.plot_equity_curve()
        
        # Drawdown
        print("  - Drawdown chart")
        charts['drawdown'] = self.plot_drawdown()
        
        # Monthly returns
        print("  - Monthly returns heatmap")
        charts['monthly_heatmap'] = self.plot_monthly_returns_heatmap()
        
        # Sector rotation
        print("  - Sector rotation heatmap")
        charts['sector_rotation'] = self.plot_sector_rotation()
        
        # Feature importance
        print("  - Feature importance")
        charts['feature_importance'] = self.plot_feature_importance()
        
        # Monthly performance
        print("  - Monthly performance bars")
        charts['monthly_bars'] = self.plot_monthly_performance()
        
        # Rolling metrics
        print("  - Rolling Sharpe ratio")
        charts['rolling_sharpe'] = self.plot_rolling_sharpe()
        
        print(f"Charts saved to {self.output_dir}/")
        
        return charts
    
    def plot_equity_curve(self):
        """Plot portfolio equity curve vs benchmark"""
        equity_df = self.runner.portfolio.get_equity_curve()
        
        if equity_df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        
        # Plot portfolio
        ax.plot(equity_df['Date'], equity_df['Portfolio_Value'],
               label='Strategy', linewidth=2, color='#2E86AB')
        
        # Plot benchmark
        benchmark_data = self.runner.data['NIFTYBEES']
        benchmark_dates = equity_df['Date']
        
        benchmark_aligned = benchmark_data.loc[
            benchmark_dates.min():benchmark_dates.max()
        ]
        
        benchmark_normalized = (
            (benchmark_aligned['Close'] / benchmark_aligned['Close'].iloc[0]) *
            self.runner.initial_capital
        )
        
        ax.plot(benchmark_aligned.index, benchmark_normalized,
               label='NIFTYBEES (Benchmark)', linewidth=2,
               color='#A23B72', linestyle='--')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax.set_title('Equity Curve: Strategy vs Benchmark', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1e6:.1f}M'))
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'equity_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_drawdown(self):
        """Plot drawdown chart"""
        equity_df = self.runner.portfolio.get_equity_curve()
        
        if equity_df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        equity_df = equity_df.set_index('Date')
        
        # Calculate drawdown
        cumulative = equity_df['Portfolio_Value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        ax.fill_between(drawdown.index, drawdown, 0, color='#E63946', alpha=0.7)
        ax.plot(drawdown.index, drawdown, color='#C1121F', linewidth=1.5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'drawdown.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_monthly_returns_heatmap(self):
        """Plot monthly returns heatmap"""
        monthly_returns = self.runner.portfolio.calculate_monthly_returns()
        
        if monthly_returns.empty:
            return None
        
        # Create pivot table
        monthly_returns['Date'] = pd.to_datetime(monthly_returns['Date'])
        monthly_returns['Year'] = monthly_returns['Date'].dt.year
        monthly_returns['Month'] = monthly_returns['Date'].dt.month
        
        pivot_data = monthly_returns.pivot(
            index='Year',
            columns='Month',
            values='Monthly_Return'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Return (%)'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels, rotation=0)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'monthly_returns_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_sector_rotation(self):
        """Plot sector rotation heatmap"""
        if self.runner.allocations is None or self.runner.allocations.empty:
            return None
        
        allocations = self.runner.allocations.copy()
        allocations['Date'] = pd.to_datetime(allocations['Selected_Date'])
        allocations['YearMonth'] = allocations['Date'].dt.to_period('M')
        
        # Create pivot table
        pivot_data = allocations.pivot_table(
            index='ETF',
            columns='YearMonth',
            values='Weight',
            fill_value=0
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sns.heatmap(
            pivot_data * 100,  # Convert to percentage
            cmap='YlOrRd',
            cbar_kws={'label': 'Allocation (%)'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('ETF', fontsize=12)
        ax.set_title('Sector Rotation Over Time', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'sector_rotation.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_feature_importance(self):
        """Plot feature importance from ML models"""
        if self.runner.trainer is None or self.runner.models_dict is None:
            return None
        
        importance_df = self.runner.trainer.get_feature_importance(
            self.runner.models_dict
        )
        
        # Plot top 15 features
        top_features = importance_df.head(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(
            range(len(top_features)),
            top_features['Importance'],
            color='#457B9D'
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Top 15 Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_monthly_performance(self):
        """Plot monthly performance as bars"""
        monthly_returns = self.runner.portfolio.calculate_monthly_returns()
        
        if monthly_returns.empty:
            return None
        
        monthly_returns = monthly_returns.dropna()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ['#06D6A0' if x > 0 else '#EF476F'
                 for x in monthly_returns['Monthly_Return']]
        
        ax.bar(
            range(len(monthly_returns)),
            monthly_returns['Monthly_Return'],
            color=colors,
            alpha=0.7
        )
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Month Number', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_title('Monthly Returns', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'monthly_performance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_rolling_sharpe(self, window=6):
        """Plot rolling Sharpe ratio"""
        monthly_returns = self.runner.portfolio.calculate_monthly_returns()
        
        if monthly_returns.empty or len(monthly_returns) < window:
            return None
        
        monthly_returns = monthly_returns.dropna()
        monthly_rets = monthly_returns['Monthly_Return'] / 100
        
        # Calculate rolling Sharpe
        rolling_mean = monthly_rets.rolling(window=window).mean() * 12
        rolling_std = monthly_rets.rolling(window=window).std() * np.sqrt(12)
        rolling_sharpe = (rolling_mean - 0.06) / rolling_std
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        monthly_returns['Date'] = pd.to_datetime(monthly_returns['Date'])
        
        ax.plot(
            monthly_returns['Date'],
            rolling_sharpe,
            linewidth=2,
            color='#1D3557'
        )
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.set_title(f'{window}-Month Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'rolling_sharpe.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath


def generate_charts(backtest_runner, output_dir='reports/charts'):
    """
    Generate all charts
    
    Args:
        backtest_runner: BacktestRunner object
        output_dir: Output directory
        
    Returns:
        dict: Paths to generated charts
    """
    generator = ChartGenerator(backtest_runner, output_dir=output_dir)
    charts = generator.generate_all_charts()
    
    return charts


if __name__ == '__main__':
    from backtest.backtest_runner import run_backtest
    
    # Run backtest
    print("Running backtest...")
    runner = run_backtest(
        start_date='2015-01-01',
        top_n=3,
        initial_capital=1000000
    )
    
    # Generate charts
    print("\nGenerating charts...")
    charts = generate_charts(runner)
    
    print("\n" + "="*80)
    print("CHARTS GENERATED")
    print("="*80)
    for chart_name, filepath in charts.items():
        if filepath:
            print(f"{chart_name:25s}: {filepath}")
