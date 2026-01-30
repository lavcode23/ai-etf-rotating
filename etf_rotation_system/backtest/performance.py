"""
Performance metrics calculation
"""
import pandas as pd
import numpy as np


class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self, portfolio, benchmark_data, risk_free_rate=0.06):
        """
        Args:
            portfolio: Portfolio object with trading history
            benchmark_data: DataFrame with benchmark prices
            risk_free_rate: Annual risk-free rate (default 6% for India)
        """
        self.portfolio = portfolio
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate
        
        # Get equity curve
        self.equity_curve = portfolio.get_equity_curve()
        
        if not self.equity_curve.empty:
            self.equity_curve['Date'] = pd.to_datetime(self.equity_curve['Date'])
            self.equity_curve = self.equity_curve.set_index('Date')
    
    def calculate_returns_metrics(self):
        """Calculate return-based metrics"""
        if self.equity_curve.empty:
            return {}
        
        # Total return
        total_return = (self.equity_curve['Portfolio_Value'].iloc[-1] / 
                       self.portfolio.initial_capital - 1) * 100
        
        # Calculate daily returns
        daily_returns = self.equity_curve['Portfolio_Value'].pct_change().dropna()
        
        # Annualized return (CAGR)
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        cagr = ((self.equity_curve['Portfolio_Value'].iloc[-1] / 
                self.portfolio.initial_capital) ** (1/years) - 1) * 100
        
        # Annualized volatility
        annual_vol = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        excess_returns = (daily_returns.mean() * 252 - self.risk_free_rate)
        sharpe = excess_returns / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = excess_returns / downside_std if downside_std > 0 else 0
        
        # Calmar ratio (CAGR / Max Drawdown)
        max_dd = self.calculate_max_drawdown()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'Total_Return_%': total_return,
            'CAGR_%': cagr,
            'Annual_Volatility_%': annual_vol,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Calmar_Ratio': calmar,
            'Years': years
        }
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        if self.equity_curve.empty:
            return 0
        
        cumulative = self.equity_curve['Portfolio_Value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        max_dd = drawdown.min()
        
        # Find drawdown period
        max_dd_idx = drawdown.idxmin()
        peak_idx = running_max[:max_dd_idx].idxmax()
        
        return max_dd
    
    def calculate_drawdown_series(self):
        """Calculate drawdown series"""
        if self.equity_curve.empty:
            return pd.Series()
        
        cumulative = self.equity_curve['Portfolio_Value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        return drawdown
    
    def calculate_monthly_metrics(self):
        """Calculate monthly performance metrics"""
        monthly_returns = self.portfolio.calculate_monthly_returns()
        
        if monthly_returns.empty:
            return {}
        
        monthly_rets = monthly_returns['Monthly_Return'].dropna()
        
        # Win rate
        winning_months = (monthly_rets > 0).sum()
        total_months = len(monthly_rets)
        win_rate = (winning_months / total_months * 100) if total_months > 0 else 0
        
        # Best and worst months
        best_month = monthly_rets.max()
        worst_month = monthly_rets.min()
        
        # Average win/loss
        avg_win = monthly_rets[monthly_rets > 0].mean() if (monthly_rets > 0).any() else 0
        avg_loss = monthly_rets[monthly_rets < 0].mean() if (monthly_rets < 0).any() else 0
        
        # Win/loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'Total_Months': total_months,
            'Winning_Months': winning_months,
            'Win_Rate_%': win_rate,
            'Best_Month_%': best_month,
            'Worst_Month_%': worst_month,
            'Avg_Win_%': avg_win,
            'Avg_Loss_%': avg_loss,
            'Win_Loss_Ratio': win_loss_ratio
        }
    
    def calculate_benchmark_comparison(self):
        """Compare performance against benchmark"""
        if self.equity_curve.empty or self.benchmark_data.empty:
            return {}
        
        # Align dates
        portfolio_dates = self.equity_curve.index
        benchmark_aligned = self.benchmark_data.loc[portfolio_dates[0]:portfolio_dates[-1]]
        
        # Calculate benchmark returns
        benchmark_total_return = (benchmark_aligned['Close'].iloc[-1] / 
                                 benchmark_aligned['Close'].iloc[0] - 1) * 100
        
        # Calculate benchmark CAGR
        days = (benchmark_aligned.index[-1] - benchmark_aligned.index[0]).days
        years = days / 365.25
        benchmark_cagr = ((benchmark_aligned['Close'].iloc[-1] / 
                          benchmark_aligned['Close'].iloc[0]) ** (1/years) - 1) * 100
        
        # Alpha (excess return)
        portfolio_metrics = self.calculate_returns_metrics()
        alpha = portfolio_metrics['CAGR_%'] - benchmark_cagr
        
        # Hit rate (months beating benchmark)
        portfolio_monthly = self.portfolio.calculate_monthly_returns()
        
        if not portfolio_monthly.empty:
            portfolio_monthly['Date'] = pd.to_datetime(portfolio_monthly['Date'])
            portfolio_monthly = portfolio_monthly.set_index('Date')
            
            # Calculate benchmark monthly returns
            benchmark_monthly = benchmark_aligned['Close'].resample('M').last().pct_change() * 100
            
            # Merge
            comparison = portfolio_monthly.join(benchmark_monthly.rename('Benchmark_Return'), how='inner')
            comparison = comparison.dropna()
            
            outperformed = (comparison['Monthly_Return'] > comparison['Benchmark_Return']).sum()
            total_months = len(comparison)
            hit_rate = (outperformed / total_months * 100) if total_months > 0 else 0
        else:
            hit_rate = 0
            outperformed = 0
            total_months = 0
        
        return {
            'Benchmark_Total_Return_%': benchmark_total_return,
            'Benchmark_CAGR_%': benchmark_cagr,
            'Alpha_%': alpha,
            'Hit_Rate_%': hit_rate,
            'Months_Outperformed': outperformed,
            'Total_Comparison_Months': total_months
        }
    
    def calculate_trade_metrics(self):
        """Calculate trade-related metrics"""
        trades_df = self.portfolio.get_trades()
        
        if trades_df.empty:
            return {}
        
        # Filter sell trades (which have PnL)
        sell_trades = trades_df[trades_df['Action'] == 'SELL'].copy()
        
        if sell_trades.empty:
            return {
                'Total_Trades': len(trades_df),
                'Completed_Trades': 0,
                'Winning_Trades': 0,
                'Trade_Win_Rate_%': 0,
                'Avg_Trade_Return_%': 0,
                'Best_Trade_%': 0,
                'Worst_Trade_%': 0
            }
        
        # Trade statistics
        winning_trades = (sell_trades['PnL'] > 0).sum()
        total_completed = len(sell_trades)
        trade_win_rate = (winning_trades / total_completed * 100) if total_completed > 0 else 0
        
        avg_trade_return = sell_trades['PnL_Pct'].mean()
        best_trade = sell_trades['PnL_Pct'].max()
        worst_trade = sell_trades['PnL_Pct'].min()
        
        # Calculate turnover
        buy_trades = trades_df[trades_df['Action'] == 'BUY']
        total_traded_value = buy_trades['Value'].sum()
        avg_portfolio_value = self.equity_curve['Portfolio_Value'].mean()
        turnover = (total_traded_value / avg_portfolio_value) if avg_portfolio_value > 0 else 0
        
        return {
            'Total_Trades': len(trades_df),
            'Completed_Trades': total_completed,
            'Winning_Trades': winning_trades,
            'Trade_Win_Rate_%': trade_win_rate,
            'Avg_Trade_Return_%': avg_trade_return,
            'Best_Trade_%': best_trade,
            'Worst_Trade_%': worst_trade,
            'Turnover': turnover
        }
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {}
        
        # Returns metrics
        returns_metrics = self.calculate_returns_metrics()
        report.update(returns_metrics)
        
        # Drawdown
        max_dd = self.calculate_max_drawdown()
        report['Max_Drawdown_%'] = max_dd
        
        # Monthly metrics
        monthly_metrics = self.calculate_monthly_metrics()
        report.update(monthly_metrics)
        
        # Benchmark comparison
        benchmark_metrics = self.calculate_benchmark_comparison()
        report.update(benchmark_metrics)
        
        # Trade metrics
        trade_metrics = self.calculate_trade_metrics()
        report.update(trade_metrics)
        
        return report


def calculate_performance(portfolio, benchmark_data):
    """
    Calculate comprehensive performance metrics
    
    Args:
        portfolio: Portfolio object
        benchmark_data: Benchmark price data
        
    Returns:
        dict: Performance metrics
    """
    metrics = PerformanceMetrics(portfolio, benchmark_data)
    report = metrics.generate_performance_report()
    
    return report


if __name__ == '__main__':
    from data.fetch_data import DataFetcher
    from features.indicators import add_all_indicators
    from features.monthly_features import MonthlyFeatureEngine
    from models.train_model import MLModelTrainer
    from models.predict import generate_predictions
    from strategies.technical_score import add_technical_scores
    from strategies.sector_rotation import generate_allocation_signals
    from backtest.portfolio import simulate_trading
    
    # Fetch data
    print("Fetching data...")
    fetcher = DataFetcher()
    data = fetcher.fetch_data()
    aligned_data = fetcher.get_aligned_data(data)
    
    # Add indicators
    print("\nAdding indicators...")
    data_with_indicators = add_all_indicators(aligned_data)
    
    # Create monthly features
    print("\nCreating monthly features...")
    feature_engine = MonthlyFeatureEngine(data_with_indicators)
    monthly_features = feature_engine.create_monthly_features()
    monthly_with_targets = feature_engine.create_targets(monthly_features)
    
    # Train models
    print("\nTraining models...")
    trainer = MLModelTrainer(monthly_with_targets)
    models_dict, _ = trainer.walk_forward_train(train_window_months=36, retrain_frequency=3)
    
    # Generate predictions and scores
    predictions = generate_predictions(monthly_with_targets, models_dict, trainer.feature_cols)
    final_data = add_technical_scores(predictions)
    
    # Generate allocations
    allocations = generate_allocation_signals(final_data, top_n=3)
    
    # Simulate trading
    portfolio = simulate_trading(aligned_data, allocations)
    
    # Calculate performance
    print("\nCalculating performance metrics...")
    benchmark_data = aligned_data['NIFTYBEES']
    performance = calculate_performance(portfolio, benchmark_data)
    
    print("\n" + "="*60)
    print("PERFORMANCE REPORT")
    print("="*60)
    for metric, value in performance.items():
        if isinstance(value, float):
            print(f"{metric:30s}: {value:10.2f}")
        else:
            print(f"{metric:30s}: {value:10}")
