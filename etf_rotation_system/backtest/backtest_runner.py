"""
Complete backtest runner orchestrating all components
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

from data.fetch_data import DataFetcher
from features.indicators import add_all_indicators
from features.monthly_features import MonthlyFeatureEngine
from models.train_model import MLModelTrainer
from models.predict import generate_predictions
from strategies.technical_score import add_technical_scores
from strategies.sector_rotation import generate_allocation_signals
from backtest.portfolio import simulate_trading
from backtest.performance import calculate_performance


class BacktestRunner:
    """Orchestrate complete backtest workflow"""
    
    def __init__(self, start_date='2013-01-01', end_date=None, 
                 top_n_etfs=3, train_window=36, retrain_freq=1,
                 initial_capital=1000000):
        """
        Args:
            start_date: Backtest start date
            end_date: Backtest end date (default: today)
            top_n_etfs: Number of ETFs to hold
            train_window: Training window in months
            retrain_freq: Retrain frequency in months
            initial_capital: Starting capital
        """
        self.start_date = start_date
        self.end_date = end_date
        self.top_n_etfs = top_n_etfs
        self.train_window = train_window
        self.retrain_freq = retrain_freq
        self.initial_capital = initial_capital
        
        # Results
        self.data = None
        self.monthly_features = None
        self.predictions = None
        self.allocations = None
        self.portfolio = None
        self.performance = None
        self.models_dict = None
        self.trainer = None
        
    def run_complete_backtest(self, force_refresh_data=False):
        """
        Run complete backtest pipeline
        
        Args:
            force_refresh_data: Force data refresh
            
        Returns:
            dict: Backtest results
        """
        print("="*80)
        print("STARTING COMPLETE BACKTEST")
        print("="*80)
        print(f"Date Range: {self.start_date} to {self.end_date or 'today'}")
        print(f"Portfolio: Top {self.top_n_etfs} ETFs")
        print(f"Training: {self.train_window} months window, retrain every {self.retrain_freq} month(s)")
        print(f"Capital: ₹{self.initial_capital:,.0f}")
        print()
        
        # Step 1: Fetch data
        print("[1/8] Fetching market data...")
        fetcher = DataFetcher()
        data = fetcher.fetch_data(
            start_date=self.start_date,
            end_date=self.end_date,
            force_refresh=force_refresh_data
        )
        self.data = fetcher.get_aligned_data(data)
        print(f"      Loaded {len(self.data)} ETFs with {len(self.data['NIFTYBEES'])} trading days")
        
        # Step 2: Calculate indicators
        print("\n[2/8] Calculating technical indicators...")
        data_with_indicators = add_all_indicators(self.data)
        print("      Technical indicators calculated")
        
        # Step 3: Generate monthly features
        print("\n[3/8] Generating monthly features...")
        feature_engine = MonthlyFeatureEngine(data_with_indicators)
        monthly_features = feature_engine.create_monthly_features()
        monthly_with_targets = feature_engine.create_targets(monthly_features)
        self.monthly_features = monthly_with_targets
        print(f"      Generated {len(monthly_with_targets)} monthly samples")
        
        # Step 4: Train ML models
        print(f"\n[4/8] Training XGBoost models (walk-forward)...")
        self.trainer = MLModelTrainer(monthly_with_targets)
        self.models_dict, training_history = self.trainer.walk_forward_train(
            train_window_months=self.train_window,
            retrain_frequency=self.retrain_freq
        )
        print(f"      Trained {len(self.models_dict)} models")
        
        # Step 5: Generate predictions
        print("\n[5/8] Generating ML predictions...")
        predictions = generate_predictions(
            monthly_with_targets,
            self.models_dict,
            self.trainer.feature_cols
        )
        print(f"      Generated {len(predictions)} predictions")
        
        # Step 6: Calculate technical scores and final scores
        print("\n[6/8] Calculating technical scores...")
        final_data = add_technical_scores(predictions)
        self.predictions = final_data
        print("      Technical scores calculated")
        
        # Step 7: Generate allocation signals
        print(f"\n[7/8] Generating allocation signals (Top {self.top_n_etfs})...")
        allocations = generate_allocation_signals(final_data, top_n=self.top_n_etfs)
        self.allocations = allocations
        print(f"      Generated {len(allocations)} allocation decisions")
        
        # Step 8: Simulate trading
        print("\n[8/8] Simulating trading...")
        portfolio = simulate_trading(
            self.data,
            allocations,
            initial_capital=self.initial_capital
        )
        self.portfolio = portfolio
        
        # Calculate performance
        print("\nCalculating performance metrics...")
        benchmark_data = self.data['NIFTYBEES']
        performance = calculate_performance(portfolio, benchmark_data)
        self.performance = performance
        
        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)
        
        self._print_performance_summary()
        
        return {
            'portfolio': portfolio,
            'performance': performance,
            'allocations': allocations,
            'predictions': final_data,
            'models': self.models_dict
        }
    
    def _print_performance_summary(self):
        """Print performance summary"""
        if self.performance is None:
            return
        
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 80)
        
        # Returns
        print("\n📈 Returns:")
        print(f"   Total Return:        {self.performance.get('Total_Return_%', 0):>10.2f}%")
        print(f"   CAGR:                {self.performance.get('CAGR_%', 0):>10.2f}%")
        print(f"   Benchmark CAGR:      {self.performance.get('Benchmark_CAGR_%', 0):>10.2f}%")
        print(f"   Alpha:               {self.performance.get('Alpha_%', 0):>10.2f}%")
        
        # Risk
        print("\n📉 Risk:")
        print(f"   Annual Volatility:   {self.performance.get('Annual_Volatility_%', 0):>10.2f}%")
        print(f"   Max Drawdown:        {self.performance.get('Max_Drawdown_%', 0):>10.2f}%")
        print(f"   Sharpe Ratio:        {self.performance.get('Sharpe_Ratio', 0):>10.2f}")
        print(f"   Sortino Ratio:       {self.performance.get('Sortino_Ratio', 0):>10.2f}")
        
        # Win rates
        print("\n🎯 Win Rates:")
        print(f"   Monthly Win Rate:    {self.performance.get('Win_Rate_%', 0):>10.2f}%")
        print(f"   vs Benchmark:        {self.performance.get('Hit_Rate_%', 0):>10.2f}%")
        print(f"   Trade Win Rate:      {self.performance.get('Trade_Win_Rate_%', 0):>10.2f}%")
        
        # Trades
        print("\n💼 Trading Activity:")
        print(f"   Total Trades:        {self.performance.get('Total_Trades', 0):>10}")
        print(f"   Completed Trades:    {self.performance.get('Completed_Trades', 0):>10}")
        print(f"   Avg Trade Return:    {self.performance.get('Avg_Trade_Return_%', 0):>10.2f}%")
        
        print("-" * 80)
    
    def get_current_allocation(self):
        """Get most recent allocation recommendation"""
        if self.allocations is None or self.allocations.empty:
            return None
        
        latest_date = self.allocations['Selected_Date'].max()
        current = self.allocations[self.allocations['Selected_Date'] == latest_date].copy()
        
        return current[['ETF', 'Weight', 'FinalScore', 'ML_Probability', 'TechScore']]
    
    def get_next_month_predictions(self):
        """Get predictions for next month"""
        if self.predictions is None or self.predictions.empty:
            return None
        
        # Get most recent month's data
        latest_date = self.predictions.index.max()
        next_month_data = self.predictions[self.predictions.index == latest_date].copy()
        
        # Sort by FinalScore
        next_month_data = next_month_data.sort_values('FinalScore', ascending=False)
        
        return next_month_data[['ETF', 'FinalScore', 'ML_Probability', 'TechScore', 
                                'RSI_14', 'MACD_Hist', 'ADX_14']]


def run_backtest(start_date='2013-01-01', end_date=None, top_n=3, 
                initial_capital=1000000, force_refresh=False):
    """
    Convenience function to run backtest
    
    Args:
        start_date: Start date
        end_date: End date
        top_n: Number of ETFs to select
        initial_capital: Starting capital
        force_refresh: Force data refresh
        
    Returns:
        BacktestRunner object with results
    """
    runner = BacktestRunner(
        start_date=start_date,
        end_date=end_date,
        top_n_etfs=top_n,
        initial_capital=initial_capital
    )
    
    runner.run_complete_backtest(force_refresh_data=force_refresh)
    
    return runner


if __name__ == '__main__':
    # Run backtest
    runner = run_backtest(
        start_date='2015-01-01',
        top_n=3,
        initial_capital=1000000,
        force_refresh=False
    )
    
    # Show current allocation
    print("\n" + "="*80)
    print("CURRENT RECOMMENDED ALLOCATION")
    print("="*80)
    current = runner.get_current_allocation()
    if current is not None:
        print(current.to_string(index=False))
    
    # Show next month predictions
    print("\n" + "="*80)
    print("NEXT MONTH TOP PREDICTIONS")
    print("="*80)
    predictions = runner.get_next_month_predictions()
    if predictions is not None:
        print(predictions.head(5).to_string(index=False))
