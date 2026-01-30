"""
Sector rotation strategy with risk management
"""
import pandas as pd
import numpy as np


class SectorRotationStrategy:
    """Monthly sector rotation strategy"""
    
    def __init__(self, top_n=3, max_exposure=0.90, cash_reserve=0.10):
        """
        Args:
            top_n: Number of top ETFs to select
            max_exposure: Maximum capital exposure (0.90 = 90%)
            cash_reserve: Minimum cash reserve (0.10 = 10%)
        """
        self.top_n = top_n
        self.max_exposure = max_exposure
        self.cash_reserve = cash_reserve
        
    def select_etfs(self, monthly_scores, date):
        """
        Select top ETFs for a given date based on FinalScore
        
        Args:
            monthly_scores: DataFrame with FinalScore for all ETFs
            date: Target date for selection
            
        Returns:
            DataFrame with selected ETFs and their weights
        """
        # Filter data for the specific date
        date_data = monthly_scores[monthly_scores.index == date].copy()
        
        if len(date_data) == 0:
            return pd.DataFrame()
        
        # Apply risk filters
        date_data = self._apply_risk_filters(date_data)
        
        if len(date_data) == 0:
            print(f"  Warning: No ETFs passed risk filters for {date}")
            return pd.DataFrame()
        
        # Sort by FinalScore
        date_data = date_data.sort_values('FinalScore', ascending=False)
        
        # Select top N
        selected = date_data.head(self.top_n).copy()
        
        # Calculate equal weights
        n_selected = len(selected)
        equal_weight = self.max_exposure / n_selected
        
        selected['Weight'] = equal_weight
        selected['Selected_Date'] = date
        
        return selected
    
    def _apply_risk_filters(self, data):
        """
        Apply risk management filters
        
        Filters:
        - ADX >= 15 (minimum trend strength)
        - RSI <= 80 (not extremely overbought)
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Filtered DataFrame
        """
        filtered = data.copy()
        
        # ADX filter
        if 'ADX_14' in filtered.columns:
            filtered = filtered[filtered['ADX_14'] >= 15]
        
        # RSI filter
        if 'RSI_14' in filtered.columns:
            filtered = filtered[filtered['RSI_14'] <= 80]
        
        return filtered
    
    def generate_monthly_allocations(self, monthly_scores):
        """
        Generate monthly allocations for entire backtest period
        
        Args:
            monthly_scores: DataFrame with FinalScore for all ETFs
            
        Returns:
            DataFrame with monthly allocation decisions
        """
        unique_dates = sorted(monthly_scores.index.unique())
        
        all_allocations = []
        
        for date in unique_dates:
            selected = self.select_etfs(monthly_scores, date)
            
            if not selected.empty:
                all_allocations.append(selected)
        
        if not all_allocations:
            return pd.DataFrame()
        
        allocations_df = pd.concat(all_allocations, axis=0)
        
        return allocations_df
    
    def get_portfolio_composition(self, allocations_df, date):
        """
        Get portfolio composition for a specific date
        
        Args:
            allocations_df: DataFrame with all allocations
            date: Target date
            
        Returns:
            DataFrame with portfolio composition
        """
        portfolio = allocations_df[allocations_df['Selected_Date'] == date].copy()
        
        if portfolio.empty:
            return pd.DataFrame()
        
        # Add cash position
        total_exposure = portfolio['Weight'].sum()
        cash_weight = 1.0 - total_exposure
        
        portfolio_summary = portfolio[['ETF', 'Weight', 'FinalScore', 'ML_Probability', 'TechScore']].copy()
        
        # Add cash row
        cash_row = pd.DataFrame({
            'ETF': ['CASH'],
            'Weight': [cash_weight],
            'FinalScore': [np.nan],
            'ML_Probability': [np.nan],
            'TechScore': [np.nan]
        })
        
        portfolio_summary = pd.concat([portfolio_summary, cash_row], axis=0, ignore_index=True)
        
        return portfolio_summary


def generate_allocation_signals(monthly_scores, top_n=3):
    """
    Generate allocation signals for backtesting
    
    Args:
        monthly_scores: DataFrame with FinalScore
        top_n: Number of ETFs to select
        
    Returns:
        DataFrame with allocation decisions
    """
    strategy = SectorRotationStrategy(top_n=top_n)
    allocations = strategy.generate_monthly_allocations(monthly_scores)
    
    return allocations


if __name__ == '__main__':
    from data.fetch_data import DataFetcher
    from features.indicators import add_all_indicators
    from features.monthly_features import MonthlyFeatureEngine
    from models.train_model import MLModelTrainer
    from models.predict import generate_predictions
    from strategies.technical_score import add_technical_scores
    
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
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = generate_predictions(monthly_with_targets, models_dict, trainer.feature_cols)
    
    # Add technical scores
    print("\nCalculating technical scores...")
    final_data = add_technical_scores(predictions)
    
    # Generate allocations
    print("\nGenerating allocations...")
    allocations = generate_allocation_signals(final_data, top_n=3)
    
    print(f"\nTotal allocation decisions: {len(allocations)}")
    print("\nSample allocations:")
    print(allocations[['ETF', 'Weight', 'FinalScore', 'Selected_Date']].tail(15))
    
    # Show monthly portfolio composition
    unique_dates = allocations['Selected_Date'].unique()[-5:]
    print("\nRecent portfolio compositions:")
    
    strategy = SectorRotationStrategy(top_n=3)
    for date in unique_dates:
        print(f"\n{date}:")
        composition = strategy.get_portfolio_composition(allocations, date)
        print(composition)
