"""
Portfolio tracking and execution for backtesting
"""
import pandas as pd
import numpy as np
from datetime import datetime


class Portfolio:
    """Track portfolio positions, trades, and returns"""
    
    def __init__(self, initial_capital=1000000, transaction_cost=0.001):
        """
        Args:
            initial_capital: Starting capital in rupees
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # {ETF: {'shares': X, 'entry_price': Y, 'entry_date': Z}}
        self.portfolio_value = initial_capital
        
        # History
        self.equity_curve = []
        self.trades = []
        self.daily_values = []
        self.monthly_returns = []
        
    def rebalance(self, date, allocations, prices):
        """
        Rebalance portfolio according to new allocations
        
        Args:
            date: Rebalance date
            allocations: DataFrame with ETF allocations (columns: ETF, Weight)
            prices: Dict of current prices {ETF: price}
        """
        # Close all existing positions
        self._close_all_positions(date, prices)
        
        # Open new positions
        target_etfs = allocations[allocations['ETF'] != 'CASH']['ETF'].tolist()
        
        for _, row in allocations.iterrows():
            etf = row['ETF']
            weight = row['Weight']
            
            if etf == 'CASH':
                continue
            
            if etf not in prices:
                print(f"  Warning: No price for {etf} on {date}")
                continue
            
            # Calculate position size
            position_value = self.portfolio_value * weight
            price = prices[etf]
            shares = int(position_value / price)
            
            if shares == 0:
                continue
            
            # Execute buy
            cost = shares * price * (1 + self.transaction_cost)
            
            if cost > self.cash:
                # Adjust shares to available cash
                shares = int(self.cash / (price * (1 + self.transaction_cost)))
                cost = shares * price * (1 + self.transaction_cost)
            
            if shares > 0:
                self.cash -= cost
                self.positions[etf] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_date': date
                }
                
                # Record trade
                self.trades.append({
                    'Date': date,
                    'ETF': etf,
                    'Action': 'BUY',
                    'Shares': shares,
                    'Price': price,
                    'Value': shares * price,
                    'Cost': cost,
                    'Weight': weight
                })
    
    def _close_all_positions(self, date, prices):
        """Close all positions"""
        for etf, position in list(self.positions.items()):
            if etf not in prices:
                print(f"  Warning: Cannot close {etf}, no price on {date}")
                continue
            
            shares = position['shares']
            entry_price = position['entry_price']
            entry_date = position['entry_date']
            exit_price = prices[etf]
            
            # Calculate P&L
            proceeds = shares * exit_price * (1 - self.transaction_cost)
            cost = shares * entry_price
            pnl = proceeds - cost
            pnl_pct = (exit_price / entry_price - 1) * 100
            
            self.cash += proceeds
            
            # Record trade
            self.trades.append({
                'Date': date,
                'ETF': etf,
                'Action': 'SELL',
                'Shares': shares,
                'Price': exit_price,
                'Value': shares * exit_price,
                'Proceeds': proceeds,
                'Entry_Price': entry_price,
                'Entry_Date': entry_date,
                'PnL': pnl,
                'PnL_Pct': pnl_pct
            })
        
        # Clear positions
        self.positions = {}
    
    def update_value(self, date, prices):
        """
        Update portfolio value based on current prices
        
        Args:
            date: Current date
            prices: Dict of current prices
        """
        positions_value = 0
        
        for etf, position in self.positions.items():
            if etf in prices:
                positions_value += position['shares'] * prices[etf]
            else:
                # Use entry price if current price not available
                positions_value += position['shares'] * position['entry_price']
        
        self.portfolio_value = self.cash + positions_value
        
        # Record
        self.equity_curve.append({
            'Date': date,
            'Portfolio_Value': self.portfolio_value,
            'Cash': self.cash,
            'Positions_Value': positions_value,
            'Return': (self.portfolio_value / self.initial_capital - 1) * 100
        })
    
    def get_equity_curve(self):
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    def get_trades(self):
        """Get trade log as DataFrame"""
        return pd.DataFrame(self.trades)
    
    def calculate_monthly_returns(self):
        """Calculate monthly returns from equity curve"""
        equity_df = self.get_equity_curve()
        
        if equity_df.empty:
            return pd.DataFrame()
        
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        equity_df = equity_df.set_index('Date')
        
        # Resample to month-end
        monthly = equity_df['Portfolio_Value'].resample('M').last()
        
        # Calculate returns
        monthly_returns = monthly.pct_change() * 100
        
        result = pd.DataFrame({
            'Date': monthly.index,
            'Portfolio_Value': monthly.values,
            'Monthly_Return': monthly_returns.values
        })
        
        return result


def simulate_trading(data_dict, allocations_df, initial_capital=1000000):
    """
    Simulate trading based on allocation signals
    
    Args:
        data_dict: Dictionary of price dataframes
        allocations_df: DataFrame with allocation decisions
        initial_capital: Starting capital
        
    Returns:
        Portfolio object with complete trading history
    """
    portfolio = Portfolio(initial_capital=initial_capital)
    
    # Get unique rebalance dates
    rebalance_dates = sorted(allocations_df['Selected_Date'].unique())
    
    print(f"\nSimulating {len(rebalance_dates)} rebalances...")
    
    for i, date in enumerate(rebalance_dates):
        # Get allocations for this date
        date_allocations = allocations_df[allocations_df['Selected_Date'] == date]
        
        # Get prices on rebalance date
        prices = {}
        for etf in data_dict.keys():
            if etf == 'NIFTYBEES':
                continue
            try:
                price = data_dict[etf].loc[date, 'Close']
                prices[etf] = price
            except:
                pass
        
        if not prices:
            print(f"  Skipping {date}: No prices available")
            continue
        
        # Rebalance
        portfolio.rebalance(date, date_allocations, prices)
        portfolio.update_value(date, prices)
        
        if (i + 1) % 12 == 0:
            print(f"  Completed {i+1}/{len(rebalance_dates)} rebalances...")
    
    print(f"Simulation complete: {len(portfolio.trades)} trades executed")
    
    return portfolio


if __name__ == '__main__':
    from data.fetch_data import DataFetcher
    from features.indicators import add_all_indicators
    from features.monthly_features import MonthlyFeatureEngine
    from models.train_model import MLModelTrainer
    from models.predict import generate_predictions
    from strategies.technical_score import add_technical_scores
    from strategies.sector_rotation import generate_allocation_signals
    
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
    print("\nGenerating predictions...")
    predictions = generate_predictions(monthly_with_targets, models_dict, trainer.feature_cols)
    final_data = add_technical_scores(predictions)
    
    # Generate allocations
    print("\nGenerating allocations...")
    allocations = generate_allocation_signals(final_data, top_n=3)
    
    # Simulate trading
    portfolio = simulate_trading(aligned_data, allocations, initial_capital=1000000)
    
    print("\nPortfolio Summary:")
    print(f"Initial Capital: ₹{portfolio.initial_capital:,.0f}")
    print(f"Final Value: ₹{portfolio.portfolio_value:,.0f}")
    print(f"Total Return: {(portfolio.portfolio_value/portfolio.initial_capital - 1)*100:.2f}%")
    print(f"Total Trades: {len(portfolio.trades)}")
