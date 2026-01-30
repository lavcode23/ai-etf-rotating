"""
Monthly feature engineering from daily data
"""
import pandas as pd
import numpy as np
from datetime import datetime


class MonthlyFeatureEngine:
    """Generate monthly features from daily OHLCV data"""
    
    def __init__(self, data_dict_with_indicators, benchmark_name='NIFTYBEES'):
        """
        Args:
            data_dict_with_indicators: Dict of dataframes with technical indicators
            benchmark_name: Name of benchmark ETF for relative strength
        """
        self.data_dict = data_dict_with_indicators
        self.benchmark_name = benchmark_name
        
    def create_monthly_features(self):
        """
        Create monthly features for all ETFs
        
        Returns:
            DataFrame with monthly features for all ETFs
        """
        monthly_data_list = []
        
        for etf_name, df in self.data_dict.items():
            if etf_name == self.benchmark_name:
                continue  # We'll handle benchmark separately
                
            print(f"Creating monthly features for {etf_name}...")
            monthly_df = self._create_monthly_features_for_etf(df, etf_name)
            monthly_data_list.append(monthly_df)
        
        # Combine all ETFs
        all_monthly = pd.concat(monthly_data_list, axis=0, ignore_index=False)
        all_monthly = all_monthly.sort_index()
        
        # Add benchmark returns for relative strength
        if self.benchmark_name in self.data_dict:
            benchmark_monthly = self._get_benchmark_monthly_returns()
            all_monthly = all_monthly.join(benchmark_monthly, how='left')
            all_monthly['RelativeStrength'] = all_monthly['MonthlyReturn'] - all_monthly['Benchmark_Return']
        
        return all_monthly
    
    def _create_monthly_features_for_etf(self, df, etf_name):
        """
        Create monthly features for a single ETF
        
        Args:
            df: Daily dataframe with indicators
            etf_name: Name of ETF
            
        Returns:
            DataFrame with monthly features
        """
        df = df.copy()
        df['YearMonth'] = df.index.to_period('M')
        
        monthly_features = []
        
        # Group by month
        for period, group in df.groupby('YearMonth'):
            if len(group) < 5:  # Skip months with too few trading days
                continue
            
            # Get month-end date (last trading day of month)
            month_end_date = group.index[-1]
            
            # Month-end values
            close_end = group['Close'].iloc[-1]
            close_start = group['Close'].iloc[0]
            
            # Calculate features at month-end
            features = {
                'Date': month_end_date,
                'ETF': etf_name,
                'Close': close_end,
                
                # Returns
                'MonthlyReturn': (close_end / close_start - 1) * 100,
                
                # Momentum features (looking back from month-end)
                'Momentum_1M': self._get_momentum(df, month_end_date, 21),
                'Momentum_3M': self._get_momentum(df, month_end_date, 63),
                'Momentum_6M': self._get_momentum(df, month_end_date, 126),
                'Momentum_12M': self._get_momentum(df, month_end_date, 252),
                
                # Volatility
                'Volatility_1M': self._get_volatility(df, month_end_date, 21),
                'Volatility_3M': self._get_volatility(df, month_end_date, 63),
                
                # Sharpe-like ratio
                'SharpeRatio_1M': self._get_sharpe_ratio(df, month_end_date, 21),
                'SharpeRatio_3M': self._get_sharpe_ratio(df, month_end_date, 63),
                
                # Technical indicators at month-end
                'RSI_14': group['RSI_14'].iloc[-1] if 'RSI_14' in group.columns else np.nan,
                'MACD_Hist': group['MACD_Hist'].iloc[-1] if 'MACD_Hist' in group.columns else np.nan,
                'ADX_14': group['ADX_14'].iloc[-1] if 'ADX_14' in group.columns else np.nan,
                'BB_PercentB': group['BB_PercentB'].iloc[-1] if 'BB_PercentB' in group.columns else np.nan,
                'ATR_Percent': group['ATR_Percent'].iloc[-1] if 'ATR_Percent' in group.columns else np.nan,
                
                # Volume metrics
                'Volume_Ratio': group['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in group.columns else np.nan,
                'Avg_Volume_1M': group['Volume'].mean(),
                'Avg_Volume_3M': self._get_avg_volume(df, month_end_date, 63),
                
                # Trend strength
                'ADX_Pos': group['ADX_Pos'].iloc[-1] if 'ADX_Pos' in group.columns else np.nan,
                'ADX_Neg': group['ADX_Neg'].iloc[-1] if 'ADX_Neg' in group.columns else np.nan,
            }
            
            # Volume spike
            if features['Avg_Volume_3M'] > 0:
                features['VolumeSpike'] = features['Avg_Volume_1M'] / features['Avg_Volume_3M']
            else:
                features['VolumeSpike'] = 1.0
            
            monthly_features.append(features)
        
        monthly_df = pd.DataFrame(monthly_features)
        monthly_df = monthly_df.set_index('Date')
        
        return monthly_df
    
    def _get_momentum(self, df, date, lookback_days):
        """Calculate momentum looking back from date"""
        try:
            date_loc = df.index.get_loc(date)
            if date_loc < lookback_days:
                return np.nan
            
            current_price = df['Close'].iloc[date_loc]
            past_price = df['Close'].iloc[date_loc - lookback_days]
            
            return (current_price / past_price - 1) * 100
        except:
            return np.nan
    
    def _get_volatility(self, df, date, lookback_days):
        """Calculate volatility looking back from date"""
        try:
            date_loc = df.index.get_loc(date)
            if date_loc < lookback_days:
                return np.nan
            
            returns = df['Returns'].iloc[date_loc - lookback_days:date_loc]
            vol = returns.std() * np.sqrt(252) * 100
            
            return vol
        except:
            return np.nan
    
    def _get_sharpe_ratio(self, df, date, lookback_days):
        """Calculate Sharpe ratio looking back from date"""
        try:
            date_loc = df.index.get_loc(date)
            if date_loc < lookback_days:
                return np.nan
            
            returns = df['Returns'].iloc[date_loc - lookback_days:date_loc]
            mean_return = returns.mean() * 252
            vol = returns.std() * np.sqrt(252)
            
            if vol > 0:
                return mean_return / vol
            else:
                return 0
        except:
            return np.nan
    
    def _get_avg_volume(self, df, date, lookback_days):
        """Calculate average volume looking back from date"""
        try:
            date_loc = df.index.get_loc(date)
            if date_loc < lookback_days:
                return np.nan
            
            volume = df['Volume'].iloc[date_loc - lookback_days:date_loc]
            return volume.mean()
        except:
            return np.nan
    
    def _get_benchmark_monthly_returns(self):
        """Get monthly returns for benchmark"""
        if self.benchmark_name not in self.data_dict:
            return pd.DataFrame()
        
        df = self.data_dict[self.benchmark_name].copy()
        df['YearMonth'] = df.index.to_period('M')
        
        monthly_returns = []
        for period, group in df.groupby('YearMonth'):
            if len(group) < 5:
                continue
            
            month_end_date = group.index[-1]
            close_end = group['Close'].iloc[-1]
            close_start = group['Close'].iloc[0]
            monthly_return = (close_end / close_start - 1) * 100
            
            monthly_returns.append({
                'Date': month_end_date,
                'Benchmark_Return': monthly_return
            })
        
        benchmark_df = pd.DataFrame(monthly_returns)
        benchmark_df = benchmark_df.set_index('Date')
        
        return benchmark_df
    
    def create_targets(self, monthly_df):
        """
        Create target labels for next month performance
        
        Target = 1 if next month return > median of all ETFs
        Target = 0 otherwise
        
        Args:
            monthly_df: DataFrame with monthly features
            
        Returns:
            DataFrame with target column added
        """
        df = monthly_df.copy()
        
        # Sort by date and ETF
        df = df.sort_index()
        
        # Create next month return for each ETF
        df['NextMonth_Return'] = df.groupby('ETF')['MonthlyReturn'].shift(-1)
        
        # For each date, calculate median return across all ETFs
        median_returns = df.groupby(df.index)['NextMonth_Return'].median()
        df['Median_NextMonth_Return'] = df.index.map(median_returns)
        
        # Binary target: 1 if above median, 0 otherwise
        df['Target'] = (df['NextMonth_Return'] > df['Median_NextMonth_Return']).astype(int)
        
        # Drop rows where we don't have next month return (last month for each ETF)
        df = df.dropna(subset=['NextMonth_Return'])
        
        return df


if __name__ == '__main__':
    from data.fetch_data import DataFetcher
    from features.indicators import add_all_indicators
    
    # Fetch data
    fetcher = DataFetcher()
    data = fetcher.fetch_data()
    aligned_data = fetcher.get_aligned_data(data)
    
    # Add indicators
    data_with_indicators = add_all_indicators(aligned_data)
    
    # Create monthly features
    feature_engine = MonthlyFeatureEngine(data_with_indicators)
    monthly_features = feature_engine.create_monthly_features()
    monthly_with_targets = feature_engine.create_targets(monthly_features)
    
    print("\nMonthly Features Shape:", monthly_features.shape)
    print("\nSample Monthly Features:")
    print(monthly_with_targets.head(20))
    
    print("\nFeature Columns:")
    print(monthly_with_targets.columns.tolist())
    
    print("\nTarget Distribution:")
    print(monthly_with_targets['Target'].value_counts())
