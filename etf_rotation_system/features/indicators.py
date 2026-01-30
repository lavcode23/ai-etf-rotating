"""
Technical indicators calculation using TA library
"""
import pandas as pd
import numpy as np
import ta


class TechnicalIndicators:
    """Calculate technical indicators for a dataframe"""
    
    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate all technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        df = df.copy()
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_PercentB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR
        df['ATR_14'] = ta.volatility.AverageTrueRange(
            df['High'], df['Low'], df['Close'], window=14
        ).average_true_range()
        df['ATR_Percent'] = (df['ATR_14'] / df['Close']) * 100
        
        # ADX
        adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX_14'] = adx_indicator.adx()
        df['ADX_Pos'] = adx_indicator.adx_pos()
        df['ADX_Neg'] = adx_indicator.adx_neg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'], window=14, smooth_window=3
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # OBV
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        # Momentum
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        df['ROC_10'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
        
        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        df['Volatility_60'] = df['Returns'].rolling(window=60).std() * np.sqrt(252)
        
        return df
    
    @staticmethod
    def calculate_momentum_scores(df, lookback_periods=[21, 63, 126]):
        """
        Calculate momentum scores for different periods
        
        Args:
            df: DataFrame with price data
            lookback_periods: List of lookback periods in days
            
        Returns:
            DataFrame with momentum scores
        """
        df = df.copy()
        
        for period in lookback_periods:
            col_name = f'Momentum_{period}d'
            df[col_name] = (df['Close'] / df['Close'].shift(period) - 1) * 100
            
        return df
    
    @staticmethod
    def calculate_volatility_metrics(df, window=20):
        """
        Calculate volatility metrics
        
        Args:
            df: DataFrame with returns
            window: Rolling window size
            
        Returns:
            DataFrame with volatility metrics
        """
        df = df.copy()
        
        if 'Returns' not in df.columns:
            df['Returns'] = df['Close'].pct_change()
        
        df['Realized_Vol'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
        df['Downside_Vol'] = df['Returns'].apply(lambda x: x if x < 0 else 0).rolling(window=window).std() * np.sqrt(252)
        
        return df


def add_all_indicators(data_dict):
    """
    Add technical indicators to all ETF dataframes
    
    Args:
        data_dict: Dictionary of dataframes
        
    Returns:
        Dictionary of dataframes with indicators
    """
    result_dict = {}
    
    for name, df in data_dict.items():
        print(f"Calculating indicators for {name}...")
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
        result_dict[name] = df_with_indicators
        
    return result_dict


if __name__ == '__main__':
    # Test
    from data.fetch_data import DataFetcher
    
    fetcher = DataFetcher()
    data = fetcher.fetch_data()
    aligned_data = fetcher.get_aligned_data(data)
    
    data_with_indicators = add_all_indicators(aligned_data)
    
    # Print sample
    for name, df in data_with_indicators.items():
        print(f"\n{name}:")
        print(df[['Close', 'RSI_14', 'MACD_Hist', 'ADX_14', 'BB_PercentB']].tail())
        break
