"""
Data fetching module for Indian ETFs
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataFetcher:
    """Fetch and cache OHLCV data for Indian ETFs"""
    
    def __init__(self, data_dir='data/cache'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Indian ETF universe
        self.tickers = {
            'NIFTYBEES': 'NIFTYBEES.NS',
            'BANKBEES': 'BANKBEES.NS',
            'ITBEES': 'ITBEES.NS',
            'PHARMABEES': 'PHARMABEES.NS',
            'PSUBANKBEES': 'PSUBANKBEES.NS',
            'FMCGBEES': 'FMCGBEES.NS',
            'METALBEES': 'METALBEES.NS',
            'CONSUMPTIONBEES': 'KONSUMBEES.NS',  # Alternative ticker
            'MIDCAPBEES': 'MIDCAPBEES.NS',
            'SMALLCAPBEES': 'SMALLCAPBEES.NS'
        }
        
    def fetch_data(self, start_date='2013-01-01', end_date=None, force_refresh=False):
        """
        Fetch data for all ETFs
        
        Args:
            start_date: Start date for data
            end_date: End date for data (default: today)
            force_refresh: Force re-download even if cached
            
        Returns:
            dict: Dictionary of dataframes keyed by ticker name
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching data from {start_date} to {end_date}")
        
        data_dict = {}
        
        for name, ticker in self.tickers.items():
            cache_file = os.path.join(self.data_dir, f"{name}.csv")
            
            # Check cache
            if os.path.exists(cache_file) and not force_refresh:
                print(f"Loading {name} from cache...")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
                # Check if we need to update
                last_date = df.index[-1]
                days_old = (datetime.now() - last_date).days
                
                if days_old <= 1:
                    data_dict[name] = df
                    continue
                    
            # Download data
            print(f"Downloading {name} ({ticker})...")
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    print(f"  WARNING: No data for {name}")
                    continue
                    
                # Clean data
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df.dropna()
                
                if len(df) < 252:  # Less than 1 year
                    print(f"  WARNING: Insufficient data for {name} ({len(df)} days)")
                    continue
                
                # Cache it
                df.to_csv(cache_file)
                data_dict[name] = df
                print(f"  Success: {len(df)} days")
                
            except Exception as e:
                print(f"  ERROR downloading {name}: {e}")
                continue
        
        print(f"\nSuccessfully loaded {len(data_dict)} ETFs")
        return data_dict
    
    def get_aligned_data(self, data_dict):
        """
        Align all dataframes to common dates
        
        Args:
            data_dict: Dictionary of dataframes
            
        Returns:
            dict: Dictionary of aligned dataframes
        """
        if not data_dict:
            return {}
            
        # Find common date range
        all_dates = None
        for name, df in data_dict.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        all_dates = sorted(list(all_dates))
        print(f"Common date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
        
        # Align all dataframes
        aligned_dict = {}
        for name, df in data_dict.items():
            aligned_df = df.loc[all_dates].copy()
            aligned_dict[name] = aligned_df
            
        return aligned_dict


if __name__ == '__main__':
    fetcher = DataFetcher()
    data = fetcher.fetch_data()
    aligned_data = fetcher.get_aligned_data(data)
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    for name, df in aligned_data.items():
        print(f"{name:20s}: {len(df):5d} days, {df.index[0]} to {df.index[-1]}")
