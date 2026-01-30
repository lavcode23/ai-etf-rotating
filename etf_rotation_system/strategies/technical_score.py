"""
Technical momentum scoring system
"""
import pandas as pd
import numpy as np


class TechnicalScorer:
    """Calculate technical momentum scores"""
    
    def __init__(self):
        pass
    
    def calculate_technical_score(self, monthly_data):
        """
        Calculate technical momentum score from indicators
        
        Score is normalized to 0-1 range based on:
        - RSI: Mean reversion + trend
        - MACD Histogram: Momentum
        - ADX: Trend strength
        - Bollinger %B: Position in range
        
        Args:
            monthly_data: DataFrame with technical indicators
            
        Returns:
            DataFrame with TechScore column added
        """
        df = monthly_data.copy()
        
        # Individual component scores (0-1)
        df['RSI_Score'] = self._normalize_rsi(df['RSI_14'])
        df['MACD_Score'] = self._normalize_macd(df['MACD_Hist'])
        df['ADX_Score'] = self._normalize_adx(df['ADX_14'])
        df['BB_Score'] = self._normalize_bb_percentb(df['BB_PercentB'])
        
        # Weighted technical score
        df['TechScore'] = (
            0.30 * df['RSI_Score'] +
            0.30 * df['MACD_Score'] +
            0.25 * df['ADX_Score'] +
            0.15 * df['BB_Score']
        )
        
        # Ensure 0-1 range
        df['TechScore'] = df['TechScore'].clip(0, 1)
        
        return df
    
    def _normalize_rsi(self, rsi):
        """
        Normalize RSI to 0-1 score
        
        Logic:
        - RSI 30-70: Neutral to bullish (score increases)
        - RSI < 30: Oversold (moderate score)
        - RSI > 70: Overbought (penalize)
        """
        score = pd.Series(index=rsi.index, dtype=float)
        
        # Handle NaN
        rsi_filled = rsi.fillna(50)
        
        # Oversold zone (RSI < 30): score = 0.3 + (30-RSI)*0.01
        mask_oversold = rsi_filled < 30
        score[mask_oversold] = 0.3 + (30 - rsi_filled[mask_oversold]) * 0.01
        
        # Neutral to bullish (30 <= RSI <= 70): linear from 0.3 to 0.9
        mask_neutral = (rsi_filled >= 30) & (rsi_filled <= 70)
        score[mask_neutral] = 0.3 + ((rsi_filled[mask_neutral] - 30) / 40) * 0.6
        
        # Overbought (RSI > 70): score decreases
        mask_overbought = rsi_filled > 70
        score[mask_overbought] = 0.9 - (rsi_filled[mask_overbought] - 70) * 0.02
        
        return score.clip(0, 1)
    
    def _normalize_macd(self, macd_hist):
        """
        Normalize MACD histogram to 0-1 score
        
        Logic:
        - Positive MACD: Bullish (score > 0.5)
        - Negative MACD: Bearish (score < 0.5)
        - Use sigmoid-like transformation
        """
        macd_filled = macd_hist.fillna(0)
        
        # Standardize MACD values
        macd_std = (macd_filled - macd_filled.mean()) / (macd_filled.std() + 1e-8)
        
        # Sigmoid transformation
        score = 1 / (1 + np.exp(-macd_std))
        
        return score
    
    def _normalize_adx(self, adx):
        """
        Normalize ADX to 0-1 score
        
        Logic:
        - ADX < 20: Weak trend (low score)
        - ADX 20-40: Developing trend (medium score)
        - ADX > 40: Strong trend (high score)
        """
        adx_filled = adx.fillna(20)
        
        # Piecewise linear
        score = pd.Series(index=adx.index, dtype=float)
        
        # Weak trend
        mask_weak = adx_filled < 20
        score[mask_weak] = adx_filled[mask_weak] / 20 * 0.3
        
        # Developing trend
        mask_medium = (adx_filled >= 20) & (adx_filled <= 40)
        score[mask_medium] = 0.3 + ((adx_filled[mask_medium] - 20) / 20) * 0.4
        
        # Strong trend
        mask_strong = adx_filled > 40
        score[mask_strong] = 0.7 + ((adx_filled[mask_strong] - 40) / 60) * 0.3
        
        return score.clip(0, 1)
    
    def _normalize_bb_percentb(self, bb_percentb):
        """
        Normalize Bollinger %B to 0-1 score
        
        Logic:
        - %B < 0: Below lower band (oversold, moderate score)
        - %B 0-0.5: Lower half (building, increasing score)
        - %B 0.5-1: Upper half (bullish, high score)
        - %B > 1: Above upper band (overbought, penalize)
        """
        bb_filled = bb_percentb.fillna(0.5)
        
        score = pd.Series(index=bb_percentb.index, dtype=float)
        
        # Below lower band
        mask_below = bb_filled < 0
        score[mask_below] = 0.3
        
        # Lower half (0 to 0.5)
        mask_lower = (bb_filled >= 0) & (bb_filled < 0.5)
        score[mask_lower] = 0.3 + (bb_filled[mask_lower] / 0.5) * 0.3
        
        # Upper half (0.5 to 1)
        mask_upper = (bb_filled >= 0.5) & (bb_filled <= 1)
        score[mask_upper] = 0.6 + ((bb_filled[mask_upper] - 0.5) / 0.5) * 0.3
        
        # Above upper band
        mask_above = bb_filled > 1
        score[mask_above] = 0.9 - (bb_filled[mask_above] - 1) * 0.2
        
        return score.clip(0, 1)
    
    def combine_ml_and_technical(self, data_with_predictions):
        """
        Combine ML probability and technical score into final score
        
        FinalScore = 0.6 * ML_Probability + 0.4 * TechScore
        
        Args:
            data_with_predictions: DataFrame with ML_Probability and TechScore
            
        Returns:
            DataFrame with FinalScore column added
        """
        df = data_with_predictions.copy()
        
        # Ensure both scores exist
        if 'ML_Probability' not in df.columns:
            df['ML_Probability'] = 0.5  # Neutral if missing
        
        if 'TechScore' not in df.columns:
            df = self.calculate_technical_score(df)
        
        # Combine scores
        df['FinalScore'] = 0.6 * df['ML_Probability'] + 0.4 * df['TechScore']
        
        # Ensure 0-1 range
        df['FinalScore'] = df['FinalScore'].clip(0, 1)
        
        return df


def add_technical_scores(monthly_data_with_predictions):
    """
    Add technical scores and final combined score
    
    Args:
        monthly_data_with_predictions: DataFrame with ML predictions
        
    Returns:
        DataFrame with technical and final scores
    """
    scorer = TechnicalScorer()
    
    # Calculate technical score
    data_with_tech = scorer.calculate_technical_score(monthly_data_with_predictions)
    
    # Combine with ML
    data_with_final = scorer.combine_ml_and_technical(data_with_tech)
    
    return data_with_final


if __name__ == '__main__':
    from data.fetch_data import DataFetcher
    from features.indicators import add_all_indicators
    from features.monthly_features import MonthlyFeatureEngine
    from models.train_model import MLModelTrainer
    from models.predict import generate_predictions
    
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
    
    print("\nSample scores:")
    print(final_data[['ETF', 'ML_Probability', 'TechScore', 'FinalScore']].tail(20))
    
    print("\nScore statistics:")
    print(final_data[['ML_Probability', 'TechScore', 'FinalScore']].describe())
