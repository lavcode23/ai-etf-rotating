"""
Prediction module for generating ML probabilities
"""
import pandas as pd
import numpy as np
import joblib
import os


class MLPredictor:
    """Generate predictions using trained models"""
    
    def __init__(self, models_dict, feature_cols):
        """
        Args:
            models_dict: Dictionary of trained models
            feature_cols: List of feature column names
        """
        self.models_dict = models_dict
        self.feature_cols = feature_cols
    
    def predict_for_month(self, monthly_data, target_month):
        """
        Generate predictions for a specific month
        
        Args:
            monthly_data: DataFrame with monthly features
            target_month: Period object for target month
            
        Returns:
            DataFrame with predictions
        """
        # Find the appropriate model
        model_key = str(target_month)
        
        if model_key not in self.models_dict:
            # Use the most recent model available before target_month
            available_keys = sorted([k for k in self.models_dict.keys() if k <= model_key])
            if not available_keys:
                print(f"Warning: No model available for {target_month}")
                return None
            model_key = available_keys[-1]
        
        model_data = self.models_dict[model_key]
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Get data for prediction
        monthly_data['YearMonth'] = pd.to_datetime(monthly_data.index).to_period('M')
        pred_data = monthly_data[monthly_data['YearMonth'] == target_month].copy()
        
        if len(pred_data) == 0:
            print(f"Warning: No data for {target_month}")
            return None
        
        # Prepare features
        X = pred_data[self.feature_cols]
        
        # Handle missing values
        X_filled = X.fillna(X.mean())
        
        # Scale
        X_scaled = scaler.transform(X_filled)
        
        # Predict
        pred_proba = model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to dataframe
        pred_data['ML_Probability'] = pred_proba
        pred_data['Model_Used'] = model_key
        
        return pred_data
    
    def predict_all_months(self, monthly_data):
        """
        Generate predictions for all available months
        
        Args:
            monthly_data: DataFrame with monthly features
            
        Returns:
            DataFrame with predictions for all months
        """
        monthly_data['YearMonth'] = pd.to_datetime(monthly_data.index).to_period('M')
        unique_months = sorted(monthly_data['YearMonth'].unique())
        
        all_predictions = []
        
        for month in unique_months:
            pred = self.predict_for_month(monthly_data, month)
            if pred is not None:
                all_predictions.append(pred)
        
        if not all_predictions:
            return pd.DataFrame()
        
        combined = pd.concat(all_predictions, axis=0)
        return combined


def generate_predictions(monthly_data, models_dict, feature_cols):
    """
    Generate predictions for all months using trained models
    
    Args:
        monthly_data: DataFrame with monthly features
        models_dict: Dictionary of trained models
        feature_cols: List of feature columns
        
    Returns:
        DataFrame with ML probabilities added
    """
    predictor = MLPredictor(models_dict, feature_cols)
    predictions = predictor.predict_all_months(monthly_data)
    
    return predictions


if __name__ == '__main__':
    from data.fetch_data import DataFetcher
    from features.indicators import add_all_indicators
    from features.monthly_features import MonthlyFeatureEngine
    from models.train_model import MLModelTrainer
    
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
    
    print(f"\nPredictions generated: {len(predictions)} samples")
    print("\nSample predictions:")
    print(predictions[['ETF', 'ML_Probability', 'Model_Used']].tail(20))
