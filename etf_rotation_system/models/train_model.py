"""
Machine Learning model training with walk-forward validation
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os
from datetime import datetime


class MLModelTrainer:
    """Train XGBoost model with walk-forward validation"""
    
    def __init__(self, monthly_data, model_dir='models/saved_models'):
        """
        Args:
            monthly_data: DataFrame with monthly features and targets
            model_dir: Directory to save trained models
        """
        self.monthly_data = monthly_data.copy()
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Feature columns (excluding metadata and target)
        self.feature_cols = [
            'MonthlyReturn', 'Momentum_1M', 'Momentum_3M', 'Momentum_6M', 'Momentum_12M',
            'Volatility_1M', 'Volatility_3M', 'SharpeRatio_1M', 'SharpeRatio_3M',
            'RSI_14', 'MACD_Hist', 'ADX_14', 'BB_PercentB', 'ATR_Percent',
            'Volume_Ratio', 'VolumeSpike', 'ADX_Pos', 'ADX_Neg', 'RelativeStrength'
        ]
        
        # Only keep features that exist in the data
        self.feature_cols = [col for col in self.feature_cols if col in self.monthly_data.columns]
        
        self.target_col = 'Target'
        
    def prepare_data(self):
        """Prepare data for training"""
        df = self.monthly_data.copy()
        
        # Remove rows with missing features or target
        df = df.dropna(subset=self.feature_cols + [self.target_col])
        
        print(f"Data prepared: {len(df)} samples")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Features: {len(self.feature_cols)}")
        
        return df
    
    def walk_forward_train(self, train_window_months=36, retrain_frequency=1):
        """
        Walk-forward training with rolling window
        
        Args:
            train_window_months: Number of months to use for training
            retrain_frequency: Retrain every N months
            
        Returns:
            dict: Dictionary of trained models and metadata
        """
        df = self.prepare_data()
        
        # Get unique months
        df['YearMonth'] = pd.to_datetime(df.index).to_period('M')
        unique_months = sorted(df['YearMonth'].unique())
        
        print(f"\nTotal months: {len(unique_months)}")
        print(f"Training window: {train_window_months} months")
        print(f"Retrain frequency: {retrain_frequency} month(s)")
        
        models_dict = {}
        training_history = []
        
        # Start training after we have enough data
        start_idx = train_window_months
        
        for i in range(start_idx, len(unique_months), retrain_frequency):
            train_end_month = unique_months[i - 1]
            train_start_month = unique_months[i - train_window_months]
            test_month = unique_months[i]
            
            print(f"\n{'='*60}")
            print(f"Training: {train_start_month} to {train_end_month}")
            print(f"Testing: {test_month}")
            
            # Prepare train and test sets
            train_data = df[df['YearMonth'].between(train_start_month, train_end_month)]
            test_data = df[df['YearMonth'] == test_month]
            
            if len(train_data) < 50 or len(test_data) == 0:
                print(f"  Insufficient data, skipping...")
                continue
            
            X_train = train_data[self.feature_cols]
            y_train = train_data[self.target_col]
            X_test = test_data[self.feature_cols]
            y_test = test_data[self.target_col]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost model
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            model.fit(X_train_scaled, y_train, verbose=False)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = np.nan
            
            print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            # Store model and metadata
            model_key = str(test_month)
            models_dict[model_key] = {
                'model': model,
                'scaler': scaler,
                'train_start': train_start_month,
                'train_end': train_end_month,
                'test_month': test_month,
                'accuracy': accuracy,
                'auc': auc,
                'feature_importance': dict(zip(self.feature_cols, model.feature_importances_))
            }
            
            # Training history
            training_history.append({
                'test_month': test_month,
                'accuracy': accuracy,
                'auc': auc,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
        
        print(f"\n{'='*60}")
        print(f"Training completed: {len(models_dict)} models trained")
        
        # Save models
        self.save_models(models_dict)
        
        # Save training history
        history_df = pd.DataFrame(training_history)
        history_path = os.path.join(self.model_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}")
        
        return models_dict, history_df
    
    def save_models(self, models_dict):
        """Save trained models to disk"""
        metadata_list = []
        
        for model_key, model_data in models_dict.items():
            # Save model
            model_path = os.path.join(self.model_dir, f"model_{model_key}.joblib")
            joblib.dump({
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'feature_cols': self.feature_cols
            }, model_path)
            
            # Collect metadata
            metadata_list.append({
                'model_key': model_key,
                'train_start': str(model_data['train_start']),
                'train_end': str(model_data['train_end']),
                'test_month': str(model_data['test_month']),
                'accuracy': model_data['accuracy'],
                'auc': model_data['auc']
            })
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata_list)
        metadata_path = os.path.join(self.model_dir, 'models_metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"Saved {len(models_dict)} models to {self.model_dir}")
    
    def load_model(self, model_key):
        """Load a trained model"""
        model_path = os.path.join(self.model_dir, f"model_{model_key}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_data = joblib.load(model_path)
        return model_data
    
    def get_feature_importance(self, models_dict):
        """
        Get average feature importance across all models
        
        Args:
            models_dict: Dictionary of trained models
            
        Returns:
            DataFrame with feature importance
        """
        all_importance = []
        
        for model_data in models_dict.values():
            importance = model_data['feature_importance']
            all_importance.append(importance)
        
        # Average importance
        importance_df = pd.DataFrame(all_importance)
        avg_importance = importance_df.mean().sort_values(ascending=False)
        
        result_df = pd.DataFrame({
            'Feature': avg_importance.index,
            'Importance': avg_importance.values,
            'Importance_Pct': (avg_importance.values / avg_importance.sum()) * 100
        })
        
        return result_df


if __name__ == '__main__':
    from data.fetch_data import DataFetcher
    from features.indicators import add_all_indicators
    from features.monthly_features import MonthlyFeatureEngine
    
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
    models_dict, history_df = trainer.walk_forward_train(train_window_months=36, retrain_frequency=1)
    
    # Feature importance
    print("\nFeature Importance:")
    importance_df = trainer.get_feature_importance(models_dict)
    print(importance_df)
    
    print("\nTraining History Summary:")
    print(history_df.describe())
