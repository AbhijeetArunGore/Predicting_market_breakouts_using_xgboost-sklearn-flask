import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import shutil
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import Config
from feature_engineer import FeatureEngineer
from labeler import BreakoutLabeler

class EnhancedBreakoutModel:
    def __init__(self):
        self.config = Config
        self.feature_engineer = FeatureEngineer()
        self.labeler = BreakoutLabeler()
        self.model = None
        self.feature_columns = None
        self.model_type = "xgboost"  # Options: xgboost, randomforest, sgd
        
    def initialize_model(self, model_type="xgboost"):
        """Initialize the ML model"""
        self.model_type = model_type
        
        if model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=Config.RANDOM_STATE,
                eval_metric='logloss'
            )
        elif model_type == "randomforest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=Config.RANDOM_STATE
            )
        elif model_type == "sgd":
            self.model = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.0001,
                learning_rate='optimal',
                random_state=Config.RANDOM_STATE
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_data(self, df):
        """Prepare features and labels for training"""
        if df is None or len(df) < 100:
            return None, None, None, None
            
        # Engineer features
        features_df = self.feature_engineer.engineer_features(df)
        if features_df is None:
            return None, None, None, None
            
        # Generate labels
        labeled_df = self.labeler.generate_labels(features_df)
        if labeled_df is None or 'label' not in labeled_df.columns:
            return None, None, None, None
            
        # Get feature columns
        self.feature_columns = self.feature_engineer.get_feature_columns()
        available_features = [col for col in self.feature_columns if col in labeled_df.columns]
        
        if len(available_features) == 0:
            return None, None, None, None
            
        X = labeled_df[available_features]
        y = labeled_df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, df, save_model=True):
        """Train the model on historical data"""
        print("Starting model training...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        if X_train is None or len(X_train) == 0:
            print("Insufficient data for training")
            return None
            
        print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features")
        
        # Initialize model if not done
        if self.model is None:
            self.initialize_model()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        if save_model:
            # Save model with versioning
            version = self._get_next_version()
            model_path = os.path.join(Config.MODEL_DIR, f"breakout_model_{version}.pkl")
            self.save_model(model_path)
            
            # Update current model
            current_model_path = os.path.join(Config.MODEL_DIR, "current_model.pkl")
            self.save_model(current_model_path)
            
            print(f"Model trained and saved as version {version}")
            print(f"Performance: {metrics}")
        
        return metrics
    
    def update_model(self, new_data):
        """Update model with new data (incremental learning)"""
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None
            
        if new_data is None or len(new_data) < 10:
            print("Insufficient new data for update")
            return None
            
        # Prepare new data
        X_new, _, y_new, _ = self.prepare_data(new_data)
        
        if X_new is None or len(X_new) == 0:
            print("Could not prepare new data for update")
            return None
            
        # Update model (for SGD classifier)
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_new, y_new)
            print(f"Model updated with {len(X_new)} new samples")
        else:
            # For tree-based models, we need to retrain
            print("Model doesn't support incremental learning. Retraining...")
            combined_data = self._combine_with_historical(new_data)
            return self.train_model(combined_data)
        
        return self.evaluate_model(X_new, y_new)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None or X_test is None or len(X_test) == 0:
            return None
            
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
        
        return metrics
    
    def predict(self, live_data):
        """Make predictions on live data"""
        if self.model is None:
            print("No model loaded")
            return None, None
            
        # Engineer features
        features_df = self.feature_engineer.engineer_features(live_data)
        if features_df is None:
            return None, None
            
        # Get latest features
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        if len(available_features) == 0:
            return None, None
            
        X_latest = features_df[available_features].iloc[-1:].values
        
        # Make prediction
        prediction = self.model.predict(X_latest)[0]
        probability = self.model.predict_proba(X_latest)[0] if hasattr(self.model, 'predict_proba') else None
        
        return prediction, probability
    
    def save_model(self, path):
        """Save model to file"""
        if self.model is None:
            print("No model to save")
            return False
            
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        return True
    
    def load_model(self, path):
        """Load model from file"""
        # Defensive loading: handle empty/corrupt pickle files gracefully
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.feature_columns = model_data.get('feature_columns')
            self.model_type = model_data.get('model_type', 'xgboost')

            print(f"Model loaded from {path}")
            return True

        except (EOFError, pickle.UnpicklingError) as e:
            # Corrupt or empty model file. Move it aside to avoid repeated failures.
            try:
                bad_path = path + ".corrupt." + datetime.now().strftime('%Y%m%d_%H%M%S')
                shutil.move(path, bad_path)
                print(f"Corrupt model detected and moved to {bad_path}")
            except Exception:
                print(f"Corrupt model detected but failed to move file: {path}")
            return False

        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _get_next_version(self):
        """Get next model version number"""
        existing_models = [f for f in os.listdir(Config.MODEL_DIR) if f.startswith('breakout_model_v')]
        if not existing_models:
            return 'v1'
        
        versions = [int(f.split('_v')[1].split('.pkl')[0]) for f in existing_models]
        return f'v{max(versions) + 1}'
    
    def _combine_with_historical(self, new_data):
        """Combine new data with historical data (simplified implementation)"""
        # In a real implementation, you'd load historical data and combine
        return new_data

if __name__ == "__main__":
    # Test model training
    from data_fetcher import CryptoDataFetcher
    
    Config.setup_directories()
    fetcher = CryptoDataFetcher()
    model = EnhancedBreakoutModel()
    
    # Fetch and train on sample data
    data = fetcher.fetch_historical_data("BTCUSDT", days=90)
    if data is not None:
        metrics = model.train_model(data)
        print("Model training completed!")
        print(f"Performance metrics: {metrics}")