import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import joblib
import os
from datetime import datetime, timedelta
from data import BitcoinDataFetcher
import warnings
warnings.filterwarnings('ignore')

class BitcoinScalpingModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.data_fetcher = BitcoinDataFetcher()
        self.model_path = "xgb_model.pkl"
        self.scaler_path = "scaler.pkl"
        self.last_retrain = None
        self.accuracy_history = []
        
        # Initialize label encoder with correct classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY'])
    
    def prepare_training_data(self) -> tuple:
        """Prepare training data with class balancing"""
        print("📊 Preparing training data...")
        
        df = self.data_fetcher.get_training_data()
        
        if df.empty or len(df) < 100:
            print("⚠️ Using synthetic data for training")
            df = self.data_fetcher.create_synthetic_data()
        
        # Get features
        features_df = self.data_fetcher.get_features(df)
        
        # Remove NaN rows
        valid_data = df[['target']].join(features_df).dropna()
        
        if valid_data.empty:
            print("❌ No valid training data")
            return None, None, None, None
        
        X = valid_data[features_df.columns]
        y = valid_data['target']
        
        # Balance classes - ensure minimum samples per class
        print(f"📊 Class distribution: {y.value_counts().to_dict()}")
        
        # If any class has too few samples, use oversampling
        if y.value_counts().min() < 10:
            print("🔄 Balancing classes...")
            X, y = self.balance_classes(X, y)
        
        print(f"✅ Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Final class distribution: {y.value_counts().to_dict()}")
        
        return X, y, features_df.columns.tolist(), valid_data
    
    def balance_classes(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Balance classes using oversampling"""
        from sklearn.utils import resample
        
        # Combine X and y
        data = pd.concat([X, y], axis=1)
        
        # Separate classes
        classes = {}
        for class_val in [0, 1, 2, 3, 4]:
            classes[class_val] = data[data['target'] == class_val]
        
        # Find max class size
        max_size = max(len(df) for df in classes.values())
        
        # Oversample minority classes
        oversampled_dfs = []
        for class_val, class_df in classes.items():
            if len(class_df) < max_size:
                # Oversample to match max class size
                oversampled = resample(class_df, 
                                    replace=True, 
                                    n_samples=max_size, 
                                    random_state=42)
                oversampled_dfs.append(oversampled)
            else:
                oversampled_dfs.append(class_df)
        
        # Combine all classes
        balanced_data = pd.concat(oversampled_dfs)
        balanced_data = balanced_data.sample(frac=1, random_state=42)  # Shuffle
        
        X_balanced = balanced_data.drop('target', axis=1)
        y_balanced = balanced_data['target']
        
        return X_balanced, y_balanced
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train the XGBoost model with class weights"""
        try:
            print("🚀 Training model...")
            
            # Ensure y has correct classes
            unique_classes = sorted(y.unique())
            print(f"📊 Classes in data: {unique_classes}")
            
            # Calculate class weights for imbalanced data
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.array([0, 1, 2, 3, 4]),
                y=y
            )
            
            # Convert to dictionary for XGBoost
            weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            sample_weights = np.array([weight_dict[val] for val in y])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Also split sample weights
            sample_weights_train, sample_weights_test = train_test_split(
                sample_weights, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost with class weights
            self.model = xgb.XGBClassifier(
                n_estimators=150,  # More trees for better learning
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                random_state=42,
                eval_metric='mlogloss',
                scale_pos_weight=1,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8
            )
            
            self.model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=10,  # Show training progress
                early_stopping_rounds=20
            )
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.accuracy_history.append(accuracy)
            self.last_retrain = datetime.now()
            
            print(f"✅ Model trained - Accuracy: {accuracy:.4f}")
            
            # Show detailed classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            print("📊 Classification Report:")
            for class_name in ['0', '1', '2', '3', '4']:
                if class_name in report:
                    print(f"   Class {class_name}: Precision={report[class_name]['precision']:.3f}, "
                          f"Recall={report[class_name]['recall']:.3f}")
            
            # Save model
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_model(self) -> bool:
        """Load trained model"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Model loaded successfully")
                return True
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, features_df: pd.DataFrame) -> dict:
        """Make prediction with confidence optimization"""
        if self.model is None:
            if not self.load_model():
                return self.get_intelligent_fallback_prediction()
        
        try:
            latest_features = features_df.iloc[-1:].copy()
            scaled_features = self.scaler.transform(latest_features)
            
            prediction_proba = self.model.predict_proba(scaled_features)[0]
            prediction_class = self.model.predict(scaled_features)[0]
            
            # Map to labels
            prediction_label = self.label_encoder.inverse_transform([prediction_class])[0]
            confidence = float(prediction_proba.max())
            
            # Optimize confidence based on prediction strength
            optimized_confidence = self.optimize_confidence(prediction_proba, confidence)
            
            # Check for high confidence breakout
            is_breakout = prediction_class in [0, 4]  # STRONG_SELL or STRONG_BUY
            is_high_confidence = optimized_confidence > 0.80
            
            alert_message = ""
            if is_breakout and is_high_confidence:
                direction = "UP" if prediction_class == 4 else "DOWN"
                alert_message = f"🚨 BREAKOUT: {direction} in 5min ({optimized_confidence*100:.1f}% confidence)"
            
            return {
                'prediction': prediction_label,
                'confidence': optimized_confidence,
                'is_breakout_signal': is_breakout,
                'is_high_confidence': is_high_confidence,
                'alert_message': alert_message,
                'timestamp': datetime.now().isoformat(),
                'model_status': 'trained'
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.get_intelligent_fallback_prediction()
    
    def optimize_confidence(self, probabilities: np.ndarray, raw_confidence: float) -> float:
        """Optimize confidence score"""
        # Boost confidence for clear predictions
        if raw_confidence > 0.7:
            # If one class dominates strongly, boost confidence
            sorted_probs = sorted(probabilities, reverse=True)
            if len(sorted_probs) > 1 and sorted_probs[0] - sorted_probs[1] > 0.3:
                return min(raw_confidence * 1.2, 0.95)
        
        return raw_confidence
    
    def get_intelligent_fallback_prediction(self) -> dict:
        """Intelligent fallback based on technical analysis"""
        try:
            # Get current market data
            df = self.data_fetcher.get_live_data_with_features()
            if not df.empty:
                # Simple technical analysis based prediction
                current_rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
                current_price = df['close'].iloc[-1]
                ema_8 = df['ema_8'].iloc[-1] if 'ema_8' in df.columns else current_price
                volume_spike = df['volume_spike_3'].iloc[-1] if 'volume_spike_3' in df.columns else 1.0
                
                # Simple trading logic
                if current_rsi < 30 and current_price > ema_8 and volume_spike > 1.5:
                    prediction = "STRONG_BUY"
                    confidence = 0.75
                elif current_rsi > 70 and current_price < ema_8 and volume_spike > 1.5:
                    prediction = "STRONG_SELL"
                    confidence = 0.75
                elif current_price > ema_8:
                    prediction = "BUY"
                    confidence = 0.65
                elif current_price < ema_8:
                    prediction = "SELL"
                    confidence = 0.65
                else:
                    prediction = "HOLD"
                    confidence = 0.5
                    
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'is_breakout_signal': prediction in ['STRONG_BUY', 'STRONG_SELL'],
                    'is_high_confidence': confidence > 0.7,
                    'alert_message': "",
                    'timestamp': datetime.now().isoformat(),
                    'model_status': 'fallback_analysis'
                }
            else:
                return self.get_basic_fallback_prediction()
                
        except Exception as e:
            print(f"❌ Intelligent fallback failed: {e}")
            return self.get_basic_fallback_prediction()
    
    def get_basic_fallback_prediction(self) -> dict:
        """Basic fallback prediction"""
        return {
            'prediction': 'HOLD',
            'confidence': 0.5,
            'is_breakout_signal': False,
            'is_high_confidence': False,
            'alert_message': '',
            'timestamp': datetime.now().isoformat(),
            'model_status': 'basic_fallback'
        }
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        if self.last_retrain is None:
            return True
        
        time_since_retrain = datetime.now() - self.last_retrain
        return time_since_retrain > timedelta(hours=6)
    
    def auto_retrain(self):
        """Auto-retrain if needed"""
        if self.should_retrain():
            print("🔄 Auto-retraining model...")
            X, y, features, data = self.prepare_training_data()
            
            if X is not None and len(X) > 100:
                success = self.train_model(X, y)
                if success:
                    print("✅ Model retrained successfully")
                else:
                    print("❌ Retraining failed - using existing model")