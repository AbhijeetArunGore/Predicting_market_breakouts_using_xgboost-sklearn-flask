try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False
    # Fallback to sklearn RandomForest if XGBoost is not available
    from sklearn.ensemble import RandomForestClassifier as _RF
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
from config import PREDICTION_HORIZON, MODEL_CONFIG
import warnings
warnings.filterwarnings('ignore')

class BitcoinScalpingModel:
    def __init__(self):
        self.model = None
        self.breakout_model = None
        self.scaler = StandardScaler()
        self.data_fetcher = BitcoinDataFetcher()
        self.model_path = "xgb_model.pkl"
        self.breakout_model_path = "xgb_breakout_model.pkl"
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
            
            # Train model (XGBoost if available, otherwise RandomForest fallback)
            if XGB_AVAILABLE:
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
            else:
                # Use RandomForest as a conservative fallback when XGBoost is not installed
                self.model = _RF(n_estimators=200, random_state=42)
                self.model.fit(X_train_scaled, y_train)
            
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
            # Save feature importances for inspection
            try:
                import pandas as _pd
                if hasattr(self.model, 'feature_importances_'):
                    fi = _pd.DataFrame({
                        'feature': X.columns,
                        'importance': self.model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    fi.to_csv('feature_importances.csv', index=False)
            except Exception:
                pass
            
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
                # Attempt to load breakout model if present
                if os.path.exists(self.breakout_model_path):
                    try:
                        self.breakout_model = joblib.load(self.breakout_model_path)
                        print("✅ Breakout model loaded successfully")
                    except Exception as e:
                        print(f"⚠️ Failed to load breakout model: {e}")
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

            # Check for breakout with a dedicated breakout model if available
            breakout_proba = 0.0
            try:
                if self.breakout_model is not None:
                    breakout_proba = float(self.breakout_model.predict_proba(scaled_features)[0][1])
                else:
                    # Fallback: treat STRONG_* classes as breakout proxy
                    breakout_proba = 1.0 if prediction_class in [0, 4] else 0.0
            except Exception:
                breakout_proba = 1.0 if prediction_class in [0, 4] else 0.0

            # Combine multiclass confidence and breakout probability for a final confidence
            combined_confidence = float(min(1.0, optimized_confidence * 0.7 + breakout_proba * 0.3))

            is_breakout = breakout_proba > 0.5
            is_high_confidence = combined_confidence > 0.80
            
            alert_message = ""
            if is_breakout and is_high_confidence:
                direction = "UP" if prediction_class == 4 else "DOWN"
                alert_message = f"🚨 BREAKOUT: {direction} in 5min ({combined_confidence*100:.1f}% confidence)"
            
            return {
                'prediction': prediction_label,
                'confidence': combined_confidence,
                'raw_confidence': optimized_confidence,
                'breakout_proba': breakout_proba,
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
        # Use configured retrain interval when available
        try:
            interval = MODEL_CONFIG.get('retrain_interval', timedelta(hours=6))
        except Exception:
            interval = timedelta(hours=6)

        return time_since_retrain > interval

    def log_prediction(self, features_row: pd.Series, prediction: dict) -> None:
        """Log each live prediction to CSV for later labeling and retraining"""
        try:
            log_path = 'predictions_log.csv'
            row = features_row.iloc[0].to_dict() if hasattr(features_row, 'iloc') else dict(features_row)
            # Add metadata
            row.update({
                'pred_time': datetime.now().isoformat(),
                'pred_label': prediction.get('prediction'),
                'pred_confidence': float(prediction.get('confidence', 0.0)),
                'breakout_proba': float(prediction.get('breakout_proba', 0.0)),
                'is_labeled': False
            })
            df_row = pd.DataFrame([row])
            if os.path.exists(log_path):
                df_row.to_csv(log_path, mode='a', header=False, index=False)
            else:
                df_row.to_csv(log_path, index=False)
        except Exception as e:
            print(f"⚠️ Failed to log prediction: {e}")

    def process_prediction_logs(self, horizon_minutes: int = None) -> int:
        """Process prediction log and label entries older than horizon; append labeled rows to relabelled_training.csv
        Returns number of newly labeled rows."""
        try:
            if horizon_minutes is None:
                horizon_minutes = int(PREDICTION_HORIZON)

            log_path = 'predictions_log.csv'
            out_path = 'relabelled_training.csv'
            if not os.path.exists(log_path):
                return 0

            df = pd.read_csv(log_path)
            if df.empty:
                return 0

            newly_labeled = 0
            now = datetime.utcnow()
            for idx, row in df[df['is_labeled'] == False].iterrows():
                try:
                    pred_time = datetime.fromisoformat(row['pred_time'])
                except Exception:
                    pred_time = datetime.strptime(row['pred_time'], '%Y-%m-%dT%H:%M:%S')

                target_time = pred_time + timedelta(minutes=horizon_minutes)
                # Only label if target_time has passed
                if target_time > datetime.now():
                    continue

                # Attempt to get future price from recent klines
                recent = self.data_fetcher.fetch_binance_klines(interval='1m', limit=120)
                if recent.empty:
                    continue

                # Find a close price at or after target_time
                recent = self.data_fetcher.calculate_technical_indicators(recent)
                # Convert index to datetime if needed
                recent_idx = recent.index
                future_rows = recent[recent.index >= target_time]
                if future_rows.empty:
                    # Can't label yet
                    continue

                future_price = float(future_rows['close'].iloc[0])
                pred_price = float(row.get('close', np.nan)) if 'close' in row else np.nan
                if np.isnan(pred_price):
                    # Use last available price at pred_time
                    past = recent[recent.index <= pred_time]
                    if past.empty:
                        continue
                    pred_price = float(past['close'].iloc[-1])

                price_change_pct = (future_price - pred_price) / pred_price * 100

                # Simple breakout labeling rules
                is_breakout = abs(price_change_pct) >= 0.8 and float(row.get('volume_spike_3', 1.0)) > 1.5

                # Append labeled row with features and binary label
                labeled_row = row.to_dict()
                labeled_row['breakout_label'] = 1 if is_breakout else 0
                labeled_row['future_price'] = future_price
                labeled_row['pred_price'] = pred_price
                labeled_row['price_change_pct'] = price_change_pct

                # Determine basic trade success for performance tracking
                # If prediction was BUY/STRONG_BUY and price_change_pct > 0 => success
                # If SELL/STRONG_SELL and price_change_pct < 0 => success
                pred_label = str(row.get('pred_label', 'HOLD'))
                success = False
                if pred_label in ['BUY', 'STRONG_BUY'] and price_change_pct > 0:
                    success = True
                elif pred_label in ['SELL', 'STRONG_SELL'] and price_change_pct < 0:
                    success = True

                labeled_row['success'] = int(success)

                # Write to relabelled training file
                df_out = pd.DataFrame([labeled_row])
                if os.path.exists(out_path):
                    df_out.to_csv(out_path, mode='a', header=False, index=False)
                else:
                    df_out.to_csv(out_path, index=False)

                # Also append to trade performance log
                perf_path = 'trade_performance.csv'
                perf_cols = {
                    'pred_time': labeled_row.get('pred_time'),
                    'pred_label': labeled_row.get('pred_label'),
                    'pred_confidence': labeled_row.get('pred_confidence'),
                    'breakout_proba': labeled_row.get('breakout_proba'),
                    'pred_price': labeled_row.get('pred_price'),
                    'future_price': labeled_row.get('future_price'),
                    'price_change_pct': labeled_row.get('price_change_pct'),
                    'success': labeled_row.get('success')
                }
                perf_df = pd.DataFrame([perf_cols])
                if os.path.exists(perf_path):
                    perf_df.to_csv(perf_path, mode='a', header=False, index=False)
                else:
                    perf_df.to_csv(perf_path, index=False)

                # Mark as labeled in df
                df.at[idx, 'is_labeled'] = True
                newly_labeled += 1

            # Save back the updated log
            df.to_csv(log_path, index=False)
            return newly_labeled
        except Exception as e:
            print(f"⚠️ Failed during log processing: {e}")
            return 0

    def trigger_retrain_from_relabels(self, min_new: int = 100) -> bool:
        """Retrain breakout model if enough relabeled samples are available"""
        try:
            out_path = 'relabelled_training.csv'
            if not os.path.exists(out_path):
                return False
            df = pd.read_csv(out_path)
            # Count total labeled samples
            if df.empty or len(df) < min_new:
                return False

            # Use available feature columns (drop metadata)
            drop_cols = ['pred_time', 'pred_label', 'pred_confidence', 'breakout_proba', 'is_labeled', 'breakout_label']
            feature_cols = [c for c in df.columns if c not in drop_cols]
            Xb = df[feature_cols].fillna(method='ffill').fillna(0)
            yb = df['breakout_label']

            # Train breakout model on relabeled data
            trained = self.train_breakout_model(Xb, yb)
            return trained
        except Exception as e:
            print(f"⚠️ Failed to trigger retrain from relabels: {e}")
            return False
    
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
                # Also train breakout binary classifier
                try:
                    Xb, yb = self.prepare_breakout_training_data(X, y)
                    if Xb is not None and len(Xb) > 100:
                        self.train_breakout_model(Xb, yb)
                except Exception as e:
                    print(f"⚠️ Breakout retrain skipped: {e}")

    def prepare_breakout_training_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Prepare binary breakout target: 1 for strong breakout (STRONG_BUY/STRONG_SELL), else 0"""
        try:
            y_binary = y.copy()
            # Strong breakout classes are mapped to 4 (STRONG_BUY) and 0 (STRONG_SELL)
            y_binary = y_binary.apply(lambda v: 1 if v in [0, 4] else 0)
            return X, y_binary
        except Exception as e:
            print(f"❌ Failed to prepare breakout training data: {e}")
            return None, None

    def train_breakout_model(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train a separate binary XGBoost classifier for breakout detection"""
        try:
            print("🚀 Training breakout binary model...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss'
            )
            # Use XGBoost if available, otherwise sklearn RandomForest fallback
            if XGB_AVAILABLE:
                clf = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss'
                )
                clf.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False, early_stopping_rounds=10)
            else:
                clf = _RF(n_estimators=150, random_state=42)
                clf.fit(X_train_scaled, y_train)

            # Save breakout model
            joblib.dump(clf, self.breakout_model_path)
            self.breakout_model = clf
            print("✅ Breakout model trained and saved")
            return True
        except Exception as e:
            print(f"❌ Breakout training failed: {e}")
            return False