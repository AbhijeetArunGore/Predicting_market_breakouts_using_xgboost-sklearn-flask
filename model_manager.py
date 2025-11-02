import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import SGDClassifier

from model import BitcoinScalpingModel


class ModelManager:
    """Wraps the existing XGBoost-based BitcoinScalpingModel and an online SGDClassifier for incremental updates.

    Responsibilities:
    - Load/save online model checkpoints to /models
    - Provide predict() that ensembles XGBoost + online learner
    - Provide partial_fit updates from newly labeled samples
    - Expose model metadata
    """
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.xgb_wrapper = BitcoinScalpingModel()
        # Try to load existing xgb model using its own loader
        try:
            self.xgb_wrapper.load_model()
        except Exception:
            pass

        self.online = None
        self.online_path = os.path.join(self.models_dir, 'sgd_online.pkl')
        self.version = None
        self._load_online()

        # Supported classes for classification (0..4)
        self.classes_ = np.array([0, 1, 2, 3, 4])

    def _load_online(self):
        try:
            if os.path.exists(self.online_path):
                self.online = joblib.load(self.online_path)
                self.version = getattr(self.online, 'model_version', None)
            else:
                self.online = None
        except Exception:
            self.online = None

    def save_online(self):
        try:
            if self.online is not None:
                # attach version metadata
                setattr(self.online, 'model_version', self._generate_version())
                joblib.dump(self.online, self.online_path)
                self.version = getattr(self.online, 'model_version')
                return True
        except Exception as e:
            print(f"⚠️ Failed to save online model: {e}")
        return False

    def _generate_version(self):
        return datetime.utcnow().strftime('v%Y%m%d_%H%M%S')

    def predict(self, features_df: pd.DataFrame) -> dict:
        """Ensemble prediction: XGBoost probs + online probs (if available)."""
        # Get xgb result
        xgb_res = self.xgb_wrapper.predict(features_df)

        # If no online model, return xgb result with model_version
        if self.online is None:
            out = dict(xgb_res)
            out['model_version'] = self.version or 'xgb_only'
            out['ensemble_source'] = 'xgb'
            return out

        try:
            latest = features_df.iloc[-1:].copy()
            # scale using xgb scaler
            X_scaled = self.xgb_wrapper.scaler.transform(latest)
            online_proba = self.online.predict_proba(X_scaled)[0]
            online_class = int(np.argmax(online_proba))

            # Combine probabilities: weighted average
            xgb_proba = self.xgb_wrapper.model.predict_proba(X_scaled)[0] if self.xgb_wrapper.model is not None else np.zeros_like(online_proba)
            combined = 0.7 * xgb_proba + 0.3 * online_proba
            pred_class = int(np.argmax(combined))
            # Map class label
            pred_label = self.xgb_wrapper.label_encoder.inverse_transform([pred_class])[0]
            confidence = float(np.max(combined))

            return {
                'prediction': pred_label,
                'confidence': confidence,
                'raw_xgb_proba': xgb_proba.tolist() if hasattr(xgb_proba, 'tolist') else [],
                'raw_online_proba': online_proba.tolist() if hasattr(online_proba, 'tolist') else [],
                'combined_proba': combined.tolist(),
                'model_version': self.version or 'hybrid',
                'ensemble_source': 'hybrid'
            }
        except Exception as e:
            print(f"⚠️ Ensemble predict failed: {e}")
            return self.xgb_wrapper.predict(features_df)

    # Convenience pass-throughs to the underlying xgb wrapper for compatibility
    @property
    def model(self):
        return getattr(self.xgb_wrapper, 'model', None)

    def load_model(self) -> bool:
        try:
            return self.xgb_wrapper.load_model()
        except Exception as e:
            print(f"⚠️ load_model passthrough failed: {e}")
            return False

    def auto_retrain(self):
        try:
            return self.xgb_wrapper.auto_retrain()
        except Exception as e:
            print(f"⚠️ auto_retrain passthrough failed: {e}")
            return False

    def log_prediction(self, features_row: pd.Series, prediction: dict) -> None:
        try:
            return self.xgb_wrapper.log_prediction(features_row, prediction)
        except Exception as e:
            print(f"⚠️ log_prediction passthrough failed: {e}")
            return None

    def process_prediction_logs(self, horizon_minutes: int = None) -> int:
        try:
            return self.xgb_wrapper.process_prediction_logs(horizon_minutes)
        except Exception as e:
            print(f"⚠️ process_prediction_logs passthrough failed: {e}")
            return 0

    def trigger_retrain_from_relabels(self, min_new: int = 100) -> bool:
        try:
            return self.xgb_wrapper.trigger_retrain_from_relabels(min_new)
        except Exception as e:
            print(f"⚠️ trigger_retrain_from_relabels passthrough failed: {e}")
            return False

    def partial_update(self, X: pd.DataFrame, y: pd.Series):
        """Perform partial_fit on online classifier with new labeled samples.

        X must be raw feature columns expected by xgb_wrapper.scaler.
        """
        try:
            # Scale features with existing scaler
            X_scaled = self.xgb_wrapper.scaler.transform(X)
        except Exception:
            # If scaler not fitted, fit on X directly
            try:
                self.xgb_wrapper.scaler.fit(X)
                X_scaled = self.xgb_wrapper.scaler.transform(X)
            except Exception as e:
                print(f"⚠️ Scaling failed for online update: {e}")
                return False

        try:
            if self.online is None:
                self.online = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
                # initial partial_fit with classes
                self.online.partial_fit(X_scaled, y, classes=self.classes_)
            else:
                self.online.partial_fit(X_scaled, y)

            self.save_online()
            return True
        except Exception as e:
            print(f"⚠️ Online partial_fit failed: {e}")
            return False

    def get_model_info(self) -> dict:
        info = {
            'model_version': self.version or 'unknown',
            'xgb_last_retrain': getattr(self.xgb_wrapper, 'last_retrain', None),
            'online_loaded': self.online is not None
        }
        return info
