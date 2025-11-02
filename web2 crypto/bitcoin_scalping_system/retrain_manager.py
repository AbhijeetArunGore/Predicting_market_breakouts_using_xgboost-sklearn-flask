import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
import sqlite3
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import Config
from data_fetcher import CryptoDataFetcher
from enhanced_model import EnhancedBreakoutModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, 'retrain.log')),
        logging.StreamHandler()
    ]
)

class RetrainManager:
    def __init__(self):
        self.config = Config
        self.data_fetcher = CryptoDataFetcher()
        self.model = EnhancedBreakoutModel()
        self.registry_file = "model_registry.json"
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize model registry if it doesn't exist"""
        if not os.path.exists(self.registry_file):
            registry = {
                "current_version": None,
                "models": {},
                "retrain_history": []
            }
            self._save_registry(registry)
    
    def _load_registry(self):
        """Load model registry"""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except:
            return {"current_version": None, "models": {}, "retrain_history": []}
    
    def _save_registry(self, registry):
        """Save model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def should_retrain(self):
        """Check if retraining is needed based on conditions"""
        registry = self._load_registry()
        
        # Check if no model exists
        if registry["current_version"] is None:
            return True, "No model exists"
        
        # Check time-based retraining
        last_retrain = None
        for history in registry["retrain_history"]:
            if history["version"] == registry["current_version"]:
                last_retrain = datetime.fromisoformat(history["timestamp"])
                break
        
        if last_retrain and datetime.now() - last_retrain < timedelta(hours=Config.RETRAIN_INTERVAL_HOURS):
            return False, f"Too soon since last retrain: {last_retrain}"
        
        # Check data drift (simplified)
        new_data = self._fetch_recent_data()
        if new_data is None or len(new_data) < Config.MIN_SAMPLES_FOR_RETRAIN:
            return False, "Insufficient new data"
        
        return True, "Conditions met for retraining"
    
    def auto_retrain(self):
        """Automatic retraining pipeline"""
        should_retrain, reason = self.should_retrain()
        
        if not should_retrain:
            logging.info(f"Skipping retraining: {reason}")
            return None
        
        logging.info("Starting automatic retraining...")
        
        # Fetch fresh data
        data = self._fetch_training_data()
        if data is None:
            logging.error("Failed to fetch training data")
            return None
        
        # Load current model for comparison
        current_model_path = os.path.join(Config.MODEL_DIR, "current_model.pkl")
        current_metrics = None
        
        if os.path.exists(current_model_path):
            self.model.load_model(current_model_path)
            # Evaluate current model on new data
            current_metrics = self._evaluate_current_model(data)
        
        # Train new model
        self.model.initialize_model()
        new_metrics = self.model.train_model(data, save_model=False)
        
        if new_metrics is None:
            logging.error("Model training failed")
            return None
        
        # Compare performance
        should_promote = self._should_promote_new_model(current_metrics, new_metrics)
        
        if should_promote:
            # Save and promote new model
            version = self.model._get_next_version()
            model_path = os.path.join(Config.MODEL_DIR, f"breakout_model_{version}.pkl")
            self.model.save_model(model_path)
            
            # Update current model
            current_model_path = os.path.join(Config.MODEL_DIR, "current_model.pkl")
            self.model.save_model(current_model_path)
            
            # Update registry
            self._update_registry(version, new_metrics, "production")
            
            logging.info(f"New model promoted to production: {version}")
            logging.info(f"Performance: {new_metrics}")
            
            return {
                "version": version,
                "metrics": new_metrics,
                "status": "promoted"
            }
        else:
            logging.info("New model not better than current. Keeping existing model.")
            return {
                "version": "none",
                "metrics": new_metrics,
                "status": "rejected"
            }
    
    def manual_retrain(self):
        """Manual retraining trigger"""
        logging.info("Starting manual retraining...")
        return self.auto_retrain()
    
    def _fetch_training_data(self):
        """Fetch comprehensive training data"""
        all_data = []
        
        for symbol in Config.SYMBOLS:
            data = self.data_fetcher.fetch_historical_data(symbol, days=90)
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            return None
        
        # Combine data from all symbols
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    
    def _fetch_recent_data(self):
        """Fetch recent data for drift detection"""
        recent_data = []
        
        for symbol in Config.SYMBOLS:
            data, _ = self.data_fetcher.fetch_live_data(symbol)
            if data is not None:
                recent_data.append(data)
        
        if not recent_data:
            return None
        
        return pd.concat(recent_data, ignore_index=True)
    
    def _evaluate_current_model(self, data):
        """Evaluate current model on new data"""
        if self.model.model is None:
            return None
        
        X_train, X_test, y_train, y_test = self.model.prepare_data(data)
        if X_test is None:
            return None
        
        return self.model.evaluate_model(X_test, y_test)
    
    def _should_promote_new_model(self, current_metrics, new_metrics):
        """Decide if new model should replace current model"""
        if current_metrics is None:
            return True
        
        # Use F1 score as primary metric for comparison
        current_f1 = current_metrics.get('f1_score', 0)
        new_f1 = new_metrics.get('f1_score', 0)
        
        improvement = new_f1 - current_f1
        
        if improvement >= Config.PERFORMANCE_THRESHOLD:
            return True
        
        # Also consider AUC-ROC if available
        if 'auc_roc' in current_metrics and 'auc_roc' in new_metrics:
            auc_improvement = new_metrics['auc_roc'] - current_metrics['auc_roc']
            if auc_improvement >= Config.PERFORMANCE_THRESHOLD:
                return True
        
        return False
    
    def _update_registry(self, version, metrics, status):
        """Update model registry with new model information"""
        registry = self._load_registry()
        
        # Update models dictionary
        registry["models"][version] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "status": status
        }
        
        # Update current version if promoted to production
        if status == "production":
            registry["current_version"] = version
        
        # Add to retrain history
        registry["retrain_history"].append({
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "status": status
        })
        
        # Keep only last 50 retrain history entries
        registry["retrain_history"] = registry["retrain_history"][-50:]
        
        self._save_registry(registry)
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        registry = self._load_registry()
        
        if registry["current_version"] is None:
            return {"error": "No model deployed"}
        
        current_model = registry["models"].get(registry["current_version"], {})
        return {
            "current_version": registry["current_version"],
            "metrics": current_model.get("metrics", {}),
            "last_retrain": current_model.get("timestamp"),
            "status": current_model.get("status", "unknown")
        }

if __name__ == "__main__":
    Config.setup_directories()
    manager = RetrainManager()
    
    # Test retraining
    result = manager.auto_retrain()
    print("Retraining result:", result)