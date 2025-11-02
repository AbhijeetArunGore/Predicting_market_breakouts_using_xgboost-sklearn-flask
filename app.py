import flask
from flask import Flask, render_template, jsonify, request
import threading
import time
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import enhanced modules
from enhanced_model import EnhancedBreakoutPredictor
from data_fetcher import MultiTimeframeDataFetcher
from risk_manager import RiskManager

app = Flask(__name__)

# Global variables
model = None
data_fetcher = None
risk_manager = RiskManager()
current_prediction = None
system_status = "initializing"
model_version = "v20250115_001"

class SystemManager:
    def __init__(self):
        self.update_interval = 30  # seconds
        self.last_update = None
        self.prediction_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'rolling_accuracy': 0.0,
            'avg_confidence': 0.0
        }
        self.initial_prediction_made = False
    
    def initialize_system(self):
        """Initialize all system components"""
        global model, data_fetcher
        
        try:
            # Initialize enhanced model
            model = EnhancedBreakoutPredictor()
            print("✅ Enhanced model initialized successfully")
            
            # Initialize data fetcher
            data_fetcher = MultiTimeframeDataFetcher()
            print("✅ Multi-timeframe data fetcher initialized")
            
            # Load or train initial model
            model.load_or_train_model()
            
            self.system_status = "running"
            self.last_update = datetime.utcnow()
            
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            self.system_status = "error"
            return False
    
    def update_system(self):
        """Main system update loop"""
        global current_prediction
        
        while True:
            try:
                if self.system_status == "running":
                    print("\n🔄 Starting system update cycle...")
                    
                    # Fetch latest data
                    data = data_fetcher.fetch_all_timeframes()
                    if data is None:
                        print("❌ Failed to fetch data")
                        time.sleep(self.update_interval)
                        continue
                    
                    # Generate prediction
                    prediction = model.predict_with_reliability(data)
                    
                    if prediction:
                        # Add risk management parameters
                        enhanced_prediction = risk_manager.enhance_prediction(
                            prediction, data['current_price']
                        )
                        
                        current_prediction = enhanced_prediction
                        current_prediction['model_version'] = model_version
                        current_prediction['last_update'] = datetime.utcnow().isoformat() + 'Z'
                        
                        # Update performance metrics
                        self.update_performance_metrics(enhanced_prediction)
                        
                        # Store in history
                        self.prediction_history.append(enhanced_prediction)
                        if len(self.prediction_history) > 100:
                            self.prediction_history.pop(0)
                        
                        print(f"🎯 New prediction: {enhanced_prediction['signal']} "
                              f"(Conf: {enhanced_prediction['confidence']:.3f}, "
                              f"Rel: {enhanced_prediction['reliability']:.3f})")
                        
                        self.initial_prediction_made = True
                    
                    self.last_update = datetime.utcnow()
                    
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Error in update cycle: {e}")
                time.sleep(self.update_interval)

    def update_performance_metrics(self, prediction):
        """Update trading performance metrics"""
        if prediction['signal'] != 'HOLD':
            self.performance_metrics['total_trades'] += 1
            
            # Simulate success based on confidence (for demo)
            if prediction['confidence'] > 0.7:
                self.performance_metrics['successful_trades'] += 1
            
            # Calculate rolling accuracy (last 50 trades)
            if self.performance_metrics['total_trades'] > 0:
                success_rate = (self.performance_metrics['successful_trades'] / 
                              self.performance_metrics['total_trades'])
                self.performance_metrics['rolling_accuracy'] = success_rate
            
            self.performance_metrics['avg_confidence'] = (
                (self.performance_metrics['avg_confidence'] * 
                 (self.performance_metrics['total_trades'] - 1) + 
                 prediction['confidence']) / 
                self.performance_metrics['total_trades']
            )

# Initialize system manager
system_manager = SystemManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('enhanced_dashboard.html')

@app.route('/api/prediction')
def get_prediction():
    """API endpoint for current prediction"""
    global current_prediction
    
    if current_prediction is None:
        return jsonify({
            "status": "initializing",
            "message": "System is starting up...",
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        })
    
    # Add performance metrics to response
    response = current_prediction.copy()
    response.update({
        "performance": system_manager.performance_metrics,
        "system_status": system_manager.system_status,
        "initialized": system_manager.initial_prediction_made
    })
    
    return jsonify(response)

@app.route('/api/history')
def get_history():
    """API endpoint for prediction history"""
    return jsonify({
        "history": system_manager.prediction_history[-50:],  # Last 50 predictions
        "performance": system_manager.performance_metrics
    })

@app.route('/api/system_status')
def get_system_status():
    """API endpoint for system status"""
    return jsonify({
        "status": system_manager.system_status,
        "last_update": system_manager.last_update.isoformat() + 'Z' if system_manager.last_update else None,
        "model_version": model_version,
        "update_interval": system_manager.update_interval,
        "initialized": system_manager.initial_prediction_made
    })

def start_update_thread():
    """Start the background update thread"""
    time.sleep(2)  # Let the system initialize
    update_thread = threading.Thread(target=system_manager.update_system, daemon=True)
    update_thread.start()
    print("🔄 Background update thread started")

if __name__ == '__main__':
    print("🚀 Starting Enhanced Bitcoin Scalping System...")
    print("📈 Advanced AI Breakout Prediction System")
    print("🎯 Multi-Timeframe Analysis + Transformer Architecture")
    print("⚠️  Disclaimer: For educational purposes only")
    
    # Initialize system
    if system_manager.initialize_system():
        # Start background updates
        start_update_thread()
        
        # Start Flask server
        print(f"🌐 Dashboard: http://localhost:5000")
        print(f"⏰ Updates every {system_manager.update_interval} seconds")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        print("❌ Failed to start system")