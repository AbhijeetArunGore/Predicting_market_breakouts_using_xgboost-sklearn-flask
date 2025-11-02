from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from config import Config
from data_fetcher import CryptoDataFetcher
from enhanced_model import EnhancedBreakoutModel
from retrain_manager import RetrainManager

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize components
Config.setup_directories()
data_fetcher = CryptoDataFetcher()
model_manager = EnhancedBreakoutModel()
retrain_manager = RetrainManager()

# Load current model
current_model_path = os.path.join(Config.MODEL_DIR, "current_model.pkl")
if os.path.exists(current_model_path):
    model_manager.load_model(current_model_path)
    print("Current model loaded successfully")
else:
    print("No current model found. Please train a model first.")

@app.route('/')
def dashboard():
    """Main dashboard"""
    performance = retrain_manager.get_performance_metrics()
    return render_template('dashboard.html', performance=performance)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Get prediction for current market conditions"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        
        # Fetch latest data
        live_data, current_price = data_fetcher.fetch_live_data(symbol)
        if live_data is None:
            return jsonify({'error': f'Could not fetch data for {symbol}'}), 500
        
        # Make prediction
        prediction, probability = model_manager.predict(live_data)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Format response
        response = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'prediction': int(prediction),
            'probability': float(probability[1]) if probability is not None else 0.5,
            'signal': 'BUY' if prediction == 1 else 'HOLD'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Get predictions for multiple symbols"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', Config.SYMBOLS)
        
        predictions = []
        for symbol in symbols:
            live_data, current_price = data_fetcher.fetch_live_data(symbol)
            if live_data is not None:
                prediction, probability = model_manager.predict(live_data)
                
                if prediction is not None:
                    predictions.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'prediction': int(prediction),
                        'probability': float(probability[1]) if probability is not None else 0.5,
                        'signal': 'BUY' if prediction == 1 else 'HOLD'
                    })
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Trigger manual retraining"""
    try:
        result = retrain_manager.manual_retrain()
        
        if result is None:
            return jsonify({'error': 'Retraining failed'}), 500
        
        return jsonify({
            'message': 'Retraining completed',
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get current model performance metrics"""
    try:
        performance = retrain_manager.get_performance_metrics()
        return jsonify(performance)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload the latest production model"""
    try:
        current_model_path = os.path.join(Config.MODEL_DIR, "current_model.pkl")
        
        if not os.path.exists(current_model_path):
            return jsonify({'error': 'No current model found'}), 404
        
        success = model_manager.load_model(current_model_path)
        
        if success:
            return jsonify({'message': 'Model reloaded successfully'})
        else:
            return jsonify({'error': 'Failed to reload model'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        # Check model status
        model_loaded = model_manager.model is not None
        
        # Check data availability
        live_data, _ = data_fetcher.fetch_live_data('BTCUSDT')
        data_available = live_data is not None and len(live_data) > 0
        
        # Get performance metrics
        performance = retrain_manager.get_performance_metrics()
        
        status = {
            'model_loaded': model_loaded,
            'data_available': data_available,
            'system_time': datetime.now().isoformat(),
            'performance': performance
        }
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """Get available trading symbols"""
    return jsonify({'symbols': Config.SYMBOLS})

if __name__ == '__main__':
    print("Starting Bitcoin Scalping System API...")
    print(f"Available symbols: {Config.SYMBOLS}")
    print("API Endpoints:")
    print("  GET  /              - Dashboard")
    print("  GET  /predict       - Get prediction for symbol")
    print("  POST /predict_batch - Get predictions for multiple symbols")
    print("  POST /retrain       - Trigger manual retraining")
    print("  GET  /metrics       - Get performance metrics")
    print("  POST /reload        - Reload latest model")
    print("  GET  /status        - System status")
    print("  GET  /symbols       - Available symbols")
    
    app.run(host='0.0.0.0', port=5000, debug=True)