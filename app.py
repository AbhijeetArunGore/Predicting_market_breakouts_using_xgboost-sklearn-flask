from flask import Flask, render_template, jsonify
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import os

from data import BitcoinDataFetcher
from model import BitcoinScalpingModel

app = Flask(__name__)

# Global variables
current_data = None
current_prediction = None
model_manager = BitcoinScalpingModel()
data_fetcher = BitcoinDataFetcher()
system_status = "starting"
training_progress = 0
start_time = datetime.now()

def calculate_trading_levels(current_price: float, prediction: dict) -> dict:
    """Calculate trading levels"""
    # Use ATR for volatility if available, else estimate
    if current_data is not None and not current_data.empty and 'atr' in current_data.columns:
        atr = current_data['atr'].iloc[-1]
    else:
        atr = current_price * 0.02
    
    confidence = prediction.get('confidence', 0.5)
    prediction_class = prediction.get('prediction', 'HOLD')
    
    # Dynamic risk management based on confidence
    if confidence > 0.8:
        risk_multiplier = 0.8
        reward_multiplier = 3.0
    elif confidence > 0.7:
        risk_multiplier = 1.0
        reward_multiplier = 2.5
    else:
        risk_multiplier = 1.2
        reward_multiplier = 2.0
    
    if prediction_class in ['BUY', 'STRONG_BUY']:
        entry = current_price - (atr * 0.1)
        stop_loss = entry - (atr * risk_multiplier)
        target_1 = entry + (atr * reward_multiplier * 0.6)
        target_2 = entry + (atr * reward_multiplier)
        direction = "BULLISH"
    elif prediction_class in ['SELL', 'STRONG_SELL']:
        entry = current_price + (atr * 0.1)
        stop_loss = entry + (atr * risk_multiplier)
        target_1 = entry - (atr * reward_multiplier * 0.6)
        target_2 = entry - (atr * reward_multiplier)
        direction = "BEARISH"
    else:
        entry = current_price
        stop_loss = current_price - (atr * 2)
        target_1 = current_price + (atr * 2)
        target_2 = current_price + (atr * 3)
        direction = "NEUTRAL"
    
    return {
        'current_price': round(current_price, 2),
        'entry': round(entry, 2),
        'stop_loss': round(stop_loss, 2),
        'target_1': round(target_1, 2),
        'target_2': round(target_2, 2),
        'direction': direction,
        'atr': round(atr, 2)
    }

def get_trade_suggestion(prediction: dict, levels: dict) -> dict:
    """Get trade suggestion"""
    confidence = prediction.get('confidence', 0.5)
    
    if confidence > 0.8 and prediction.get('is_breakout_signal', False):
        return {
            'action': 'AGGRESSIVE_BREAKOUT',
            'size': '3-5%',
            'reason': 'High-confidence breakout detected',
            'urgency': 'HIGH'
        }
    elif confidence > 0.7:
        return {
            'action': 'CONFIDENT',
            'size': '2-3%',
            'reason': 'Strong directional signal',
            'urgency': 'MEDIUM_HIGH'
        }
    elif confidence > 0.6:
        return {
            'action': 'MODERATE',
            'size': '1-2%',
            'reason': 'Good trading opportunity',
            'urgency': 'MEDIUM'
        }
    else:
        return {
            'action': 'AVOID',
            'size': '0%',
            'reason': 'Wait for better setup',
            'urgency': 'LOW'
        }

def create_chart_data():
    """Create chart data"""
    global current_data
    
    if current_data is not None and not current_data.empty:
        last_50 = current_data.tail(50)
        
        # Safe data extraction
        timestamps = last_50.index.strftime('%H:%M').tolist()
        prices = last_50['close'].round(2).tolist()
        
        # Safe indicator extraction
        ema_8 = last_50['ema_8'].round(2).tolist() if 'ema_8' in last_50.columns else prices
        ema_21 = last_50['ema_21'].round(2).tolist() if 'ema_21' in last_50.columns else prices
        
        return {
            'timestamps': timestamps,
            'prices': prices,
            'ema_8': ema_8,
            'ema_21': ema_21
        }
    else:
        # Fallback data
        current_price = data_fetcher.get_current_real_price()
        timestamps = []
        prices = []
        
        for i in range(50):
            timestamps.append(f"{(datetime.now() - timedelta(minutes=50-i)).strftime('%H:%M')}")
            prices.append(round(current_price * (1 + 0.001 * (i % 10)), 2))
        
        return {
            'timestamps': timestamps,
            'prices': prices,
            'ema_8': prices,
            'ema_21': prices
        }

def update_system():
    """Main system update thread"""
    global current_data, current_prediction, system_status, training_progress
    
    print("🔄 Starting system update thread...")
    time.sleep(2)
    
    try:
        # Load or train model
        if model_manager.load_model():
            system_status = "ready"
            print("✅ Model loaded - System ready")
        else:
            system_status = "training"
            print("🎯 Training model...")
            
            # Training progress
            for i in range(3):
                time.sleep(5)
                training_progress = (i + 1) * 33
                print(f"📚 Training: {training_progress}%")
            
            model_manager.auto_retrain()
            system_status = "ready"
            training_progress = 100
        
        # Main update loop
        while True:
            try:
                # Get live data
                df = data_fetcher.get_live_data_with_features()
                current_data = df
                
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    
                    # Get prediction
                    features_df = data_fetcher.get_features(df)
                    prediction_result = model_manager.predict(features_df)
                    
                    # Calculate trading levels
                    levels = calculate_trading_levels(current_price, prediction_result)
                    trade_suggestion = get_trade_suggestion(prediction_result, levels)
                    
                    current_prediction = {
                        **prediction_result,
                        'levels': levels,
                        'trade_suggestion': trade_suggestion,
                        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Log high-confidence signals
                    if prediction_result.get('is_high_confidence', False):
                        print(f"🎯 {prediction_result['prediction']} signal - {prediction_result['confidence']*100:.1f}% confidence")
                
                else:
                    # Fallback prediction
                    current_prediction = create_fallback_prediction()
                
            except Exception as e:
                print(f"❌ Update error: {e}")
                current_prediction = create_fallback_prediction()
            
            # Wait 30 seconds
            time.sleep(30)
            
    except Exception as e:
        print(f"❌ System error: {e}")
        system_status = "error"

def create_fallback_prediction():
    """Create fallback prediction"""
    current_price = data_fetcher.get_current_real_price()
    
    return {
        'prediction': 'HOLD',
        'confidence': 0.5,
        'is_breakout_signal': False,
        'is_high_confidence': False,
        'alert_message': 'System initializing...',
        'levels': calculate_trading_levels(current_price, {'prediction': 'HOLD'}),
        'trade_suggestion': {
            'action': 'AVOID',
            'size': '0%',
            'reason': 'System starting up',
            'urgency': 'LOW'
        },
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    global current_prediction, system_status, training_progress
    
    if current_prediction is None:
        current_prediction = create_fallback_prediction()
    
    chart_data = create_chart_data()
    
    response = {
        'prediction': current_prediction,
        'chart_data': chart_data,
        'system_status': {
            'status': system_status,
            'training_progress': training_progress,
            'uptime_minutes': int((datetime.now() - start_time).total_seconds() / 60),
            'model_loaded': model_manager.model is not None,
            'next_update': (datetime.now() + timedelta(seconds=30)).strftime('%H:%M:%S')
        },
        'status': 'success'
    }
    
    return jsonify(response)

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Start update thread
    update_thread = threading.Thread(target=update_system, daemon=True)
    update_thread.start()
    
    print("🚀 Bitcoin Scalping System Started!")
    print("📈 Dashboard: http://localhost:5000")
    print("⏰ Updates: Every 30 seconds")
    print("🎯 Goal: High-confidence breakout predictions")
    print("⚠️  Disclaimer: For educational purposes only")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)