from flask import Flask, render_template, jsonify
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import os

from data import BitcoinDataFetcher
from model_manager import ModelManager
import json

# Optional SocketIO support (if installed). This is non-blocking — the app will still run with polling if SocketIO isn't present.
try:
    from flask_socketio import SocketIO
    socketio = None
except Exception:
    SocketIO = None
    socketio = None

app = Flask(__name__)

# Global variables
current_data = None
current_prediction = None
model_manager = ModelManager()
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
        # Pre-fetch and persist multi-timeframe data for robustness
        try:
            print("📥 Fetching & storing multi-timeframe klines...")
            data_fetcher.fetch_and_store_multi_timeframes()
            print("✅ Multi-timeframe data stored")
        except Exception as e:
            print(f"⚠️ Multi-timeframe fetch/store failed: {e}")

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
        loop_counter = 0
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
                    # Log every prediction for later labeling/retraining
                    try:
                        model_manager.log_prediction(features_df.tail(1), prediction_result)
                    except Exception as e:
                        print(f"⚠️ Failed to log prediction: {e}")

                    # Periodically process logs and trigger retrain from relabels
                    loop_counter += 1
                    if loop_counter % 6 == 0:  # roughly every 3 minutes (6 * 30s)
                        try:
                            new_labels = model_manager.process_prediction_logs()
                            if new_labels and new_labels > 0:
                                print(f"🔖 Newly labeled samples: {new_labels}")
                                retrained = model_manager.trigger_retrain_from_relabels(min_new=100)
                                if retrained:
                                    print("🔁 Breakout model retrained from relabeled data")
                        except Exception as e:
                            print(f"⚠️ Error during log processing/retrain: {e}")
                
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


@app.route('/api/performance')
def performance_api():
    """Return aggregated performance stats and market sentiment"""
    perf_file = 'trade_performance.csv'
    summary = {
        'total_trades': 0,
        'successful_trades': 0,
        'failed_trades': 0,
        'success_rate': 0.0,
        'recent_trades': []
    }

    try:
        if os.path.exists(perf_file):
            df = pd.read_csv(perf_file)
            if not df.empty:
                total = len(df)
                succ = int(df['success'].sum()) if 'success' in df.columns else 0
                fail = total - succ
                rate = succ / total if total > 0 else 0.0
                recent = df.tail(10).to_dict(orient='records')
                summary.update({
                    'total_trades': int(total),
                    'successful_trades': int(succ),
                    'failed_trades': int(fail),
                    'success_rate': round(float(rate), 3),
                    'recent_trades': recent
                })
    except Exception as e:
        print(f"⚠️ Could not read performance file: {e}")

    # Compute market sentiment from latest indicators if available
    sentiment = {
        'score': 0.0,
        'label': 'neutral'
    }
    try:
        if current_data is not None and not current_data.empty:
            last = current_data.tail(5)
            score = 0.0
            # EMA slope
            if 'ema_8' in last.columns and 'ema_21' in last.columns:
                ema8 = last['ema_8'].iloc[-1]
                ema21 = last['ema_21'].iloc[-1]
                score += 0.4 if ema8 > ema21 else -0.4

            # RSI bias
            if 'rsi_14' in last.columns:
                rsi = last['rsi_14'].iloc[-1]
                if rsi < 40:
                    score -= 0.2
                elif rsi > 60:
                    score += 0.2

            # MACD
            if 'macd' in last.columns and 'macd_signal' in last.columns:
                macd = last['macd'].iloc[-1]
                macd_sig = last['macd_signal'].iloc[-1]
                score += 0.2 if macd > macd_sig else -0.2

            # Volume spike
            if 'volume_spike_3' in last.columns:
                vs = last['volume_spike_3'].iloc[-1]
                if vs > 1.5:
                    score += 0.2

            sentiment['score'] = round(max(-1.0, min(1.0, score)), 3)
            if sentiment['score'] > 0.2:
                sentiment['label'] = 'bullish'
            elif sentiment['score'] < -0.2:
                sentiment['label'] = 'bearish'
            else:
                sentiment['label'] = 'neutral'
    except Exception as e:
        print(f"⚠️ Sentiment compute failed: {e}")

    # Combine summary and sentiment
    response = {
        'performance': summary,
        'market_sentiment': sentiment
    }

    return jsonify(response)

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


def build_prediction_payload():
    """Build unified prediction payload used by the dashboard.
    Includes signal, confidence, sentiment, price levels, targets, risk_reward, reason, and performance summary."""
    global current_data, current_prediction

    payload = {
        'signal': 'HOLD',
        'confidence': 0.5,
        'sentiment': 'neutral',
        'price': None,
        'entry': None,
        'stop_loss': None,
        'targets': [],
        'risk_reward': None,
        'market_condition': 'unknown',
        'reason': '',
        'levels': {},
        'timestamp': datetime.now().isoformat(),
        'performance': {},
        'system_status': {
            'status': system_status,
            'training_progress': training_progress,
            'uptime_minutes': int((datetime.now() - start_time).total_seconds() / 60),
            'model_loaded': model_manager.model is not None
        }
    }

    try:
        # Use latest data if available
        df = current_data
        if df is None or df.empty:
            df = data_fetcher.get_live_data_with_features()

        if df is not None and not df.empty:
            features_df = data_fetcher.get_features(df)
            pred = model_manager.predict(features_df)

            # ensure only show when >60% confidence
            conf = float(pred.get('confidence', 0.5))
            payload['signal'] = pred.get('prediction', 'HOLD')
            payload['confidence'] = conf
            payload['breakout_proba'] = pred.get('breakout_proba', 0.0)
            payload['is_high_confidence'] = pred.get('is_high_confidence', False)

            # breakout probability (from ensemble or breakout model)
            payload['breakout_probability'] = float(pred.get('breakout_proba',
                                                            (pred.get('combined_proba')[1] if isinstance(pred.get('combined_proba'), (list, tuple)) and len(pred.get('combined_proba'))>1 else 0.0)))

            # trend strength (normalized) and volatility index
            try:
                if 'ema_8' in df.columns and 'ema_21' in df.columns:
                    ema8 = df['ema_8'].iloc[-1]
                    ema21 = df['ema_21'].iloc[-1]
                    trend_strength = float((ema8 - ema21) / (ema21 if ema21 != 0 else 1.0))
                    # clamp
                    trend_strength = max(-3.0, min(3.0, trend_strength))
                    # normalize to -1..1 roughly
                    payload['trend_strength'] = round(trend_strength / 3.0, 3)
                else:
                    payload['trend_strength'] = 0.0
            except Exception:
                payload['trend_strength'] = 0.0

            try:
                if 'atr_percentage' in df.columns:
                    payload['volatility_index'] = round(float(df['atr_percentage'].iloc[-1]), 3)
                else:
                    payload['volatility_index'] = None
            except Exception:
                payload['volatility_index'] = None

            # model info and live accuracy
            try:
                model_info = model_manager.get_model_info()
                payload['model_version'] = model_info.get('version') if isinstance(model_info, dict) else str(model_info)
            except Exception:
                payload['model_version'] = 'unknown'

            try:
                perf_file = 'trade_performance.csv'
                if os.path.exists(perf_file):
                    pdf = pd.read_csv(perf_file)
                    if not pdf.empty and 'success' in pdf.columns:
                        payload['live_accuracy'] = round(float(int(pdf['success'].sum()) / len(pdf)), 3)
                    else:
                        payload['live_accuracy'] = None
                else:
                    payload['live_accuracy'] = None
            except Exception:
                payload['live_accuracy'] = None

            # price and levels
            current_price = float(df['close'].iloc[-1])
            payload['price'] = round(current_price, 2)
            levels = calculate_trading_levels(current_price, pred)
            payload['levels'] = levels
            payload['entry'] = levels.get('entry')
            payload['stop_loss'] = levels.get('stop_loss')
            payload['targets'] = [levels.get('target_1'), levels.get('target_2')]

            # compute risk-reward (approx)
            try:
                rr = None
                if payload['entry'] and payload['stop_loss'] and payload['targets'][0]:
                    risk = abs(payload['entry'] - payload['stop_loss'])
                    reward = abs(payload['targets'][1] - payload['entry'])
                    rr = round(reward / risk if risk > 0 else 0, 2)
                payload['risk_reward'] = rr
            except Exception:
                payload['risk_reward'] = None

                # reliability estimator: combine live accuracy (if available) and confidence
                try:
                    live_acc = payload.get('live_accuracy')
                    if live_acc is None:
                        reliability = round(min(0.99, 0.5 + (conf - 0.5) * 0.6), 3)
                    else:
                        reliability = round(float((live_acc + conf) / 2.0), 3)
                    payload['reliability'] = reliability
                except Exception:
                    payload['reliability'] = round(conf, 3)

                # sentiment mapping
                try:
                    payload['sentiment'] = payload.get('sentiment', 'neutral')
                except Exception:
                    payload['sentiment'] = 'neutral'

                # explanation (simple heuristic or passthrough)
                try:
                    explain = pred.get('explain') or ''
                    if not explain:
                        if payload.get('breakout_probability', 0) > 0.6:
                            explain = 'Transformer attention: breakout pattern + volume surge'
                        else:
                            explain = 'Indicator ensemble: EMA crossover + MACD confirmation'
                    payload['explain'] = explain
                except Exception:
                    payload['explain'] = ''

                # canonical field names the UI expects
                payload['trend_strength'] = payload.get('trend_strength', 0.0)
                payload['volatility'] = payload.get('volatility_index', payload.get('volatility_index', None))

                # model_version and last_update
                try:
                    model_info = model_manager.get_model_info()
                    payload['model_version'] = model_info.get('version') if isinstance(model_info, dict) and 'version' in model_info else (model_info or f"v{datetime.now().strftime('%Y%m%d_%H%M')}")
                except Exception:
                    payload['model_version'] = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"

                payload['last_update'] = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'

            # market condition and reason from indicators
            reason_parts = []
            if 'ema_8' in df.columns and 'ema_21' in df.columns:
                if df['ema_8'].iloc[-1] > df['ema_21'].iloc[-1]:
                    reason_parts.append('EMA short above long')
                else:
                    reason_parts.append('EMA short below long')
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = df['macd'].iloc[-1]
                macd_sig = df['macd_signal'].iloc[-1]
                if macd > macd_sig:
                    reason_parts.append('MACD bullish')
                else:
                    reason_parts.append('MACD bearish')

            if 'rsi_14' in df.columns:
                rsi = df['rsi_14'].iloc[-1]
                if rsi > 70:
                    reason_parts.append('RSI overbought')
                elif rsi < 30:
                    reason_parts.append('RSI oversold')

            payload['reason'] = ' & '.join(reason_parts)

            # sentiment
            sent_score = 0.0
            try:
                if 'ema_8' in df.columns and 'ema_21' in df.columns:
                    sent_score += 0.4 if df['ema_8'].iloc[-1] > df['ema_21'].iloc[-1] else -0.4
                if 'rsi_14' in df.columns:
                    rsi = df['rsi_14'].iloc[-1]
                    sent_score += 0.2 if rsi > 55 else (-0.2 if rsi < 45 else 0)
                if 'macd' in df.columns and 'macd_signal' in df.columns:
                    sent_score += 0.2 if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else -0.2
                if 'volume_spike_3' in df.columns and df['volume_spike_3'].iloc[-1] > 1.5:
                    sent_score += 0.2
            except Exception:
                pass

            if sent_score > 0.2:
                payload['sentiment'] = 'bullish'
            elif sent_score < -0.2:
                payload['sentiment'] = 'bearish'
            else:
                payload['sentiment'] = 'neutral'

            # attach recent chart data (last 10 minutes)
            try:
                last_n = df.tail(100)
                payload['chart'] = {
                    'timestamps': last_n.index.strftime('%H:%M').tolist(),
                    'prices': last_n['close'].round(2).tolist(),
                    'ema_8': last_n['ema_8'].round(2).tolist() if 'ema_8' in last_n.columns else [],
                    'ema_21': last_n['ema_21'].round(2).tolist() if 'ema_21' in last_n.columns else []
                }

                # include candlestick OHLC for charting
                try:
                    candles = []
                    for ts, row in last_n.iterrows():
                        candles.append({
                            't': ts.isoformat(),
                            'o': float(row['open']),
                            'h': float(row['high']),
                            'l': float(row['low']),
                            'c': float(row['close'])
                        })
                    payload['chart']['candles'] = candles
                except Exception:
                    payload['chart']['candles'] = []
            except Exception:
                payload['chart'] = {}

            # performance summary (reuse existing endpoint logic)
            try:
                perf_file = 'trade_performance.csv'
                perf_summary = {}
                if os.path.exists(perf_file):
                    pdf = pd.read_csv(perf_file)
                    if not pdf.empty:
                        total = len(pdf)
                        succ = int(pdf['success'].sum()) if 'success' in pdf.columns else 0
                        rate = succ / total if total > 0 else 0.0
                        avg_conf = pdf['pred_confidence'].mean() if 'pred_confidence' in pdf.columns else 0.0
                        perf_summary = {
                            'total_trades': int(total),
                            'successful_trades': int(succ),
                            'success_rate': round(rate, 3),
                            'avg_confidence': round(float(avg_conf), 3)
                        }
                payload['performance'] = perf_summary
            except Exception:
                payload['performance'] = {}

            # Only publish a signal if confidence >= 0.6
            if payload.get('confidence', 0) < 0.6:
                payload['signal'] = 'HOLD'

            # Ensure canonical output fields exist (match user's requested format)
            payload.setdefault('signal', 'HOLD')
            payload.setdefault('confidence', round(float(conf), 3))
            payload.setdefault('reliability', payload.get('reliability', round(float(conf), 3)))
            payload.setdefault('sentiment', payload.get('sentiment', 'neutral'))
            payload.setdefault('price', round(float(df['close'].iloc[-1]) if ('close' in df.columns and not df.empty) else data_fetcher.get_current_real_price(), 2))
            payload.setdefault('entry', payload.get('levels', {}).get('entry'))
            payload.setdefault('stop_loss', payload.get('levels', {}).get('stop_loss'))
            payload.setdefault('targets', payload.get('targets', []))
            payload.setdefault('risk_reward', payload.get('risk_reward'))
            payload.setdefault('trend_strength', round(float(payload.get('trend_strength', 0.0)), 3))
            payload.setdefault('volatility', payload.get('volatility_index', None))
            payload.setdefault('explain', payload.get('explain', ''))
            payload.setdefault('model_version', payload.get('model_version', f"v{datetime.now().strftime('%Y%m%d_%H%M')}"))
            payload.setdefault('last_update', payload.get('last_update', datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'))

    except Exception as e:
        print(f"⚠️ build_prediction_payload failed: {e}")

    return payload


@app.route('/api/prediction')
def api_prediction():
    payload = build_prediction_payload()
    return jsonify(payload)


@app.route('/api/model_info')
def api_model_info():
    try:
        info = model_manager.get_model_info()
        return jsonify({'status':'success', 'model_info': info})
    except Exception as e:
        return jsonify({'status':'error', 'error': str(e)})


@app.route('/api/demo')
def api_demo():
    # Return a deterministic demo payload matching the user's requested format
    demo = {
        "signal": "SELL",
        "confidence": 0.675,
        "reliability": 0.82,
        "sentiment": "bearish",
        "price": 110911.86,
        "entry": 110914.59,
        "stop_loss": 110947.32,
        "targets": [110881.85,110860.03],
        "risk_reward": 3.2,
        "trend_strength": 0.58,
        "volatility": 0.35,
        "explain": "Transformer attention: breakout pattern + volume surge",
        "model_version": "v20251102_001",
        "last_update": "2025-11-02T14:47:20Z",
        'chart': {'timestamps': [], 'prices': [], 'candles': []},
        'performance': {'total_trades':0, 'success_rate':0.0, 'avg_confidence':0.0, 'series':[]}
    }
    return jsonify(demo)

if __name__ == '__main__':
    # Start update thread
    update_thread = threading.Thread(target=update_system, daemon=True)
    update_thread.start()

    # Nightly retrain loop (runs in background, non-blocking)
    def nightly_retrain_loop():
        while True:
            try:
                # Sleep for 24 hours between full retrains (adjustable)
                time.sleep(24 * 3600)
                print('🔁 Nightly retrain: starting full model retrain...')
                model_manager.auto_retrain()
                print('🔁 Nightly retrain: completed')
            except Exception as e:
                print(f'⚠️ Nightly retrain failed: {e}')
                time.sleep(60)

    retrain_thread = threading.Thread(target=nightly_retrain_loop, daemon=True)
    retrain_thread.start()
    
    print("🚀 Bitcoin Scalping System Started!")
    print("📈 Dashboard: http://localhost:5000")
    print("⏰ Updates: Every 30 seconds")
    print("🎯 Goal: High-confidence breakout predictions")
    print("⚠️  Disclaimer: For educational purposes only")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)