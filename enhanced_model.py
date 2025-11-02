import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import requests
import ta
from datetime import datetime, timedelta

class EnhancedBreakoutPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = "models/enhanced_breakout_model.pkl"
        self.historical_data = []
        self.max_history = 100
        
    def load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            # Try to load existing model
            self.model = joblib.load(self.model_path)
            print("✅ Pre-trained model loaded successfully")
        except:
            print("⚠️  No pre-trained model found, using advanced real-time analysis...")
            self.model = None

    def fetch_historical_data(self, symbol='BTCUSDT', interval='1m', limit=100):
        """Fetch historical data for technical analysis"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy', 'taker_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return None

    def calculate_advanced_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        if df is None or len(df) < 20:
            return {}
        
        try:
            # Convert to proper format for ta library
            df = df.copy()
            
            # RSI
            rsi_14 = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            rsi_7 = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            macd_histogram = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'], window=20)
            bb_upper = bollinger.bollinger_hband()
            bb_lower = bollinger.bollinger_lband()
            bb_middle = bollinger.bollinger_mavg()
            bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            stoch_k = stoch.stoch()
            stoch_d = stoch.stoch_signal()
            
            # Volume indicators
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'] / volume_sma
            
            # Price action
            price_change = df['close'].pct_change()
            volatility = price_change.rolling(20).std()
            
            # Support and Resistance levels
            resistance = df['high'].rolling(20).max()
            support = df['low'].rolling(20).min()
            
            # Trend indicators
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            
            # Get latest values
            latest = {
                'rsi_14': rsi_14.iloc[-1],
                'rsi_7': rsi_7.iloc[-1],
                'macd_line': macd_line.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'macd_histogram': macd_histogram.iloc[-1],
                'bb_upper': bb_upper.iloc[-1],
                'bb_lower': bb_lower.iloc[-1],
                'bb_position': bb_position.iloc[-1],
                'stoch_k': stoch_k.iloc[-1],
                'stoch_d': stoch_d.iloc[-1],
                'volume_ratio': volume_ratio.iloc[-1],
                'volatility': volatility.iloc[-1],
                'resistance': resistance.iloc[-1],
                'support': support.iloc[-1],
                'sma_20': sma_20.iloc[-1],
                'sma_50': sma_50.iloc[-1],
                'ema_12': ema_12.iloc[-1],
                'ema_26': ema_26.iloc[-1],
                'price': df['close'].iloc[-1],
                'high': df['high'].iloc[-1],
                'low': df['low'].iloc[-1],
                'volume': df['volume'].iloc[-1]
            }
            
            return latest
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return {}

    def analyze_breakout_conditions(self, indicators):
        """Advanced breakout detection algorithm"""
        if not indicators:
            return "HOLD", 0.5, 0.5
        
        # Extract indicators
        rsi_14 = indicators['rsi_14']
        rsi_7 = indicators['rsi_7']
        macd_histogram = indicators['macd_histogram']
        bb_position = indicators['bb_position']
        stoch_k = indicators['stoch_k']
        volume_ratio = indicators['volume_ratio']
        price = indicators['price']
        resistance = indicators['resistance']
        support = indicators['support']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        
        # Initialize scores
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        # 1. RSI Analysis
        if rsi_14 < 30 and rsi_7 < 25:
            buy_signals += 2
            total_signals += 2
        elif rsi_14 > 70 and rsi_7 > 75:
            sell_signals += 2
            total_signals += 2
        elif rsi_14 < 40:
            buy_signals += 1
            total_signals += 1
        elif rsi_14 > 60:
            sell_signals += 1
            total_signals += 1
        
        # 2. MACD Momentum
        if macd_histogram > 0 and macd_histogram > indicators.get('macd_histogram_prev', 0):
            buy_signals += 1
            total_signals += 1
        elif macd_histogram < 0 and macd_histogram < indicators.get('macd_histogram_prev', 0):
            sell_signals += 1
            total_signals += 1
        
        # 3. Bollinger Bands Position
        if bb_position < 0.1:  # Near lower band - potential bounce
            buy_signals += 1
            total_signals += 1
        elif bb_position > 0.9:  # Near upper band - potential rejection
            sell_signals += 1
            total_signals += 1
        
        # 4. Stochastic Oscillator
        if stoch_k < 20 and stoch_k > indicators.get('stoch_k_prev', 0):  # Oversold and turning up
            buy_signals += 1
            total_signals += 1
        elif stoch_k > 80 and stoch_k < indicators.get('stoch_k_prev', 0):  # Overbought and turning down
            sell_signals += 1
            total_signals += 1
        
        # 5. Volume Confirmation
        if volume_ratio > 1.5:  # High volume - confirms moves
            if buy_signals > sell_signals:
                buy_signals += 1
            elif sell_signals > buy_signals:
                sell_signals += 1
            total_signals += 1
        
        # 6. Support/Resistance Breakouts
        if price > resistance * 1.001:  # Breaking resistance
            buy_signals += 2
            total_signals += 2
        elif price < support * 0.999:  # Breaking support
            sell_signals += 2
            total_signals += 2
        
        # 7. Moving Average Trends
        if price > sma_20 > sma_50:  # Strong uptrend
            buy_signals += 1
            total_signals += 1
        elif price < sma_20 < sma_50:  # Strong downtrend
            sell_signals += 1
            total_signals += 1
        
        # Calculate final signal
        if total_signals == 0:
            return "HOLD", 0.5, 0.5
        
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        # Determine signal with confidence
        if buy_ratio > 0.6 and buy_ratio > sell_ratio:
            confidence = min(0.95, 0.6 + (buy_ratio - 0.6) * 0.5)
            reliability = self.calculate_reliability(indicators, "BUY")
            return "BUY", confidence, reliability
        elif sell_ratio > 0.6 and sell_ratio > buy_ratio:
            confidence = min(0.95, 0.6 + (sell_ratio - 0.6) * 0.5)
            reliability = self.calculate_reliability(indicators, "SELL")
            return "SELL", confidence, reliability
        else:
            # Calculate hold confidence based on market clarity
            market_clarity = 1 - abs(buy_ratio - sell_ratio)
            confidence = max(0.3, 0.5 * market_clarity)
            reliability = self.calculate_reliability(indicators, "HOLD")
            return "HOLD", confidence, reliability

    def calculate_reliability(self, indicators, signal):
        """Calculate signal reliability based on market conditions"""
        reliability_factors = []
        
        # Volume reliability
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.2:
            reliability_factors.append(0.9)
        elif volume_ratio > 0.8:
            reliability_factors.append(0.7)
        else:
            reliability_factors.append(0.5)
        
        # Volatility reliability (moderate volatility is best)
        volatility = indicators.get('volatility', 0.02)
        if 0.01 < volatility < 0.04:
            reliability_factors.append(0.8)
        else:
            reliability_factors.append(0.6)
        
        # RSI reliability (extremes are less reliable)
        rsi = indicators.get('rsi_14', 50)
        if 30 < rsi < 70:
            reliability_factors.append(0.8)
        else:
            reliability_factors.append(0.6)
        
        # Indicator convergence reliability
        macd_hist = indicators.get('macd_histogram', 0)
        stoch_k = indicators.get('stoch_k', 50)
        
        if signal == "BUY":
            if macd_hist > 0 and stoch_k < 80:
                reliability_factors.append(0.8)
            else:
                reliability_factors.append(0.6)
        elif signal == "SELL":
            if macd_hist < 0 and stoch_k > 20:
                reliability_factors.append(0.8)
            else:
                reliability_factors.append(0.6)
        else:  # HOLD
            reliability_factors.append(0.7)
        
        return min(0.95, max(0.3, np.mean(reliability_factors)))

    def predict_with_reliability(self, data):
        """Generate real prediction based on advanced technical analysis"""
        try:
            # Fetch historical data for proper technical analysis
            historical_df = self.fetch_historical_data()
            if historical_df is None:
                return self._create_default_prediction(data['current_price'])
            
            # Calculate advanced technical indicators
            indicators = self.calculate_advanced_indicators(historical_df)
            if not indicators:
                return self._create_default_prediction(data['current_price'])
            
            # Generate trading signal
            signal, confidence, reliability = self.analyze_breakout_conditions(indicators)
            
            # Generate detailed explanation
            explanation = self.generate_detailed_explanation(signal, confidence, reliability, indicators)
            
            # Feature importance for display
            feature_importance = self.get_feature_importance(indicators)
            
            return {
                "signal": signal,
                "confidence": float(confidence),
                "reliability": float(reliability),
                "sentiment": "bullish" if signal == "BUY" else "bearish" if signal == "SELL" else "neutral",
                "price": float(data['current_price']),
                "explain": explanation,
                "feature_importance": feature_importance,
                "indicators": {k: float(v) for k, v in indicators.items() if isinstance(v, (int, float))}
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self._create_default_prediction(data.get('current_price', 0))

    def generate_detailed_explanation(self, signal, confidence, reliability, indicators):
        """Generate comprehensive explanation for the trading signal"""
        explanations = []
        
        rsi = indicators.get('rsi_14', 50)
        bb_pos = indicators.get('bb_position', 0.5)
        volume_ratio = indicators.get('volume_ratio', 1)
        macd_hist = indicators.get('macd_histogram', 0)
        stoch_k = indicators.get('stoch_k', 50)
        price = indicators.get('price', 0)
        resistance = indicators.get('resistance', price)
        support = indicators.get('support', price)
        
        # Price action context
        price_vs_resistance = (price / resistance - 1) * 100
        price_vs_support = (price / support - 1) * 100
        
        if abs(price_vs_resistance) < 0.5:
            explanations.append("Price near resistance level")
        if abs(price_vs_support) < 0.5:
            explanations.append("Price near support level")
        
        # RSI analysis
        if rsi < 30:
            explanations.append("RSI indicates strongly oversold conditions")
        elif rsi < 40:
            explanations.append("RSI indicates oversold conditions")
        elif rsi > 70:
            explanations.append("RSI indicates strongly overbought conditions")
        elif rsi > 60:
            explanations.append("RSI indicates overbought conditions")
        else:
            explanations.append("RSI in neutral range")
        
        # Bollinger Bands analysis
        if bb_pos < 0.2:
            explanations.append("Price near Bollinger lower band - potential support")
        elif bb_pos > 0.8:
            explanations.append("Price near Bollinger upper band - potential resistance")
        
        # Volume analysis
        if volume_ratio > 2.0:
            explanations.append("Very high volume - strong momentum")
        elif volume_ratio > 1.5:
            explanations.append("High volume - good confirmation")
        elif volume_ratio < 0.7:
            explanations.append("Low volume - weak momentum")
        
        # MACD analysis
        if macd_hist > 0:
            explanations.append("MACD showing bullish momentum")
        else:
            explanations.append("MACD showing bearish momentum")
        
        # Stochastic analysis
        if stoch_k < 20:
            explanations.append("Stochastic indicates oversold")
        elif stoch_k > 80:
            explanations.append("Stochastic indicates overbought")
        
        # Signal-specific context
        if signal == "BUY":
            explanations.append("Multiple bullish signals aligned")
            if confidence > 0.7:
                explanations.append("Strong buy signal confidence")
        elif signal == "SELL":
            explanations.append("Multiple bearish signals aligned")
            if confidence > 0.7:
                explanations.append("Strong sell signal confidence")
        else:
            explanations.append("Mixed signals - waiting for clearer direction")
        
        # Reliability context
        if reliability > 0.8:
            explanations.append("High reliability market conditions")
        elif reliability < 0.6:
            explanations.append("Lower reliability - exercise caution")
        
        return " • ".join(explanations)

    def get_feature_importance(self, indicators):
        """Get feature importance based on current market conditions"""
        importance_weights = {
            "rsi_14": 0.18,
            "macd_histogram": 0.16,
            "bb_position": 0.14,
            "volume_ratio": 0.12,
            "stoch_k": 0.10,
            "volatility": 0.08,
            "price_vs_resistance": 0.08,
            "price_vs_support": 0.08,
            "sma_trend": 0.06
        }
        
        # Filter available indicators
        available_importance = {}
        for feature, weight in importance_weights.items():
            if feature in indicators or feature.replace('_', ' ') in str(indicators.keys()):
                available_importance[feature] = weight
        
        # Normalize to sum to 1.0
        total = sum(available_importance.values())
        if total > 0:
            available_importance = {k: v/total for k, v in available_importance.items()}
        
        return available_importance

    def _create_default_prediction(self, price):
        """Create default prediction when analysis fails"""
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "reliability": 0.5,
            "sentiment": "neutral",
            "price": float(price),
            "explain": "Analyzing market conditions...",
            "feature_importance": {
                "rsi_14": 0.25,
                "macd_histogram": 0.20,
                "bb_position": 0.15,
                "volume_ratio": 0.15,
                "stoch_k": 0.10,
                "volatility": 0.10,
                "price_action": 0.05
            }
        }