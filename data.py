import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
from technical_analysis import TechnicalAnalysis as ta
import json
import os

class BitcoinDataFetcher:
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.base_url = "https://api.binance.com/api/v3"
        
    def fetch_binance_klines(self, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
        """Fetch real market data from Binance"""
        try:
            print(f"📡 Fetching REAL {self.symbol} data...")
            url = f"{self.base_url}/klines"
            params = {
                'symbol': self.symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Successfully fetched {len(df)} records")
            print(f"💰 Latest price: ${df['close'].iloc[-1]:.2f}")
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"❌ Error fetching Binance data: {e}")
            return pd.DataFrame()
    
    def get_current_real_price(self) -> float:
        """Get current Bitcoin price"""
        try:
            url = f"{self.base_url}/ticker/price"
            params = {'symbol': self.symbol}
            response = requests.get(url, params=params, timeout=5)
            ticker_data = response.json()
            return float(ticker_data['price'])
        except:
            return 65000.0
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with robust error handling"""
        if df.empty:
            return df
            
        try:
            # EMAs
            for period in [3, 5, 8, 13, 21]:
                df[f'ema_{period}'] = ta.EMA(df['close'], period)
            
            # Volume indicators
            df['volume_ma_3'] = ta.SMA(df['volume'], 3)
            df['volume_ma_5'] = ta.SMA(df['volume'], 5)
            df['volume_spike_3'] = df['volume'] / df['volume_ma_3']
            df['volume_spike_5'] = df['volume'] / df['volume_ma_5']
            
            # RSI
            df['rsi_6'] = ta.RSI(df['close'], 6)
            df['rsi_14'] = ta.RSI(df['close'], 14)
            
            # Volatility (ATR)
            df['true_range'] = ta.TRANGE(df['high'], df['low'], df['close'])
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], 14)
            df['atr_percentage'] = (df['atr'] / df['close']) * 100
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'], 20, 2)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            
            # Stochastic
            stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'], 14)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Williams %R
            df['williams_r'] = ta.WILLIAMS_R(df['high'], df['low'], df['close'], 14)
            
            # Price relationships
            df['price_vs_ema3'] = (df['close'] - df['ema_3']) / df['ema_3'] * 100
            df['price_vs_ema8'] = (df['close'] - df['ema_8']) / df['ema_8'] * 100
            
            # Market structure
            df['support_level'] = df['low'].rolling(10).min()
            df['resistance_level'] = df['high'].rolling(10).max()
            
            print("✅ All technical indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"⚠️ Some indicators failed: {e}")
            # Return basic dataframe
            return df
    
    def create_target_variable(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """Create target variable for breakout prediction"""
        if df.empty or len(df) < horizon + 5:
            return df
            
        # Calculate future price movement
        future_price = df['close'].shift(-horizon)
        price_change_pct = (future_price - df['close']) / df['close'] * 100
        
        # Enhanced breakout detection
        conditions = [
            (price_change_pct >= 0.8) & (df['volume_spike_3'] > 1.5),
            (price_change_pct >= 0.3) & (price_change_pct < 0.8),
            (price_change_pct > -0.3) & (price_change_pct < 0.3),
            (price_change_pct > -0.8) & (price_change_pct <= -0.3),
            (price_change_pct <= -0.8) & (df['volume_spike_3'] > 1.5)
        ]
        
        choices = [4, 3, 2, 1, 0]  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
        df['target'] = np.select(conditions, choices, default=2)
        
        return df
    
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for model training"""
        feature_columns = [
            'ema_3', 'ema_5', 'ema_8', 'ema_13', 'ema_21',
            'volume_spike_3', 'volume_spike_5', 
            'rsi_6', 'rsi_14',
            'atr', 'atr_percentage',
            'bb_position', 'macd', 'macd_signal',
            'stoch_k', 'stoch_d', 'williams_r',
            'price_vs_ema3', 'price_vs_ema8'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        features_df = df[available_features].copy()
        
        # Fill NaN values
        features_df = features_df.ffill().bfill()
        
        return features_df
    
    def get_live_data_with_features(self) -> pd.DataFrame:
        """Get live data with all technical indicators"""
        df = self.fetch_binance_klines(interval="1m", limit=100)
        
        if not df.empty:
            df = self.calculate_technical_indicators(df)
            df = self.create_target_variable(df)
        
        return df
    
    def get_training_data(self, days: int = 30) -> pd.DataFrame:
        """Get training data"""
        df = self.fetch_binance_klines(interval="5m", limit=1000)
        
        if not df.empty:
            df = self.calculate_technical_indicators(df)
            df = self.create_target_variable(df)
        else:
            # Create synthetic data for training
            df = self.create_synthetic_data()
        
        return df
    
    def create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data for training"""
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='5min')
        base_price = 110000
        
        # Realistic price series
        returns = np.random.normal(0, 0.001, 1000)
        prices = base_price * (1 + np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.002, 1000)),
            'low': prices * (1 - np.random.uniform(0, 0.002, 1000)),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 1000)
        }, index=dates)
        
        df = self.calculate_technical_indicators(df)
        df = self.create_target_variable(df)
        
        return df