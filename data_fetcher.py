import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframeDataFetcher:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.symbol = "BTCUSDT"
        self.timeframes = ['1m', '5m', '15m']
        
    def fetch_klines(self, interval='1m', limit=500):
        """Fetch klines data from Binance with more data for analysis"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': self.symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'number_of_trades',
                             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching {interval} data: {e}")
            return None
    
    def fetch_all_timeframes(self):
        """Fetch and combine data from all timeframes"""
        try:
            # Fetch current price first
            current_price = self.fetch_current_price()
            if current_price is None:
                print("❌ Failed to fetch current price")
                return None
            
            # Fetch 1-minute data for main analysis
            df_1m = self.fetch_klines('1m', 500)
            if df_1m is None or df_1m.empty:
                print("❌ Failed to fetch 1m data")
                return None
            
            # Get latest candle
            latest_candle = df_1m.iloc[-1]
            
            # Calculate volume ratio (current volume vs average)
            avg_volume = df_1m['volume'].tail(50).mean()
            current_volume = latest_candle['volume']
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate basic price action
            price_change_1h = (latest_candle['close'] - df_1m['close'].iloc[-60]) / df_1m['close'].iloc[-60] * 100 if len(df_1m) >= 60 else 0
            price_change_15m = (latest_candle['close'] - df_1m['close'].iloc[-15]) / df_1m['close'].iloc[-15] * 100 if len(df_1m) >= 15 else 0
            
            # Create feature dictionary with enhanced data
            features = {
                'timestamp': datetime.utcnow().isoformat(),
                'open': float(latest_candle['open']),
                'high': float(latest_candle['high']),
                'low': float(latest_candle['low']),
                'close': float(latest_candle['close']),
                'volume': float(latest_candle['volume']),
                'current_price': current_price,
                'number_of_trades': float(latest_candle.get('number_of_trades', 0)),
                'volume_ratio': volume_ratio,
                'quote_volume': float(latest_candle.get('quote_asset_volume', 0)),
                'price_change_1h': price_change_1h,
                'price_change_15m': price_change_15m,
                'price_range': (latest_candle['high'] - latest_candle['low']) / latest_candle['close'] * 100
            }
            
            print(f"✅ Data fetched - Price: ${current_price:,.2f}, 1h Change: {price_change_1h:+.2f}%, Volume Ratio: {volume_ratio:.2f}")
            return features
            
        except Exception as e:
            print(f"❌ Error in fetch_all_timeframes: {e}")
            return None
    
    def fetch_current_price(self):
        """Fetch current BTC price"""
        try:
            url = f"{self.base_url}/ticker/price"
            params = {'symbol': self.symbol}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"❌ Error fetching current price: {e}")
            return None