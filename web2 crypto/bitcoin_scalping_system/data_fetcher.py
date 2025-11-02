import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import os
from config import Config

class CryptoDataFetcher:
    def __init__(self):
        self.base_url = Config.BINANCE_BASE_URL
        self.symbols = Config.SYMBOLS
        
    def fetch_historical_data(self, symbol, days=30):
        """Fetch historical data for backtesting and initial training"""
        try:
            interval_map = {
                '5m': '5m',
                '15m': '15m', 
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            interval = interval_map.get(Config.TIMEFRAME, '5m')
            limit = min(days * 288, 1000)  # Binance limit
            
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Save historical data
            historical_path = os.path.join(Config.DATA_DIR, "historical", f"{symbol}_historical.csv")
            df.to_csv(historical_path)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def fetch_live_data(self, symbol):
        """Fetch current price and recent candles"""
        try:
            # Fetch current price
            ticker_url = f"{self.base_url}/ticker/price"
            ticker_response = requests.get(f"{ticker_url}?symbol={symbol}")
            current_price = float(ticker_response.json()['price'])
            
            # Fetch recent candles for features
            kline_url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
                'limit': 100
            }
            
            kline_response = requests.get(kline_url, params=params)
            kline_data = kline_response.json()
            
            df = pd.DataFrame(kline_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Save live data
            live_path = os.path.join(Config.DATA_DIR, "live", f"{symbol}_live.csv")
            df.to_csv(live_path)
            
            return df, current_price
            
        except Exception as e:
            print(f"Error fetching live data for {symbol}: {e}")
            return None, None
    
    def fetch_multiple_symbols(self):
        """Fetch data for all configured symbols"""
        all_data = {}
        
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            historical_data = self.fetch_historical_data(symbol)
            live_data, current_price = self.fetch_live_data(symbol)
            
            all_data[symbol] = {
                'historical': historical_data,
                'live': live_data,
                'current_price': current_price
            }
            
            time.sleep(0.5)  # Rate limiting
            
        return all_data

if __name__ == "__main__":
    Config.setup_directories()
    fetcher = CryptoDataFetcher()
    data = fetcher.fetch_multiple_symbols()
    print("Data fetching completed!")