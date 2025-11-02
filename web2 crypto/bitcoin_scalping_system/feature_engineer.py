import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')

from config import Config

class FeatureEngineer:
    def __init__(self):
        self.config = Config.TECHNICAL_INDICATORS
        
    def calculate_sma(self, df, periods):
        """Calculate Simple Moving Averages"""
        for period in periods:
            sma = SMAIndicator(close=df['close'], window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
            df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        return df
    
    def calculate_ema(self, df, periods):
        """Calculate Exponential Moving Averages"""
        for period in periods:
            ema = EMAIndicator(close=df['close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        return df
    
    def calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index"""
        rsi = RSIIndicator(close=df['close'], window=period)
        df['rsi'] = rsi.rsi()
        return df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        macd = MACD(close=df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        return df
    
    def calculate_bollinger_bands(self, df, period=20, std=2):
        """Calculate Bollinger Bands"""
        bb = BollingerBands(close=df['close'], window=period, window_dev=std)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period)
        df['atr'] = atr.average_true_range()
        df['atr_percentage'] = df['atr'] / df['close']
        return df
    
    def calculate_volume_indicators(self, df):
        """Calculate volume-based indicators"""
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
        df['vwap'] = vwap.volume_weighted_average_price()
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        return df
    
    def calculate_price_features(self, df):
        """Calculate price-based features"""
        # Price changes
        for period in [1, 3, 5, 10]:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
            df[f'high_low_ratio_{period}'] = (df['high'] / df['low']).rolling(period).mean()
        
        # Volatility
        df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
        df['volatility_50'] = df['close'].pct_change().rolling(window=50).std()
        
        # Support and resistance levels (simplified)
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()
        df['distance_to_resistance'] = (df['resistance_20'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_20']) / df['close']
        
        return df
    
    def engineer_features(self, df):
        """Apply all feature engineering steps"""
        if df is None or len(df) < 50:
            return None
            
        df = df.copy()
        
        # Calculate all technical indicators
        df = self.calculate_sma(df, self.config['sma_periods'])
        df = self.calculate_ema(df, self.config['ema_periods'])
        df = self.calculate_rsi(df, self.config['rsi_period'])
        df = self.calculate_macd(df, self.config['macd_fast'], self.config['macd_slow'], self.config['macd_signal'])
        df = self.calculate_bollinger_bands(df, self.config['bb_period'], self.config['bb_std'])
        df = self.calculate_atr(df, self.config['atr_period'])
        df = self.calculate_volume_indicators(df)
        df = self.calculate_price_features(df)
        
        # Drop rows with NaN values (due to rolling windows)
        df = df.dropna()
        
        return df
    
    def get_feature_columns(self):
        """Get list of all feature columns"""
        feature_columns = []
        
        # SMA features
        for period in Config.TECHNICAL_INDICATORS['sma_periods']:
            feature_columns.extend([f'sma_{period}', f'price_vs_sma_{period}'])
        
        # EMA features  
        for period in Config.TECHNICAL_INDICATORS['ema_periods']:
            feature_columns.extend([f'ema_{period}', f'price_vs_ema_{period}'])
        
        # Other indicators
        feature_columns.extend([
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
            'atr', 'atr_percentage', 'volume_sma_20', 'volume_ratio',
            'vwap', 'price_vs_vwap'
        ])
        
        # Price features
        for period in [1, 3, 5, 10]:
            feature_columns.extend([f'price_change_{period}', f'high_low_ratio_{period}'])
        
        feature_columns.extend([
            'volatility_20', 'volatility_50', 'resistance_20', 'support_20',
            'distance_to_resistance', 'distance_to_support'
        ])
        
        return feature_columns

if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    from data_fetcher import CryptoDataFetcher
    
    Config.setup_directories()
    fetcher = CryptoDataFetcher()
    engineer = FeatureEngineer()
    
    # Fetch sample data
    data = fetcher.fetch_historical_data("BTCUSDT", days=30)
    if data is not None:
        features = engineer.engineer_features(data)
        print(f"Generated {len(engineer.get_feature_columns())} features")
        print(f"Feature matrix shape: {features[engineer.get_feature_columns()].shape}")