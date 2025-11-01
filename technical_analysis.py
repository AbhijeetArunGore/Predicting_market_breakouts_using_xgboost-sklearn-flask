import pandas as pd
import numpy as np
from typing import Optional

class TechnicalAnalysis:
    """Complete technical analysis library with all required methods"""
    
    @staticmethod
    def EMA(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def SMA(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def RSI(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD, Signal Line, and Histogram"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    @staticmethod
    def BBANDS(series: pd.Series, period: int = 20, std: int = 2) -> tuple:
        """Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std_dev = series.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr = TechnicalAnalysis.TRANGE(high, low, close)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def TRANGE(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range
    
    @staticmethod
    def STOCH(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()
        return k, d
    
    @staticmethod
    def WILLIAMS_R(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def CCI(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci