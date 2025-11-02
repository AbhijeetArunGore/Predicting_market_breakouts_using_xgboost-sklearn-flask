import os
from datetime import datetime


class Config:
    """Central configuration for the scalping system.

    Keeps only basic, safe defaults. Other modules import Config
    and expect attributes like MODEL_DIR, LOG_DIR, SYMBOLS, etc.
    """
    # API endpoints
    BINANCE_BASE_URL = "https://api.binance.com/api/v3"
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

    # Trading parameters
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    TIMEFRAME = "5m"
    LOOKBACK_PERIOD = 1000

    # Technical indicator defaults
    TECHNICAL_INDICATORS = {
        'sma_periods': [10, 20, 50],
        'ema_periods': [12, 26],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'bb_std': 2,
        'atr_period': 14
    }

    # Model / retrain settings
    BREAKOUT_THRESHOLD = 0.015
    PREDICTION_HORIZON = 10
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    RETRAIN_INTERVAL_HOURS = 24
    MIN_SAMPLES_FOR_RETRAIN = 1000
    PERFORMANCE_THRESHOLD = 0.02

    # Paths (used by retrain_manager and logging)
    MODEL_DIR = os.path.join(os.getcwd(), "bitcoin_scalping_system", "models")
    DATA_DIR = os.path.join(os.getcwd(), "bitcoin_scalping_system", "data")
    LOG_DIR = os.path.join(os.getcwd(), "bitcoin_scalping_system", "logs")

    @classmethod
    def setup_directories(cls):
        for d in (cls.MODEL_DIR, cls.DATA_DIR, cls.LOG_DIR):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass


def get_config():
    return Config


# Create dirs on import (best-effort)
try:
    Config.setup_directories()
except Exception:
    pass

