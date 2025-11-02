import os
from datetime import timedelta

# API Configuration
FREECRYPTOAPI_KEY = "7s3dmzpb2yn0um4alacv"
BINANCE_API_KEY = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
# Ultimat
SYMBOL = "BTCUSDT"

# Ultra-Fast Scalping Parameters
TIMEFRAME = "1m"
PREDICTION_HORIZON = 10  # minutes ahead to predict breakouts
REWARD_RISK_RATIO = 3.0

# Advanced Technical Indicators
EMA_PERIODS = [3, 5, 8, 13, 21]
RSI_PERIODS = [6, 14]
VOLUME_SENSITIVITY = 2.0

# AI Model Configuration for Maximum Accuracy
MODEL_CONFIG = {
    'output_classes': ['STRONG_SELL', 'SELL', 'HOLD', 'BUY', 'STRONG_BUY'],
    'prediction_horizon': 'next_5_minutes',
    'prediction_horizon_minutes': 10,
    'retrain_interval': timedelta(hours=4),
    'confidence_thresholds': {
        'STRONG_BUY': 0.90,
        'BUY': 0.80,
        'HOLD': 0.0,
        'SELL': 0.80,
        'STRONG_SELL': 0.90
    },
    'accuracy_target': 0.85,
    'daily_improvement_target': 0.02
}

# Advanced Risk Management
RISK_CONFIG = {
    'portfolio_risk_per_trade': 0.01,
    'max_daily_loss': 0.03,
    'position_sizing': {
        '90_100_confidence': 0.05,
        '80_90_confidence': 0.03,
        '70_80_confidence': 0.02,
        'below_70': 0.00
    },
    'dynamic_stop_loss': True,
    'trailing_stop': True
}

# Performance Tracking
PERFORMANCE_TRACKING = {
    'accuracy_file': 'model_accuracy.csv',
    'trade_log_file': 'trade_performance.csv'
}