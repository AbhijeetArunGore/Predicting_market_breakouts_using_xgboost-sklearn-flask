import pandas as pd
import numpy as np
from config import Config

class RiskManager:
    def __init__(self):
        self.max_position_size = 0.1  # 10% of portfolio per trade
        self.stop_loss_pct = 0.02    # 2% stop loss
        self.take_profit_pct = 0.03  # 3% take profit
        self.max_daily_loss = 0.05   # 5% maximum daily loss
        
    def calculate_position_size(self, portfolio_value, prediction_confidence, volatility):
        """Calculate position size based on risk parameters"""
        base_size = self.max_position_size * portfolio_value
        
        # Adjust based on prediction confidence
        confidence_multiplier = min(1.0, prediction_confidence / 0.7)
        
        # Adjust based on volatility (reduce position in high volatility)
        volatility_multiplier = max(0.1, 1.0 - volatility * 10)
        
        position_size = base_size * confidence_multiplier * volatility_multiplier
        
        return min(position_size, portfolio_value * self.max_position_size)
    
    def calculate_stop_loss(self, entry_price, signal_type):
        """Calculate stop loss price"""
        if signal_type == 'BUY':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price, signal_type):
        """Calculate take profit price"""
        if signal_type == 'BUY':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def should_enter_trade(self, current_volatility, recent_performance, daily_pnl):
        """Determine if should enter new trade based on market conditions"""
        # Check daily loss limit
        if daily_pnl <= -self.max_daily_loss:
            return False, "Daily loss limit reached"
        
        # Check volatility
        if current_volatility > 0.05:  # 5% volatility threshold
            return False, "High volatility"
        
        return True, "OK"

if __name__ == "__main__":
    risk_mgr = RiskManager()
    
    # Test risk management
    portfolio_value = 10000
    confidence = 0.85
    volatility = 0.02
    
    position_size = risk_mgr.calculate_position_size(portfolio_value, confidence, volatility)
    print(f"Recommended position size: ${position_size:.2f}")