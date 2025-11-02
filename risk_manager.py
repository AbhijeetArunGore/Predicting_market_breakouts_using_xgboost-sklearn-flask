import numpy as np
from datetime import datetime

class RiskManager:
    def __init__(self):
        self.min_confidence = 0.65
        self.min_reliability = 0.60
        self.risk_reward_ratio = 3.0
        self.max_position_size = 0.1
        
    def enhance_prediction(self, prediction, current_price):
        """Add dynamic risk-managed trading parameters to prediction"""
        if prediction['signal'] == 'HOLD':
            return self._create_hold_prediction(prediction, current_price)
        
        # Get market volatility from indicators if available
        indicators = prediction.get('indicators', {})
        volatility = indicators.get('volatility', 0.02)
        bb_position = indicators.get('bb_position', 0.5)
        
        # Adjust risk based on market conditions
        dynamic_risk_multiplier = self.calculate_dynamic_risk(volatility, bb_position, prediction['confidence'])
        
        # Calculate entry, stop loss, and targets
        if prediction['signal'] == 'BUY':
            entry, stop_loss, targets = self.calculate_buy_levels(current_price, volatility, dynamic_risk_multiplier)
        else:  # SELL
            entry, stop_loss, targets = self.calculate_sell_levels(current_price, volatility, dynamic_risk_multiplier)
        
        # Calculate risk-reward ratio
        risk = abs(entry - stop_loss)
        reward = abs(targets[0] - entry) if targets else risk * self.risk_reward_ratio
        risk_reward = reward / risk if risk > 0 else self.risk_reward_ratio
        
        # Calculate position size based on confidence and volatility
        position_size = self.calculate_position_size(prediction['confidence'], volatility)
        
        # Update prediction with risk parameters
        enhanced_prediction = prediction.copy()
        enhanced_prediction.update({
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'targets': [round(t, 2) for t in targets],
            'risk_reward': round(risk_reward, 2),
            'position_size': round(position_size, 4),
            'volatility': round(volatility, 4),
            'trend_strength': self.calculate_trend_strength(indicators)
        })
        
        return enhanced_prediction
    
    def calculate_dynamic_risk(self, volatility, bb_position, confidence):
        """Calculate dynamic risk multiplier based on market conditions"""
        base_multiplier = 1.0
        
        # Adjust for volatility (lower risk in high volatility)
        if volatility > 0.04:
            base_multiplier *= 0.7
        elif volatility < 0.01:
            base_multiplier *= 1.2
        
        # Adjust for Bollinger Band position
        if bb_position < 0.2 or bb_position > 0.8:
            base_multiplier *= 0.8  # Reduce risk at extremes
        
        # Adjust for confidence
        base_multiplier *= min(1.5, confidence * 2)
        
        return max(0.5, min(2.0, base_multiplier))
    
    def calculate_position_size(self, confidence, volatility):
        """Calculate position size based on confidence and volatility"""
        base_size = self.max_position_size
        
        # Adjust for volatility (lower size in high volatility)
        vol_adjustment = max(0.3, 1.0 - (volatility * 20))
        
        # Adjust for confidence
        conf_adjustment = min(1.5, confidence * 2)
        
        position_size = base_size * vol_adjustment * conf_adjustment
        return max(0.01, min(self.max_position_size, position_size))
    
    def calculate_trend_strength(self, indicators):
        """Calculate trend strength from indicators"""
        if not indicators:
            return 0.5
        
        strength_factors = []
        
        # MACD histogram strength
        macd_hist = abs(indicators.get('macd_histogram', 0))
        if macd_hist > 0.001:
            strength_factors.append(0.8)
        else:
            strength_factors.append(0.3)
        
        # Price vs moving averages
        price = indicators.get('price', 0)
        sma_20 = indicators.get('sma_20', price)
        if abs(price / sma_20 - 1) > 0.01:
            strength_factors.append(0.7)
        else:
            strength_factors.append(0.4)
        
        # Volume strength
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.2:
            strength_factors.append(0.8)
        else:
            strength_factors.append(0.5)
        
        return min(1.0, max(0.0, np.mean(strength_factors)))
    
    def calculate_buy_levels(self, current_price, volatility, risk_multiplier):
        """Calculate buy entry, stop loss, and targets"""
        # Dynamic entry based on volatility
        entry_spread = volatility * 0.5 * risk_multiplier
        entry = current_price * (1 + entry_spread)
        
        # Dynamic stop loss based on volatility and risk multiplier
        stop_loss_spread = volatility * 1.5 * risk_multiplier
        stop_loss = current_price * (1 - stop_loss_spread)
        
        # Profit targets with dynamic spacing
        target1 = entry + (entry - stop_loss) * 1.0
        target2 = entry + (entry - stop_loss) * 2.0
        target3 = entry + (entry - stop_loss) * 3.0
        
        targets = [target1, target2, target3]
        
        return entry, stop_loss, targets
    
    def calculate_sell_levels(self, current_price, volatility, risk_multiplier):
        """Calculate sell entry, stop loss, and targets"""
        # Dynamic entry based on volatility
        entry_spread = volatility * 0.5 * risk_multiplier
        entry = current_price * (1 - entry_spread)
        
        # Dynamic stop loss based on volatility and risk multiplier
        stop_loss_spread = volatility * 1.5 * risk_multiplier
        stop_loss = current_price * (1 + stop_loss_spread)
        
        # Profit targets with dynamic spacing
        target1 = entry - (stop_loss - entry) * 1.0
        target2 = entry - (stop_loss - entry) * 2.0
        target3 = entry - (stop_loss - entry) * 3.0
        
        targets = [target1, target2, target3]
        
        return entry, stop_loss, targets
    
    def _create_hold_prediction(self, prediction, current_price):
        """Create enhanced prediction for HOLD signal"""
        enhanced_prediction = prediction.copy()
        
        # Use default risk parameters for HOLD
        volatility = prediction.get('volatility', 0.02)
        
        enhanced_prediction.update({
            'entry': round(current_price, 2),
            'stop_loss': round(current_price * 0.98, 2),
            'targets': [round(current_price * 1.02, 2)],
            'risk_reward': 1.0,
            'position_size': 0.0,
            'volatility': round(volatility, 4),
            'trend_strength': 0.5
        })
        return enhanced_prediction