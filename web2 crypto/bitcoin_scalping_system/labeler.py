import pandas as pd
import numpy as np
from config import Config

class BreakoutLabeler:
    def __init__(self):
        self.breakout_threshold = Config.BREAKOUT_THRESHOLD
        self.prediction_horizon = Config.PREDICTION_HORIZON
        
    def generate_labels(self, df):
        """
        Generate breakout labels based on future price movement
        Label 1 if price moves above current high by > threshold in next N periods
        Label 0 otherwise
        """
        if df is None or len(df) < self.prediction_horizon + 10:
            return None
            
        df = df.copy()
        
        # Calculate future max price in prediction horizon
        df['future_max'] = df['high'].shift(-self.prediction_horizon).rolling(
            window=self.prediction_horizon, min_periods=1
        ).max()
        
        # Calculate breakout condition
        df['breakout_move'] = (df['future_max'] - df['high']) / df['high']
        df['label'] = (df['breakout_move'] > self.breakout_threshold).astype(int)
        
        # Drop rows where we don't have enough future data
        df = df[:-self.prediction_horizon]
        
        # Remove temporary columns
        df = df.drop(['future_max', 'breakout_move'], axis=1)
        
        return df
    
    def calculate_class_weights(self, labels):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        return dict(zip(classes, weights))
    
    def get_label_stats(self, df_with_labels):
        """Get statistics about the generated labels"""
        if df_with_labels is None or 'label' not in df_with_labels.columns:
            return None
            
        total_samples = len(df_with_labels)
        breakout_samples = df_with_labels['label'].sum()
        non_breakout_samples = total_samples - breakout_samples
        
        stats = {
            'total_samples': total_samples,
            'breakout_samples': breakout_samples,
            'non_breakout_samples': non_breakout_samples,
            'breakout_ratio': breakout_samples / total_samples,
            'non_breakout_ratio': non_breakout_samples / total_samples
        }
        
        return stats

if __name__ == "__main__":
    # Test labeling
    import pandas as pd
    from data_fetcher import CryptoDataFetcher
    from feature_engineer import FeatureEngineer
    
    Config.setup_directories()
    fetcher = CryptoDataFetcher()
    engineer = FeatureEngineer()
    labeler = BreakoutLabeler()
    
    # Fetch and process sample data
    data = fetcher.fetch_historical_data("BTCUSDT", days=60)
    if data is not None:
        features = engineer.engineer_features(data)
        labeled_data = labeler.generate_labels(features)
        
        if labeled_data is not None:
            stats = labeler.get_label_stats(labeled_data)
            print("Label Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value:.4f}")