from data import BitcoinDataFetcher

def test_real_data():
    """Test if we can get real Bitcoin data"""
    fetcher = BitcoinDataFetcher()
    
    print("🧪 Testing Real Bitcoin Data Fetching...")
    
    # Test current price
    current_price = fetcher.get_current_real_price()
    print(f"✅ Current Bitcoin Price: ${current_price:,.2f}")
    
    # Test historical data
    df = fetcher.fetch_binance_klines(limit=10)
    if not df.empty:
        print(f"✅ Historical Data: {len(df)} records")
        print(f"💰 Latest: ${df['close'].iloc[-1]:,.2f}")
        print(f"📈 Range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
    else:
        print("❌ Failed to get historical data")
    
    # Test full data pipeline
    full_data = fetcher.get_live_data_with_enhanced_features()
    if not full_data.empty:
        print(f"✅ Full Data Pipeline: {len(full_data)} records with indicators")
        print(f"📊 Indicators: EMA5=${full_data['ema_5'].iloc[-1]:,.2f}, RSI={full_data['rsi_14'].iloc[-1]:.1f}")
    else:
        print("❌ Full data pipeline failed")

if __name__ == '__main__':
    test_real_data()