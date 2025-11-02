import sqlite3
import pandas as pd
import os
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(__file__), 'data.db')


def init_db(db_path: Optional[str] = None):
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # klines table: store OHLCV per symbol/interval/timestamp
    cur.execute('''
    CREATE TABLE IF NOT EXISTS klines (
        symbol TEXT,
        interval TEXT,
        timestamp TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        open_interest REAL,
        PRIMARY KEY(symbol, interval, timestamp)
    )
    ''')

    # predictions, relabels, performance tables (minimal columns)
    cur.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pred_time TEXT,
        symbol TEXT,
        interval TEXT,
        prediction TEXT,
        confidence REAL,
        breakout_proba REAL,
        features_json TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pred_time TEXT,
        pred_label TEXT,
        pred_confidence REAL,
        pred_price REAL,
        future_price REAL,
        price_change_pct REAL,
        success INTEGER
    )
    ''')

    conn.commit()
    conn.close()


def save_klines(df: pd.DataFrame, symbol: str = 'BTCUSDT', interval: str = '1m', db_path: Optional[str] = None):
    """Save klines DataFrame to the klines table. Expects index as timestamp."""
    if db_path is None:
        db_path = DB_PATH
    if df is None or df.empty:
        return 0

    df2 = df.copy()
    df2 = df2.reset_index()
    df2.rename(columns={'index': 'timestamp'}, inplace=True)
    # Ensure timestamp is ISO string
    df2['timestamp'] = pd.to_datetime(df2['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    records = []
    for _, r in df2.iterrows():
        records.append((symbol, interval, r['timestamp'], float(r.get('open', 0)), float(r.get('high', 0)),
                        float(r.get('low', 0)), float(r.get('close', 0)), float(r.get('volume', 0)),
                        float(r.get('open_interest', 'nan') if pd.notna(r.get('open_interest', None)) else None)))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executemany('''INSERT OR REPLACE INTO klines
                       (symbol, interval, timestamp, open, high, low, close, volume, open_interest)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', records)
    conn.commit()
    conn.close()
    return len(records)


def get_klines(symbol: str = 'BTCUSDT', interval: str = '1m', limit: int = 1000, db_path: Optional[str] = None):
    if db_path is None:
        db_path = DB_PATH
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    query = f"SELECT timestamp, open, high, low, close, volume, open_interest FROM klines WHERE symbol=? AND interval=? ORDER BY timestamp DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(symbol, interval, limit))
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    # order ascending
    df = df.sort_index()
    return df


def save_performance_row(row: dict, db_path: Optional[str] = None):
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''INSERT INTO performance (pred_time, pred_label, pred_confidence, pred_price, future_price, price_change_pct, success)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''', (
                   row.get('pred_time'), row.get('pred_label'), row.get('pred_confidence'), row.get('pred_price'),
                   row.get('future_price'), row.get('price_change_pct'), int(row.get('success', 0))
    ))
    conn.commit()
    conn.close()
