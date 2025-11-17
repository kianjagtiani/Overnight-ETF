import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# CONFIG
# -------------------------
RANDOM_STATE = 42
SLIPPAGE_PCT = 0.0005
COMMISSION_PCT = 0.0000
EXIT_MINUTES_AFTER_OPEN = 5

# Map CSV files to symbols
CSV_FILES = {
    'SPY': 'spy_5min_polygon_6months.csv',
    'QQQ': 'qqq_5min_polygon_6months.csv',
    'IWM': 'iwm_5min_polygon_6months.csv',
    'XLF': 'xlf_5min_polygon_6months.csv',
    'XLE': 'xle_5min_polygon_6months.csv',
    'GLD': 'gld_5min_polygon_6months.csv',
}

# -------------------------
# FUNCTIONS FROM YOUR EXISTING CODE
# -------------------------

def get_eod_candle_features(df_intraday, minutes_from_close_list=[10,20,30,60]):
    records = []
    for date, group in df_intraday.groupby(df_intraday.index.date):
        rec = {'date': pd.to_datetime(date)}
        g = group.copy()
        market_close = pd.Timestamp(date) + pd.Timedelta(hours=16)
        
        for m in minutes_from_close_list:
            start = market_close - pd.Timedelta(minutes=m)
            sub = g[g.index >= start]
            if len(sub) >= 1:
                o = sub['Open'].iloc[0]
                c = sub['Close'].iloc[-1]
                rec[f'dir_last_{m}'] = np.sign(c - o)
                rec[f'mag_last_{m}'] = (c - o) / o if o != 0 else 0
                if 'Volume' in sub.columns:
                    rec[f'vol_last_{m}'] = sub['Volume'].sum()
            else:
                rec[f'dir_last_{m}'] = np.nan
                rec[f'mag_last_{m}'] = np.nan
                rec[f'vol_last_{m}'] = np.nan
        
        if len(g) > 0:
            day_open = g['Open'].iloc[0]
            day_close = g['Close'].iloc[-1]
            rec['day_mag'] = (day_close - day_open) / day_open if day_open != 0 else 0
            rec['close'] = day_close
        
        records.append(rec)
    
    return pd.DataFrame(records).set_index('date').sort_index()

def build_open_exit_lookup(df_intraday, exit_minutes=5):
    lookup = {}
    for date, g in df_intraday.groupby(df_intraday.index.date):
        g2 = g.copy()
        market_open = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30)
        exit_time = market_open + pd.Timedelta(minutes=exit_minutes)
        
        exit_row = g2[g2.index >= exit_time]
        if len(exit_row) > 0:
            exit_price = exit_row['Close'].iloc[0]
            open_series = g2[g2.index >= market_open]
            if len(open_series) > 0:
                entry_price = open_series['Open'].iloc[0]
                lookup[pd.to_datetime(date)] = {'entry': entry_price, 'exit': exit_price}
    return lookup

def backtest_symbol(symbol, filepath):
    """Run backtest for a single symbol"""
    print(f"\n{'='*60}")
    print(f"Processing {symbol}")
    print(f"{'='*60}")
    
    # Load data
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    data = data.between_time('09:30', '16:00')
    
    print(f"Loaded {len(data)} bars, {len(set(data.index.date))} trading days")
    
    # Build daily data
    daily = data.groupby(data.index.date).agg({'Open':'first','Close':'last'})
    daily.index = pd.to_datetime(daily.index)
    daily['next_open'] = daily['Open'].shift(-1)
    daily['overnight_return'] = (daily['next_open'] - daily['Close']) / daily['Close']
    daily['overnight_dir'] = (daily['overnight_return'] > 0).astype(int)
    
    # Extract features
    feats = get_eod_candle_features(data, minutes_from_close_list=[10,20,30,60])
    
    # Merge
    df = daily.join(feats, how='inner').dropna(subset=['overnight_dir'])
    df = df[~df['next_open'].isna()].dropna()
    
    if len(df) < 30:
        print(f"Insufficient data ({len(df)} days)")
        return None
    
    # Features and target
    feature_cols = [c for c in df.columns if c.startswith('dir_last_') or 
                    c.startswith('mag_last_') or c.startswith('vol_last_') or c=='day_mag']
    X = df[feature_cols].copy()
    y = df['overnight_dir'].astype(int).copy()
    
    # Train/test split
    split = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    if len(X_test) < 10:
        print(f"Insufficient test data ({len(X_test)} days)")
        return None
    
    print(f"Training: {len(X_train)} days | Testing: {len(X_test)} days")
    
    # Train model
    param_grid = {
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [5, 10, 15],
    }
    
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=min(3, len(np.unique(y_train))), shuffle=False)
    grid = GridSearchCV(dt, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    model = grid.best_estimator_
    
    # Build exit lookup
    open_exit = build_open_exit_lookup(data, exit_minutes=EXIT_MINUTES_AFTER_OPEN)
    
    # Backtest
    trades = []
    for dt_index in X_test.index:
        pred = model.predict(X.loc[[dt_index]])[0]
        next_day = (dt_index + pd.Timedelta(days=1)).normalize()
        
        info = open_exit.get(next_day)
        if info is None:
            key_candidates = [k for k in open_exit.keys() if k.date() == next_day.date()]
            if key_candidates:
                info = open_exit[key_candidates[0]]
        
        if info is None:
            continue
        
        entry = info['entry']
        exitp = info['exit']
        pos = 1 if pred == 1 else -1
        gross_ret = pos * (exitp / entry - 1)
        net_ret = gross_ret - (2 * SLIPPAGE_PCT + 2 * COMMISSION_PCT)
        
        trades.append({
            'day': dt_index,
            'net_ret': net_ret
        })
    
    if not trades:
        print("No trades generated")
        return None
    
    trades_df = pd.DataFrame(trades).set_index('day')
    
    # Calculate metrics
    avg_ret = trades_df['net_ret'].mean()
    win_rate = (trades_df['net_ret'] > 0).mean()
    cum_ret = (1 + trades_df['net_ret']).prod()
    
    t_stat, p_value = stats.ttest_1samp(trades_df['net_ret'], 0)
    sharpe = (avg_ret / trades_df['net_ret'].std()) * np.sqrt(252) if trades_df['net_ret'].std() > 0 else 0
    
    # Feature importances
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_feature = imp.index[0] if len(imp) > 0 else 'None'
    
    print(f"Results: {len(trades_df)} trades | Avg: {avg_ret:.4f} | Win: {win_rate:.1%} | Sharpe: {sharpe:.2f} | p-val: {p_value:.3f}")
    
    return {
        'symbol': symbol,
        'num_trades': len(trades_df),
        'avg_return': avg_ret,
        'win_rate': win_rate,
        'cum_return': cum_ret,
        'sharpe_ratio': sharpe,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'profitable': avg_ret > 0,
        'top_feature': top_feature,
    }

# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    print("="*60)
    print("MULTI-ASSET STRATEGY BACKTEST")
    print("="*60)
    
    results = []
    
    for symbol, filepath in CSV_FILES.items():
        try:
            result = backtest_symbol(symbol, filepath)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"ERROR with {symbol}: {e}")
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY RESULTS")
        print("="*70)
        
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values('sharpe_ratio', ascending=False)
        
        print("\nRanked by Sharpe Ratio:")
        print(summary_df.to_string(index=False))
        
        profitable = summary_df[summary_df['profitable']]
        significant = summary_df[summary_df['significant']]
        
        print(f"\n{'='*70}")
        print(f"Profitable: {len(profitable)}/{len(summary_df)}")
        print(f"Significant: {len(significant)}/{len(summary_df)}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summary_df.plot(x='symbol', y='sharpe_ratio', kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Sharpe Ratio by Symbol')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        summary_df.plot(x='symbol', y='avg_return', kind='bar', ax=axes[0, 1], color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Average Return per Trade')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        summary_df.plot(x='symbol', y='win_rate', kind='bar', ax=axes[1, 0], color='orange')
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--')
        axes[1, 0].set_title('Win Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        summary_df.plot(x='symbol', y='cum_return', kind='bar', ax=axes[1, 1], color='purple')
        axes[1, 1].axhline(y=1, color='r', linestyle='--')
        axes[1, 1].set_title('Cumulative Return')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_asset_comparison.png', dpi=150)
        print(f"\nSaved chart to multi_asset_comparison.png")
        plt.show()
        
        summary_df.to_csv('multi_asset_summary.csv', index=False)
        print(f"Saved summary to multi_asset_summary.csv")