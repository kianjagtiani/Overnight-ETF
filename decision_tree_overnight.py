import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
import joblib

# -------------------------
# CONFIG
RANDOM_STATE = 42
SLIPPAGE_PCT = 0.0005
COMMISSION_PCT = 0.0000
EXIT_MINUTES_AFTER_OPEN = 5

# Path to your downloaded CSV file
DATA_FILE = 'spy_5min_polygon_6months.csv'  # or 'spy_5min_polygon_6months.csv'
# -------------------------

print("Loading data from CSV...")
# Load the data (already has Open, High, Low, Close, Volume columns)
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

# Add date and time columns
data['date'] = data.index.date
data['time'] = data.index.time

print(f"Loaded {len(data)} bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Unique trading days: {len(set(data['date']))}")

# Filter to regular market hours (9:30 AM - 4:00 PM ET)
data = data.between_time('09:30', '16:00')
print(f"After filtering to market hours: {len(data)} bars")

# 2) Make daily open/close and next open (overnight)
daily = data.groupby('date').agg({'Open':'first','Close':'last'})
daily.index = pd.to_datetime(daily.index)
daily['next_open'] = daily['Open'].shift(-1)
daily['overnight_return'] = (daily['next_open'] - daily['Close']) / daily['Close']
daily['overnight_dir'] = (daily['overnight_return'] > 0).astype(int)

# 3) Feature engineering: end-of-day candle directions and magnitudes
def get_eod_candle_features(df_intraday, minutes_from_close_list=[10,20,30,60]):
    """
    Returns a DataFrame indexed by date with features:
      - dir_last_N: sign of (close - open) in the last N minutes
      - mag_last_N: (close - open)/open in that same window
      - vol_last_N: sum(volume) in last N minutes
    """
    records = []
    for date, group in df_intraday.groupby('date'):
        rec = {'date': pd.to_datetime(date)}
        g = group.copy()
        g.index = pd.to_datetime(g.index)
        
        # Market close is 4:00 PM
        market_close = g.index[0].normalize() + pd.Timedelta(hours=16)
        
        for m in minutes_from_close_list:
            start = market_close - pd.Timedelta(minutes=m)
            sub = g[g.index >= start]
            if len(sub) >= 1:
                o = sub['Open'].iloc[0]
                c = sub['Close'].iloc[-1]
                rec[f'dir_last_{m}'] = np.sign(c - o)
                rec[f'mag_last_{m}'] = (c - o) / o
                if 'Volume' in sub.columns:
                    rec[f'vol_last_{m}'] = sub['Volume'].sum()
            else:
                rec[f'dir_last_{m}'] = np.nan
                rec[f'mag_last_{m}'] = np.nan
                rec[f'vol_last_{m}'] = np.nan
        
        # Entire day open->close magnitude
        day_open = g['Open'].iloc[0]
        day_close = g['Close'].iloc[-1]
        rec['day_mag'] = (day_close - day_open) / day_open
        rec['close'] = day_close
        records.append(rec)
    
    df_feats = pd.DataFrame(records).set_index('date').sort_index()
    return df_feats

print("\nBuilding EOD features...")
feats = get_eod_candle_features(data, minutes_from_close_list=[10,20,30,60])

# 4) Merge features with daily overnight target
df = daily.join(feats, how='inner').dropna(subset=['overnight_dir'])
df = df[~df['next_open'].isna()]
df = df.dropna()

print(f"Total labeled days: {len(df)}")
print(f"Class distribution: {df['overnight_dir'].value_counts().to_dict()}")

# 5) Prepare X, y
feature_cols = [c for c in df.columns if c.startswith('dir_last_') or 
                c.startswith('mag_last_') or c.startswith('vol_last_') or c=='day_mag']
X = df[feature_cols].copy()
y = df['overnight_dir'].astype(int).copy()

# 6) Train/test split (70/30 time-based)
n = len(df)
split = int(n * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\nTraining on {len(X_train)} days ({X_train.index[0].date()} to {X_train.index[-1].date()})")
print(f"Testing on {len(X_test)} days ({X_test.index[0].date()} to {X_test.index[-1].date()})")

# 7) Train Decision Tree with grid search
param_grid = {
    'max_depth': [2,3,4,5,6,7],
    'min_samples_leaf': [5,10,15,20],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=5, shuffle=False)
grid = GridSearchCV(dt, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
print("\nBest params:", grid.best_params_)
print(f"Best CV score: {grid.best_score_:.4f}")

# 8) Evaluate
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:,1]

print("\n" + "="*70)
print("CLASSIFICATION METRICS (TEST SET)")
print("="*70)
print(classification_report(y_test, y_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

if len(np.unique(y_test)) > 1:
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")

# 9) Feature importances
imp = pd.Series(best.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n" + "="*70)
print("FEATURE IMPORTANCES")
print("="*70)
print(imp)

# 10) Build open/exit lookup
def build_open_exit_lookup(df_intraday, exit_minutes=EXIT_MINUTES_AFTER_OPEN):
    lookup = {}
    for date, g in df_intraday.groupby('date'):
        g2 = g.copy()
        g2.index = pd.to_datetime(g2.index)
        market_open = g2.index[0].normalize() + pd.Timedelta(hours=9, minutes=30)
        exit_time = market_open + pd.Timedelta(minutes=exit_minutes)
        
        exit_row = g2[g2.index >= exit_time]
        if len(exit_row) > 0:
            exit_price = exit_row['Close'].iloc[0]
            open_series = g2[g2.index >= market_open]
            if len(open_series) > 0:
                entry_price = open_series['Open'].iloc[0]
                lookup[pd.to_datetime(date)] = {'entry': entry_price, 'exit': exit_price}
    return lookup

print("\nBuilding open->exit lookup...")
open_exit = build_open_exit_lookup(data, exit_minutes=EXIT_MINUTES_AFTER_OPEN)

# 11) Backtest on test set
trades = []
test_dates = X_test.index

for dt_index in test_dates:
    pred = best.predict(X.loc[[dt_index]])[0]
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
        'pred': pred,
        'actual': y.loc[dt_index],
        'entry': entry,
        'exit': exitp,
        'gross_ret': gross_ret,
        'net_ret': net_ret
    })

trades_df = pd.DataFrame(trades).set_index('day')

if trades_df.empty:
    print("\nNo trades could be simulated!")
else:
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    
    total_trades = len(trades_df)
    avg_net = trades_df['net_ret'].mean()
    median_net = trades_df['net_ret'].median()
    win_rate = (trades_df['net_ret'] > 0).mean()
    cum = (1 + trades_df['net_ret']).cumprod().iloc[-1]
    
    print(f"Total trades: {total_trades}")
    print(f"Average return: {avg_net:.6f} ({avg_net*100:.4f}%)")
    print(f"Median return: {median_net:.6f} ({median_net*100:.4f}%)")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Cumulative return: {(cum-1)*100:.2f}%")
    print(f"Final equity: ${1000*cum:.2f} (starting with $1000)")
    
    # Statistical significance tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE")
    print("="*70)
    
    # T-test
    t_stat, p_value = stats.ttest_1samp(trades_df['net_ret'], 0)
    print(f"T-test (H0: mean = 0):")
    print(f"  t-stat: {t_stat:.4f}, p-value: {p_value:.4f}")
    print(f"  Result: {'✓ SIGNIFICANT' if p_value < 0.05 else '✗ NOT significant'} at 5% level")
    
    # Sharpe ratio
    if trades_df['net_ret'].std() > 0:
        sharpe = (avg_net / trades_df['net_ret'].std()) * np.sqrt(252)
        print(f"\nSharpe Ratio: {sharpe:.4f}")
        print(f"  {'✓ Good' if sharpe > 1 else '✗ Needs improvement'} (>1.0 is good, >2.0 is excellent)")
    
    # Binomial test for win rate
    wins = (trades_df['net_ret'] > 0).sum()
    from scipy.stats import binomtest
    binom_result = binomtest(wins, total_trades, 0.5, alternative='greater')
    print(f"\nWin rate test (H0: 50%):")
    print(f"  Win rate: {win_rate:.2%}, p-value: {binom_result.pvalue:.4f}")
    print(f"  Result: {'✓ SIGNIFICANT' if binom_result.pvalue < 0.05 else '✗ NOT significant'}")
    
    # Max drawdown
    equity_curve = (1 + trades_df['net_ret']).cumprod()
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    print(f"\nMaximum Drawdown: {max_dd:.2%}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Equity curve
    axes[0, 0].plot(equity_curve.index, equity_curve.values, linewidth=2)
    axes[0, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Equity Curve', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Cumulative Return Factor')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Drawdown
    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    axes[0, 1].set_title(f'Drawdown (Max: {max_dd:.2%})', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Drawdown')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Return distribution
    axes[1, 0].hist(trades_df['net_ret'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].axvline(x=avg_net, color='g', linestyle='--', linewidth=2, label=f'Mean: {avg_net:.4f}')
    axes[1, 0].set_title('Return Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Return per Trade')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature importances
    imp.head(10).plot(kind='barh', ax=axes[1, 1], color='steelblue')
    axes[1, 1].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    print(f"\nSaved visualization to backtest_results.png")
    plt.show()
    
    # Save trades to CSV
    trades_df.to_csv('backtest_trades.csv')
    print(f"Saved trade log to backtest_trades.csv")
    
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    is_significant = p_value < 0.05
    good_sharpe = sharpe > 1.0 if trades_df['net_ret'].std() > 0 else False
    profitable = avg_net > 0
    
    if is_significant and good_sharpe and profitable:
        print("✓ Strategy shows PROMISING signs of edge")
        print("  Next steps: Test on different time periods, add risk management")
    elif profitable and (is_significant or good_sharpe):
        print("⚠ Strategy shows SOME promise but needs improvement")
        print("  Consider: Feature engineering, different exit timing, position sizing")
    else:
        print("✗ Strategy does NOT show significant edge")
        print("  Consider: Different features, different prediction target, other approaches")

# Save model
joblib.dump(best, "dt_overnight_model.pkl")
print(f"\nSaved model to dt_overnight_model.pkl")
