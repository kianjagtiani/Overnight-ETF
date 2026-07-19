# Overnight ETF Returns

QuantSC · Single-Stock & ETF Desk, Project #2 \
**Desk Head:** Kian Jagtiani

---

Most of the equity premium historically shows up overnight rather than intraday. This project asks whether that overnight hold can be timed with a model instead of always taken, using end-of-day signals to decide when to carry a position from the close to the next open.

## What the work included

- Six months of 5-minute bars from Polygon for six liquid ETFs (SPY, QQQ, IWM, XLF, XLE, GLD).
- End-of-day candle features: direction, magnitude, and volume over the last 10, 20, 30, and 60 minutes before the close.
- A grid-searched, cross-validated decision-tree classifier that predicts the sign of the overnight move.
- A backtest that enters at the close on an up prediction and exits five minutes after the next open, with slippage modeled per side.
- A multi-asset harness comparing trade count, win rate, Sharpe, and cumulative return across all six ETFs, with the fitted model, trade logs, and figures saved.

## Repository contents

- `polygonCSV.py`: pulls 5-minute bars from Polygon into the `*_5min_polygon_6months.csv` files.
- `decision_tree_overnight.py`: single-asset feature build, tree fit, and backtest; saves `dt_overnight_model.pkl` and `backtest_trades.csv`.
- `multi_asset_backtest.py`: runs the pipeline across all six ETFs into `multi_asset_summary.csv` and the comparison figures.
