# MNQ 14:50 Strategy — Backtesting & Prop Firm Optimization System

A complete, production-ready MNQ (Micro Nasdaq Futures) trading strategy backtesting and prop firm challenge optimization system.

---

## Strategy Overview

The strategy trades the **14:50 ET candle** on MNQ:

| Parameter | Value |
|-----------|-------|
| Entry | Close of 14:50 candle |
| Direction | Follow candle (Bullish → LONG, Bearish → SHORT) |
| Exit | Close of 14:59 candle |
| Instrument | MNQ ($2/point) or NQ ($20/point) |

### Smart Filter Rules
- **Skip Thursdays** — historically weaker performance
- **Skip June & October** — seasonally adverse months
- **Skip candles with body < 3.0 points** — filter out indecision candles

### Key Results
- **86.7% prop firm pass rate** using Ensemble volatility prediction at 60% safety buffer
- **0.0% daily limit failures** with prediction-based position sizing
- ~9.2 average contracts per trade
- ~$86 average profit per trade
- Beats fixed 9-contract baseline of 59.1%

---

## Project Structure

```
mnq-1450-strategy/
├── config.py                 # Central configuration
├── core_strategy.py          # Main backtester
├── advanced_validation.py    # Statistical validation suite
├── volatility_predictor.py   # ML volatility prediction & position sizing
├── propfirm_optimizer.py     # Prop firm challenge optimizer
├── selection_bias.py         # Selection bias analysis (new)
├── run_all.py                # Single entry point CLI
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

---

## Usage

### Run Everything
```bash
python run_all.py --csv path/to/data.csv
```

### Run Specific Module
```bash
python run_all.py --csv data.csv --module core
python run_all.py --csv data.csv --module validation
python run_all.py --csv data.csv --module prediction
python run_all.py --csv data.csv --module propfirm
python run_all.py --csv data.csv --module bias        # selection bias analysis
```

### Custom Capital & Point Value
```bash
python run_all.py --csv data.csv --capital 25000 --point-value 20  # NQ
```

### Output Directory
```bash
python run_all.py --csv data.csv --output-dir my_results/
```

---

## CSV Data Format

| Column | Format | Description |
|--------|--------|-------------|
| `timestamp ET` | `MM/DD/YYYY HH:MM` | Bar timestamp in Eastern Time |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `volume` | int | Bar volume |
| `Vwap_RTH` | float (optional) | VWAP during regular trading hours |

- 1-minute bars
- ~1M+ rows covering multi-year history

---

## Module Descriptions

### `config.py`
Central configuration for all parameters:
- Initial capital, point values, entry/exit times
- Smart filter settings
- 25+ prop firm configurations (Topstep, Apex, MyFundedFutures, TradeDay, etc.)
- DD reduction levels (3%/6%/8%)
- Monte Carlo and walk-forward settings

### `core_strategy.py`
Main backtester with:
- **Three strategy directions**: Bullish Only, Bearish Only, Both (follow candle)
- **Buy & Hold benchmark** comparison
- **Full statistics**: Sharpe, Sortino, Calmar, Profit Factor, Max Drawdown ($ and %), CAGR, Risk of Ruin, streaks, and more
- **Side-by-side comparison table** for all 4 approaches
- **TradingView-style dark-theme charts** for last 5 trades
- **Trade log CSV export**

### `advanced_validation.py`
Statistical validation suite:
- **Monte Carlo Simulation** (10,000 paths): percentile distributions, probability of profit
- **Walk-Forward Analysis**: 5 windows, 60/40 train/test split
- **Pattern Detection**: Day-of-week, monthly, volatility regime, candle-size filter
- **Statistical Edge**: t-test, bootstrap 95% CI, skewness/kurtosis, runs test

### `volatility_predictor.py`
The KEY innovation — predicts the full 14:50-14:59 window range:
- 5 prediction models: EWMA, P75, Feature Scaling, Gradient Boosting, **Ensemble**
- Walk-forward ML (no lookahead bias)
- Positions sized as: `contracts = (daily_limit / (predicted_range × $2)) × safety_buffer`
- Safety buffer sweep (30%→100%)
- **Winning config: Ensemble + 60% buffer = 86.7% pass rate**

### `propfirm_optimizer.py`
- Contract sweep (1-50) for any prop firm
- Multi-firm sweep across 25+ firms
- Rankings by pass rate, ROI, cheapest to pass
- Expected cost to get funded
- 6-month expected value calculation
- Detailed top-3 analysis with multi-attempt probabilities

### `selection_bias.py`
Addresses the **multiple-comparisons / selection bias risk** introduced when
scanning 381 entry windows and keeping only the top performers.  Three
complementary methods:

- **Walk-forward cross-validation**: Divides trading days into 4 equal folds.
  In each fold, only prior data is used to identify the top-5 windows by
  Sharpe; those same windows are then evaluated out-of-sample in the next
  fold.  The IS→OOS Sharpe retention ratio quantifies how much of the
  apparent edge survives in unseen data.
- **Null distribution simulation**: Simulates 10,000 trials of 381 independent
  "fair-coin" windows (zero-mean PnL, ~225 pseudo-trades each) and records
  the best Sharpe and win-rate achievable by pure luck.  Answers: *"How
  likely is WR ≥ 56 % or Sharpe ≥ 2.5 by chance alone?"*
- **Shrinkage estimates**: Applies regression-to-the-mean shrinkage
  (`sf = 1 / (1 + √n_windows / √n_trades)`) to deflate backtested metrics
  toward a realistic forward expectation.  With 381 windows and 225 trades
  each, the shrinkage factor is ≈ 0.43 — only ~43 % of the apparent edge
  above the null baseline is expected to persist live.

> ⚠️ **Selection Bias Warning**: Picking the top result from 381 candidates
> will almost certainly yield at least one window with WR ≥ 56 % or
> Sharpe ≥ 2.5 even if no real edge exists.  Always interpret backtest
> rankings with the OOS walk-forward results and shrunk estimates in mind.

---

## ⚠️ Selection Bias Risk

When scanning a large number of candidate windows and keeping the best
performers, some results will appear strong purely due to random variation —
this is the **multiple-comparisons problem** (also called *data dredging* or
*p-hacking*).

### Why it matters for this strategy

| Scenario | Probability |
|----------|-------------|
| At least one of 381 windows has WR ≥ 56% by chance | ~100% |
| At least one of 381 windows has Sharpe ≥ 2.5 by chance | ~95%+ |

These probabilities are derived from the null-distribution simulation in
`selection_bias.py`.  The key takeaway: **the observed rankings are
expected rankings under pure chance** — they do not by themselves prove
a real edge.

### How to mitigate it

1. **Walk-forward cross-validation** — if OOS performance closely tracks IS
   performance across all folds, the edge is likely real.  A low
   IS→OOS Sharpe retention ratio (< 50%) is a red flag.

2. **Shrinkage** — apply the conservative shrunk estimates as your planning
   baseline rather than the raw backtest numbers.

3. **Extended out-of-sample period** — paper-trade or forward-test for at
   least 3–6 months before committing real capital.

4. **Consistency across sub-periods** — a genuine edge should appear in
   most calendar years and market regimes, not just in aggregate.

---

## Daily Trading Playbook

1. Check smart filters: Is today Thursday? Is it June or October?
2. At 14:50 ET, observe the 1-minute candle
3. Measure candle body: if < 3.0 points, **skip**
4. Get today's predicted window range from volatility model
5. Calculate contracts: `(daily_limit / (predicted_range × $2)) × 60%`
6. Enter at the **close** of the 14:50 candle in the candle's direction
7. Exit at the **close** of the 14:59 candle
8. Monitor drawdown: reduce contracts at 3%/6%/8% DD from peak

---

## Position Sizing Formula

```
contracts = floor( (daily_loss_limit / (predicted_window_range × point_value)) × safety_buffer )
contracts = clip(contracts, min=1, max=50)
```

Also calculate from predicted MAE — use the **more conservative (minimum)** of the two.

**Example** (Topstep $50K, daily limit $1,000, predicted range 32 pts, 60% buffer):
```
contracts = floor(1000 / (32 × 2) × 0.60) = floor(9.375) = 9 contracts
```

---

## Risk Management Rules

### DD Reduction
| Drawdown from Peak | Action |
|--------------------|--------|
| 3% | Cut contracts by 50% |
| 6% | Cut contracts by 75% |
| 8% | Exit entirely (no more trades today) |

### Daily Loss Limit
- Never risk more than prop firm's daily loss limit in a single trade
- Prediction-based sizing ensures 0% daily limit failures

---

## Max Drawdown Calculation

```python
# Dollar DD (negative)
dollar_dd = equity - peak

# Percentage DD
pct_dd = (equity - peak) / peak * 100
```

> ⚠️ Do NOT divide by initial capital — divide by the running peak.

---

## Prop Firm Configurations

| Firm | Account | Profit Target | Max DD | Daily Limit | Fee |
|------|---------|---------------|--------|-------------|-----|
| Topstep $50K | $50,000 | $3,000 | $2,000 | $1,000 | $165 |
| Topstep $100K | $100,000 | $6,000 | $3,000 | $2,000 | $325 |
| Apex $50K | $50,000 | $3,000 | $2,500 | $1,000 | $187 |
| Apex $100K | $100,000 | $6,000 | $3,000 | $2,000 | $247 |
| … | … | … | … | … | … |

Full list in `config.py`.

---

## Feature Importance (Gradient Boosting)

| Feature | Importance |
|---------|-----------|
| `last30_avg_range` | 43.6% |
| `last10_avg_range` | 13.3% |
| `last10_total_range` | 6.2% |
| Other features | 36.9% |

The 30-bar lookback average range dominates — recent volatility is highly predictive of the 14:50-14:59 window range.

---

## Critical Insight: Entry Candle vs Window Range

| Metric | Value |
|--------|-------|
| Entry candle (14:50) average range | ~9 pts |
| Full window (14:50-14:59) average range | ~32 pts |
| Multiplier | 3.5× |

Previous versions sized positions on the entry candle range, causing frequent daily limit breaches. This system sizes on the **predicted full window range**, eliminating that failure mode.

---

## License

MIT License — see repository for details.

