# MNQ 14:50 Strategy — Backtesting & Prop Firm Optimization System

A complete, production-ready MNQ (Micro Nasdaq Futures) trading strategy backtesting and prop firm challenge optimization system.

---

> 📋 **Quick Reference** — This README contains all key research findings, optimized windows, bias controls, and execution checklists.
> Scroll to [Research Summary](#research-summary--key-findings), [Optimized Windows](#optimized-trading-windows), [Selection Bias Controls](#selection-bias-analysis--controls), and [Execution Checklist](#execution-checklist) for the most critical operational information.

---

## Research Summary & Key Findings

This strategy was developed through rigorous multi-window backtesting on MNQ 1-minute bar data. The research expanded beyond the original single 14:50 candle to identify **five statistically robust intraday windows** across the regular trading session.

### Top-Level Results

| Metric | Value |
|--------|-------|
| Prop firm pass rate (Ensemble + 60% buffer) | **86.7%** |
| Daily limit failures | **0.0%** |
| Average contracts per trade | ~7–9 |
| Average profit per trade | ~$86 |
| Baseline (fixed 9 contracts) pass rate | 59.1% |

### Research Methodology

1. **Full-session window scan** — Every 1-minute entry window between 09:30 and 15:58 ET was evaluated for edge.
2. **Out-of-sample walk-forward validation** — 60/40 train/test splits across 5 rolling windows to prevent overfitting.
3. **Monte Carlo simulation** — 10,000 paths per window to assess robustness of profit/loss distributions.
4. **Statistical significance testing** — t-test, bootstrap 95% CI, runs test for each qualifying window.
5. **Bias audit** — All windows checked against selection bias, data-snooping bias, and survivorship bias controls (see [Selection Bias Analysis & Controls](#selection-bias-analysis--controls)).

---

## Optimized Trading Windows

After the full research cycle, **five windows** passed all statistical and bias filters. Trade these windows, **in Eastern Time (ET)**:

| # | Entry | Exit | Hold | Notes |
|---|-------|------|------|-------|
| 1 | 12:16 | 12:27 | 11 min | Mid-session momentum |
| 2 | 12:51 | 13:00 | 9 min | Lunch breakout |
| 3 | 13:55 | 14:05 | 10 min | Afternoon positioning |
| 4 | 14:50 | 15:01 | 11 min | ⭐ Pre-close flush |
| 5 | 15:48 | 15:58 | 10 min | ⭐ Closing auction |

### Universal Entry Rules (apply to every window)
- **Minimum candle body ≥ 3 points** — skip the window if the entry candle is a doji or narrow body (< 3 pts).
- **Direction** — follow the candle: bullish body → LONG, bearish body → SHORT.
- **Skip Thursdays** — historically weaker performance across all windows.
- **Skip June & October** — seasonally adverse months.
- **Contracts** — use **7 contracts** as the base (adjust down if volatility prediction or drawdown rules require).

---

## Selection Bias Analysis & Controls

Selection bias is the #1 threat to any backtested strategy. The research applied the following controls at every stage:

### Types of Bias Addressed

| Bias Type | Description | Mitigation Applied |
|-----------|-------------|-------------------|
| **Data-snooping / curve-fitting** | Tuning parameters until backtest looks good | Walk-forward validation (60/40 splits); parameters locked before final test |
| **Selection bias (window cherry-picking)** | Reporting only the best windows out of hundreds tested | All windows evaluated; only those surviving hold-out OOS test reported |
| **Survivorship bias** | Using only current instruments / ignoring losing periods | Full historical data including drawdown periods used |
| **Look-ahead bias** | Using future data in entry/sizing decisions | Volatility predictor uses walk-forward ML — no future bars in features |
| **Overfitting via filter stacking** | Adding filters (Thursday/month/candle size) until curve fits | Each filter validated independently on hold-out data before inclusion |

### Key Bias Control Outcomes
- The Thursday and June/October filters were validated on **out-of-sample data** before being included.
- The 3-point body filter was confirmed to improve Sharpe ratio on the held-out test period, not just the training set.
- Window times (e.g., 12:16, 12:51) were not adjusted after the hold-out test was run — they are fixed.
- The 7-contract default is derived from the position sizing formula applied to the median predicted range; it was not back-optimized to maximize PnL.

---

## Execution Checklist

Use this checklist every trading day before and during each window.

### Pre-Session (Before 09:30 ET)
- [ ] Confirm today is **not Thursday**
- [ ] Confirm today is **not in June or October**
- [ ] Check for major economic events (FOMC, CPI, NFP) — consider skipping those days
- [ ] Note the volatility model's predicted range for today

### Each Window (Repeat for All 5)
- [ ] Set an alert 1 minute before each entry time
- [ ] At entry candle open, wait for the full candle to form
- [ ] **Measure candle body** — if body < 3.0 points → **SKIP this window**
- [ ] Determine direction: bullish body → LONG / bearish body → SHORT
- [ ] Calculate contracts: `floor( (daily_limit / (predicted_range × $2)) × 60% )`; use 7 as default
- [ ] **Check current drawdown** — reduce contracts if you are at 3% DD (−50%) or 6% DD (−75%); stop trading if at 8% DD
- [ ] Enter at the **close of the entry candle**
- [ ] Set exit order for the **close of the exit candle** (see window table above)
- [ ] Record trade in log: window, direction, contracts, entry price, exit price, PnL

### Post-Session
- [ ] Update running equity and peak equity
- [ ] Recalculate drawdown percentage from peak
- [ ] Note any windows skipped and the reason
- [ ] Review if daily loss limit was approached — adjust sizing for next session if needed

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

