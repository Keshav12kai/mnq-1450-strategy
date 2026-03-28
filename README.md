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
├── config.py                      # Central configuration
├── core_strategy.py               # Main backtester
├── advanced_validation.py         # Statistical validation suite
├── volatility_predictor.py        # ML volatility prediction & position sizing
├── propfirm_optimizer.py          # Prop firm challenge optimizer
├── selection_bias_analysis.py     # Selection bias mitigation & forward expectations
├── run_all.py                     # Single entry point CLI
├── requirements.txt               # Python dependencies
├── SELECTION_BIAS_ANALYSIS.md     # Selection bias methodology & realistic expectations
└── README.md                      # This file
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

### `run_all.py`
Single CLI entry point:
```bash
python run_all.py --csv data.csv [--module all|core|validation|prediction|propfirm|selection_bias]
```

---

## Selection Bias & Realistic Expectations

See **[SELECTION_BIAS_ANALYSIS.md](SELECTION_BIAS_ANALYSIS.md)** for the full analysis, including:
- Identified bias risks (time-window mining, filter tuning, parameter sweeps)
- Mitigation techniques (structural market rationale, IS/OOS split, walk-forward ML, sensitivity analysis)
- Realistic forward-performance ranges (live Sharpe expected 0.7–1.5× backtest)
- Prop-firm pass rate adjusted for OOS degradation: **55–75%** (vs 86.7% IS)

To run the analysis and generate a results table with your actual data:
```bash
python run_all.py --csv Dataset_NQ_1min_2022_2025.csv --module selection_bias
```

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

### `selection_bias_analysis.py`
Dedicated selection bias mitigation and forward-performance estimation:
- **Chronological 70/30 IS/OOS split** — honest out-of-sample performance estimate
- **Parameter sensitivity** — candle-body threshold, Thursday filter, month filters
- **Temporal stability** — per-year Sharpe to detect regime dependence
- **Filter contribution** — how much each filter genuinely adds
- **Forward expectation** — OOS + temporal-stability + regime haircuts
- **Auto-generated markdown report** (`output/SELECTION_BIAS_RESULTS.md`)

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

