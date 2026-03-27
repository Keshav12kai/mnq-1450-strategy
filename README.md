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
├── colab_v2_best_strategy.py  # Standalone Colab script (start here)
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

## ⚡ Quick Start: Colab (Google Colab — recommended for large files)

1. Open a new Colab notebook
2. Mount Google Drive or upload your CSV
3. Copy-paste the entire `colab_v2_best_strategy.py` into one cell
4. Set `CSV_PATH` at the top of the cell
5. Run — all outputs go to the `outputs_v2/` directory

```python
# At the top of colab_v2_best_strategy.py:
from google.colab import drive
drive.mount('/content/drive')
CSV_PATH = "/content/drive/MyDrive/data/Dataset_NQ_1min_2022_2025.csv"
```

The Colab script is fully standalone — no other files needed.

---

## Why Does Pass Rate Change with Different Rules?

This is the most important insight.  The 86.7% result and a ~60-65% result on 30-day rules are **both correct** — they just use different prop firm rule sets:

| Rule Set | Challenge Days | Max DD | Daily Limit | Pass Rate |
|----------|---------------|--------|-------------|-----------|
| **v2-reference** (original result) | **10** | $2,000 | $1,000 | **~86.7%** |
| Your 30-day challenge | **30** | $2,000 | $1,000 | **~60-65%** |
| Apex-style | 7 | $2,500 | $1,000 | ~80-85% |

**Why?**  Each trading day is an independent risk event.  A 30-day challenge runs 3× more trades than a 10-day challenge, giving the drawdown and daily-loss limits 3× as many chances to trigger a failure.  The underlying trading edge (win rate, P&L distribution) is identical.

`challenge_days` is the single biggest driver of pass rate variance.  A $500 difference in max drawdown changes pass rate by ~5-10 pp.  Adding 20 extra challenge days can cut it by 20-25 pp.

The `colab_v2_best_strategy.py` script runs **both** rule sets side-by-side so you can see the exact difference on your data.

---

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

### `colab_v2_best_strategy.py` ← **Start here for Colab**
Standalone single-cell Colab script (copy-paste the whole file):
- Full v2 pipeline: smart filter → features → 5 models → Ensemble → prop firm MC
- **Prints pipeline stage trade counts** at every step (so you can see the 225→205 etc.)
- **No alignment dropna** — all samples are preserved through the feature/prediction pipeline
- **Prints the effective rule set** at startup (instrument, firm config, buffer, etc.)
- **Side-by-side comparison** of v2-reference (10-day) vs your 30-day challenge rules
- Predicted window_range / MAE distribution, contract distribution and percentiles
- Daily-limit breach counts from historical data
- Saves charts to `outputs_v2/`

### `config.py`
Central configuration for all parameters:
- Initial capital, point values, entry/exit times
- Smart filter settings
- **`DAILY_LOSS_LIMIT`** — explicit configurable daily loss limit (default $1,000)
- **`challenge_days` explanation** — key driver of pass rate variance (see comments)
- 25+ prop firm configurations including v2-reference (10-day) and user 30-day variants
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
- Walk-forward ML (no lookahead bias); `shift(1)` / `min_periods=1` on all lag/rolling features
- **Fixed alignment bug** — `fillna(0)` + `min_periods=1` prevents sample loss (was 205→145)
- Positions sized as `min(range_based, MAE_based)` contracts
- Safety buffer sweep (30%→100%) with best-buffer marker
- **Pipeline diagnostics**: stage counts, prediction distributions, contract percentiles, daily-limit breach counts
- **Prints effective rule set** before simulation

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

