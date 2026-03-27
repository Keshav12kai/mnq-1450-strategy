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
├── colab_v2_best_strategy.py      # v2 best strategy — Colab & local runner
├── run_all.py                     # Single entry point CLI
├── requirements.txt               # Python dependencies
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

# v2 best strategy with default prop firm parameters
python run_all.py --csv data.csv --module v2

# v2 with custom prop firm parameters
python run_all.py --csv data.csv --module v2 \
    --daily-limit 1000 --target 3000 --max-dd 2000 --days 30 --point-value 2
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

## Colab Quickstart (v2 best)

Run the full v2 best strategy end-to-end in Google Colab — no local Python setup needed.

### Step 1 — Open a new Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com) → **File → New notebook**.

### Step 2 — Mount Google Drive (recommended for large CSVs)

```python
from google.colab import drive
drive.mount('/content/drive')
```

> **Alternative — direct upload:**
> ```python
> from google.colab import files
> uploaded = files.upload()   # pick your CSV
> ```

### Step 3 — Install dependencies

```python
!pip install -q pandas numpy matplotlib scikit-learn scipy
```

### Step 4 — Upload `colab_v2_best_strategy.py` to Colab

Either upload the file via the sidebar (**Files → Upload**), or fetch it directly:

```python
!wget -q https://raw.githubusercontent.com/Keshav12kai/mnq-1450-strategy/main/colab_v2_best_strategy.py
```

### Step 5 — Set your parameters and run

```python
# ── Edit these to match your prop firm & instrument ─────────────────────────
CSV_PATH       = "/content/drive/MyDrive/Dataset_NQ_1min_2022_2025.csv"
POINT_VALUE    = 2.0       # MNQ = $2/pt  |  NQ = $20/pt
DAILY_LIMIT    = 1_000.0   # prop firm daily loss limit
PROFIT_TARGET  = 3_000.0   # prop firm profit target
MAX_DRAWDOWN   = 2_000.0   # prop firm max drawdown
CHALLENGE_DAYS = 30        # challenge window in trading days

import importlib, colab_v2_best_strategy as v2
importlib.reload(v2)       # picks up any parameter edits above

# Patch parameters, then run
v2.CSV_PATH       = CSV_PATH
v2.POINT_VALUE    = POINT_VALUE
v2.DAILY_LIMIT    = DAILY_LIMIT
v2.PROFIT_TARGET  = PROFIT_TARGET
v2.MAX_DRAWDOWN   = MAX_DRAWDOWN
v2.CHALLENGE_DAYS = CHALLENGE_DAYS

v2.run(
    csv_path       = CSV_PATH,
    point_value    = POINT_VALUE,
    daily_limit    = DAILY_LIMIT,
    profit_target  = PROFIT_TARGET,
    max_drawdown   = MAX_DRAWDOWN,
    challenge_days = CHALLENGE_DAYS,
)
```

### v2 CLI Parameters (local usage)

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | `data.csv` | Path to 1-minute bar CSV |
| `--daily-limit` | `1000` | Prop firm daily loss limit ($) |
| `--target` | `3000` | Prop firm profit target ($) |
| `--max-dd` | `2000` | Prop firm max drawdown ($) |
| `--days` | `30` | Prop firm challenge days |
| `--point-value` | `2` | $/point  (MNQ=2, NQ=20) |
| `--capital` | `10000` | Initial capital for equity curve |
| `--output-dir` | `outputs_v2` | Directory for charts & logs |

**Example — MNQ Topstep $50K challenge:**
```bash
python colab_v2_best_strategy.py \
    --csv Dataset_NQ_1min_2022_2025.csv \
    --point-value 2 \
    --daily-limit 1000 \
    --target 3000 \
    --max-dd 2000 \
    --days 30
```

**Or via `run_all.py`:**
```bash
python run_all.py --csv data.csv --module v2 \
    --daily-limit 1000 --target 3000 --max-dd 2000 --days 30 --point-value 2
```

### v2 Outputs

| File | Description |
|------|-------------|
| `outputs_v2/v2_equity_comparison.png` | Dynamic sizing vs fixed-9-contract baseline |
| `outputs_v2/v2_pass_rate_sweep.png` | Pass rate vs safety buffer (30 %–100 %) |
| `outputs_v2/v2_prediction_scatter.png` | Predicted vs actual window_range & MAE |
| `outputs_v2/v2_model_mae.png` | Model accuracy comparison (MAE) |
| `outputs_v2/v2_trade_log.csv` | Full per-trade log with dynamic & baseline contracts |

---

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

### `colab_v2_best_strategy.py`
The v2 best strategy — a **fully self-contained** Colab & local runner that:
- Builds **v2 targets**: `window_range` (14:50-14:59 high-low) and `MAE` (max adverse excursion)
- Engineers **40+ features** from strictly pre-14:50 data (no lookahead)
- Runs **5 prediction models**: EWMA, P75, Feature Scaling, walk-forward Gradient Boosting, **Ensemble**
- Sizes contracts as `min(range-based, mae-based) × safety_buffer`
- **Safety buffer sweep** (30 %–100 %) with prop-firm Monte Carlo per buffer
- Backtests dynamic sizing vs **baseline fixed 9 MNQ** with identical smart filter
- Outputs: equity comparison, pass-rate sweep, prediction scatter, model MAE, trade log CSV

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

