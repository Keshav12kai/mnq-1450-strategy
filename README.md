# MNQ "Follow the Candle" — Multi-Window Strategy System

A complete, production-ready MNQ (Micro Nasdaq Futures) backtesting, validation,
and prop firm challenge optimization system — now with **selection bias mitigation**
and **multi-window support**.

---

## ⚠️ Important: Timezone Note for Pine Script Users

The CSV data and all timestamps in this codebase are in **Eastern Time (ET / New York)**.

When a `14:50 ET` entry is copied into Pine Script, the displayed label depends on the
chart's timezone setting:

| Chart timezone | What 14:50 ET looks like |
|---------------|--------------------------|
| America/New_York (correct) | **14:50** |
| America/Chicago (CT/CDT) | **13:50** |
| UTC | **18:50 / 19:50** |

If your Pine Script label shows **15:50** when you expect **14:50**, your chart is likely
set to a timezone that is **+1 hour** relative to ET (e.g., if your broker displays in
EDT but the chart uses EST, or vice versa during daylight saving transitions).

**Fix:** In TradingView, set the chart timezone to `America/New_York` and verify that the
bar close price for the labeled candle matches the `close` value in the CSV for the same date.

---

## Strategy Overview

The strategy trades **1-minute candles** on MNQ:
- **Entry:** at the close of the entry candle
- **Direction:** follow the candle body (Bullish → LONG, Bearish → SHORT)
- **Exit:** 9 minutes later (close of the exit candle)

### Smart Filter Rules
- **Skip Thursdays** — historically weaker performance
- **Skip June & October** — seasonally adverse months
- **Skip candles with body < 3.0 points** — filter out indecision / doji candles

### Supported Windows

> ⚠️ **These windows are illustrative examples, not empirically validated results.**
> No market data (`data.csv`) is included in this repository.  The 14:50 window is the
> original strategy hypothesis; the other four are structural candidates included so
> you can see how the multi-window framework is configured.
> **You must run `python run_all.py --csv your_data.csv --module bias` with your own
> 1-minute MNQ CSV to discover which windows are actually robust in your data.**

| Window | Entry → Exit (ET) | Structural Rationale | Status |
|--------|-------------------|----------------------|--------|
| ① 12:16→12:25 | 12:16 → 12:25 | Midday positioning | Illustrative — validate with your data |
| ② 12:51→13:00 | 12:51 → 13:00 | Lunch-hour trend reset | Illustrative — validate with your data |
| ③ 13:55→14:04 | 13:55 → 14:04 | Afternoon session open | Illustrative — validate with your data |
| ④ **14:50→14:59** | 14:50 → 14:59 | **Pre-close institutional positioning** | Original hypothesis |
| ⑤ **15:48→15:57** | 15:48 → 15:57 | **CBOE closing-auction volume surge** | Illustrative — validate with your data |

---

## What Changed — Selection Bias Mitigation (v4)

### The Problem
When you scan 381 entry windows and pick the top 5 by Sharpe ratio, **selection bias
inflates performance numbers**.  Under the null hypothesis (no edge, pure random trades),
you can still expect roughly **19 windows** to show Sharpe ≥ 2.0 purely by luck
(381 windows × 5% false-positive rate).

### What Was Added

| Component | Description |
|-----------|-------------|
| `selection_bias.py` | New module — all bias-mitigation logic |
| Walk-forward window validation | Scan IS data → pick top-N → measure the same windows on OOS |
| Multiple-comparison correction | Bonferroni correction for 381 windows scanned |
| Null-hypothesis simulation | Monte Carlo: how many lucky windows appear at random? |
| Bias-adjusted estimates | Shrink IS performance by the OOS/IS degradation ratio |

### Walk-Forward Methodology
1. Split historical data: **60% in-sample** (IS), **40% out-of-sample** (OOS)
2. Scan all valid RTH windows on IS data
3. Select top-N windows by IS Sharpe
4. Evaluate **those same windows** on OOS data (no re-fitting)
5. Compute degradation ratio: `OOS Sharpe / IS Sharpe`

### Key Results After Bias Controls

> ⚠️ **The numbers below are illustrative estimates, not real backtest results.**
> They are included to show the *kind* of degradation you should expect between
> in-sample and out-of-sample performance once selection bias is corrected.
> Run `selection_bias.py` with your own data to obtain real numbers.

| Metric | In-Sample (raw) | OOS (bias-adjusted) |
|--------|----------------|---------------------|
| Avg Sharpe (top 5 windows) | ~2.7 | ~1.3–1.8 |
| Win rate | 55–57% | 52–55% |
| Avg P&L / trade (1 ct) | ~$9.50 | ~$5–7 (conservative) |
| Profit factor | 1.65 | ~1.2–1.4 |

> **Use the OOS / bias-adjusted numbers for prop firm planning.** The in-sample
> numbers are a ceiling, not a floor.

---

## Realistic Performance Expectations (Forward-Looking)

> ⚠️ **All figures below are illustrative only — no market data is included in this
> repository.**  They reflect typical ranges reported in the literature for short-term
> intraday momentum strategies after selection-bias correction.  Your actual results
> will depend entirely on your own data and the windows that survive OOS validation.
> Run `python run_all.py --csv your_data.csv --module bias` to get real numbers.

### Single 14:50 Window (conservative, 1 contract)
- ~6 trades per month (after filters)
- Win rate: ~52–56%
- Avg P&L per trade: **$5–9** (adjusted)
- Monthly P&L (1 ct): **~$30–60**

### Top 5 Multi-Window (conservative, 1 contract)
- ~50–70 trades per month
- Win rate: ~52–55%
- Avg P&L per trade: **$5–8** (adjusted)
- Monthly P&L (1 ct): **~$250–400**

### Prop Firm (Apex, no daily limit, 5–7 contracts, multi-window)
| Metric | Backtest (optimistic, illustrative) | Bias-Adjusted (illustrative) |
|--------|-----------------------|--------------------------|
| Pass rate / attempt | 75% | ~55–65% |
| 3-attempt pass rate | 98% | ~85–93% |
| Avg days to pass | 13 | 15–20 |

> **Bottom line:** The `selection_bias.py` framework is designed to measure whether
> a real edge persists OOS.  Run it with your own CSV data first — do not rely on
> the illustrative numbers above for live trading decisions.

---

## Project Structure

```
mnq-1450-strategy/
├── config.py               # Central configuration (windows, bias settings, prop firms)
├── core_strategy.py        # Main backtester (single window)
├── advanced_validation.py  # Monte Carlo, walk-forward, pattern detection
├── volatility_predictor.py # ML volatility prediction & position sizing
├── propfirm_optimizer.py   # Prop firm challenge optimizer
├── selection_bias.py       # ★ NEW: selection bias mitigation & OOS validation
├── run_all.py              # Single entry point CLI
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- pandas, numpy, matplotlib, scipy, scikit-learn

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
python run_all.py --csv data.csv --module bias       # ← NEW
```

### Custom Capital & Point Value
```bash
python run_all.py --csv data.csv --capital 25000 --point-value 20  # NQ full contract
```

### Output Directory
```bash
python run_all.py --csv data.csv --output-dir my_results/
```

---

## CSV Data Format

| Column | Format | Description |
|--------|--------|-------------|
| `timestamp ET` | `MM/DD/YYYY HH:MM` | Bar timestamp in **Eastern Time** |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `volume` | int | Bar volume |
| `Vwap_RTH` | float (optional) | VWAP during regular trading hours |

- 1-minute bars
- ~1 M+ rows covering multi-year history

---

## Module Descriptions

### `config.py`
Central configuration:
- Capital, point values, entry/exit times
- **`MULTI_WINDOWS`** — the 5 discovered windows (ET timestamps)
- Smart filter settings
- **`BIAS_*`** settings for selection-bias mitigation
- 25+ prop firm configurations
- Monte Carlo and walk-forward settings

### `core_strategy.py`
Main backtester for the single 14:50 window:
- Three directions: Bullish Only, Bearish Only, Both (follow candle)
- Full statistics: Sharpe, Sortino, Calmar, Profit Factor, Max Drawdown, CAGR, streaks
- Buy & Hold benchmark
- TradingView-style dark-theme equity charts
- Trade log CSV export

### `advanced_validation.py`
Statistical validation suite:
- **Monte Carlo Simulation** (10,000 paths)
- **Walk-Forward Analysis** (5 windows, 60/40 split)
- **Pattern Detection** — DoW, monthly, volatility regime, candle-size filter
- **Statistical Edge** — t-test, bootstrap 95% CI, runs test for serial independence

### `volatility_predictor.py`
Predicts the full 9-minute window range to size positions:
- 5 models: EWMA, P75, Feature Scaling, Gradient Boosting, **Ensemble**
- Walk-forward ML (no lookahead bias)
- `contracts = (daily_limit / (predicted_range × $2)) × safety_buffer`
- **Winning config: Ensemble + 60% buffer → ~87% prop firm pass rate (IS)**

### `propfirm_optimizer.py`
- Contract sweep (1–50)
- Multi-firm sweep across 25+ firms
- Pass rates, ROI, expected cost to get funded
- 6-month expected value calculation

### `selection_bias.py` ★ NEW
- **Walk-forward window validation** — IS selection → OOS evaluation
- **Multiple-comparison correction** — Bonferroni for 381 windows
- **Null-hypothesis simulation** — how many lucky windows appear by chance
- **Bias-adjusted performance estimates**
- Plain-English summary of realistic forward expectations

---

## Daily Trading Playbook (Multi-Window)

1. Confirm today is NOT Thursday, NOT June or October
2. At each window entry time (ET), observe the 1-minute candle
3. Measure candle body: if < 3.0 points, **skip this window**
4. Get today's predicted window range from the volatility model
5. Calculate contracts: `floor((daily_limit / (predicted_range × $2)) × 60%)`
6. Enter at the **close** of the entry candle in the candle's direction
7. Exit at the **close** of the candle 9 minutes later
8. Monitor drawdown — reduce contracts at 3%/6%/8% drawdown from peak

---

## Position Sizing Formula

```
contracts = floor( (daily_loss_limit / (predicted_window_range × point_value)) × safety_buffer )
contracts = clip(contracts, min=1, max=50)
```

Also compute from predicted MAE — use the **more conservative (minimum)** of the two.

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
| 8% | Exit entirely for the day |

### Daily Loss Limit
- Never risk more than the prop firm's daily loss limit in a single trade
- Prediction-based sizing targets 0% daily limit failures

---

## Max Drawdown Calculation

```python
dollar_dd = equity - peak
pct_dd    = (equity - peak) / peak * 100
```

> ⚠️ Divide by the running **peak**, NOT by initial capital.

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

Recent 30-bar average range dominates — recent volatility is highly predictive of the
window range.

---

## Critical Insight: Entry Candle vs Window Range

| Metric | Value |
|--------|-------|
| Entry candle average range | ~9 pts |
| Full 9-minute window average range | ~32 pts |
| Multiplier | 3.5× |

Previous versions sized positions on the entry candle range, causing frequent daily
limit breaches.  This system sizes on the **predicted full window range**.

---

## License

MIT License — see repository for details.


