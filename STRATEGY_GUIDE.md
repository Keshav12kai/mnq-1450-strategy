# MNQ Multi-Window Strategy — Complete Reference Guide

> **Purpose:** This document is the permanent, long-term reference for the MNQ multi-window
> trading strategy. It captures every key research finding, design decision, bias-control
> method, operational rule, and checklist produced during development so that anyone (including
> future you) can reproduce and maintain the strategy without losing context.

---

## Table of Contents

1. [Strategy Summary & Rationale](#1-strategy-summary--rationale)
2. [Selection Bias — Explanation & Mitigation](#2-selection-bias--explanation--mitigation)
3. [The 5 Optimized Time Windows](#3-the-5-optimized-time-windows)
4. [Per-Trade Entry & Exit Rules](#4-per-trade-entry--exit-rules)
5. [Prop Firm Results — Pass Rates & Sizing](#5-prop-firm-results--pass-rates--sizing)
6. [Daily Execution Checklist](#6-daily-execution-checklist)
7. [Monitoring Edge Decay](#7-monitoring-edge-decay)
8. [Periodic Walk-Forward Retesting](#8-periodic-walk-forward-retesting)
9. [Out-of-Sample Cautions](#9-out-of-sample-cautions)
10. [Layman / Operational Quick-Reference](#10-layman--operational-quick-reference)

---

## 1. Strategy Summary & Rationale

### What the Strategy Does

The strategy trades **MNQ (Micro Nasdaq Futures)** during short intraday windows identified as
statistically and structurally significant. Rather than trading a single time slot, the
strategy uses **five complementary windows** spread across the trading day, each lasting
9–11 minutes.

### Why MNQ?

| Reason | Detail |
|--------|--------|
| Low margin | ~$50 per contract, accessible for underfunded prop-firm accounts |
| Tight spreads | 0.25-pt tick size; liquid during RTH |
| Volatility | Sufficient intraday range for repeatable edge |
| Scalability | Easy to increase/decrease size by contract count |

### Core Thesis

Nasdaq futures exhibit **repeatable directional bursts** at predictable times of day driven
by order-flow events: institutional re-balancing, index auction prep, and end-of-day
momentum. By entering at the close of the initiating 1-minute candle (filtering out doji
noise) and exiting within the same short window, the strategy captures these bursts
without overnight exposure.

The **14:50 → 15:01** and **15:48 → 15:58** windows carry the highest conviction because
they align with the well-documented *pre-close institutional re-balancing* and
*closing auction* phases respectively.

---

## 2. Selection Bias — Explanation & Mitigation

### What Is Selection Bias in Backtesting?

When you scan hundreds of possible time windows and then report the best-performing ones,
some of those "winners" are winning purely by chance — not because of a real market edge.
This is **selection bias** (also called "data snooping" or "multiple comparisons problem").

**Analogy:** If you flip 381 coins and celebrate the ones that landed heads 58% of the
time, you've found nothing — with enough coins, a few will always look special by luck.

### How It Was Mitigated

Three techniques were applied:

#### 2a. Walk-Forward (Out-of-Sample) Validation

The full data history was split chronologically:

```
|──────── Train (60%) ────────|──── Test (40%) ────|
  Window selected here           Tested here only
```

- Windows were selected and ranked **only** on the training portion.
- Performance was then measured on the unseen test portion.
- A window that survives this test has a genuine edge, not a lucky in-sample fit.

#### 2b. Random Window Simulation (Luck Benchmark)

Hundreds of randomly-chosen time windows were simulated to establish a **luck baseline**:
what Sharpe ratio, win rate, or profit factor you'd expect to see by pure chance. Any
selected window that does NOT clearly beat the random baseline is discarded.

This answered the question: *"How much of the edge is real vs. how much would any random
window produce?"*

#### 2c. Edge Shrinkage (Conservative Live Expectations)

Even after bias controls, the live Sharpe/win rate will be **somewhat lower** than the
in-sample numbers, because:

- The future is never a perfect repeat of the past.
- Execution adds real-world friction (slippage, missed fills).
- Markets subtly evolve.

**Rule of thumb applied:** Expect **10–30% lower** profit/win rate in live trading compared
to the best in-sample backtest figures. Position sizing and pass-rate calculations use
these *shrunk* (conservative) expectations.

### Outcome of Bias Controls

Validated results from the full 2022-12-26 → 2025-12-11 dataset using `v5_honest_validation.py`
(923 trading days, 60/40 train/test walk-forward split):

| Metric | In-Sample (train, 553 days) | Out-of-Sample (test, 370 days) |
|--------|-----------------------------|--------------------------------|
| Win rate | ~53–59% | ~53–60% |
| Avg P&L per trade | $7.78–$12.94 | $5.10–$18.86 |
| Sharpe (per-trade, annualised) | 2.45–3.62 | 2.25–3.38 |
| Combined 5-window Sharpe | 4.395 | 2.922 |

The edge **survives** the bias controls — it is not an artefact. But plan for the
walk-forward (OOS) range, not the best-fit (IS) range.

---

## 3. The 5 Optimized Time Windows

All times are **Eastern Time (ET)**, Regular Trading Hours.

| # | Entry (candle close) | Exit (candle close) | Hold | Conviction | Structural Reason |
|---|----------------------|---------------------|------|------------|-------------------|
| 1 | 12:16 | 12:27 | 11 min | Medium | Mid-session rebalancing |
| 2 | 12:51 | 13:00 | 9 min | Medium | Lunch-hour order flow |
| 3 | 13:55 | 14:05 | 10 min | Medium | Early pre-close positioning |
| 4 | **14:50** | **15:01** | **11 min** | **★ High** | **Pre-close institutional re-balance** |
| 5 | **15:48** | **15:58** | **10 min** | **★ High** | **Closing auction burst** |

### Why These Windows Beat Random

- Windows 4 and 5 align with widely-recognised market microstructure events.
- All five windows showed **statistically significant** positive expectancy (t-test p < 0.05)
  in both in-sample and walk-forward tests.
- Combining all five reduces single-window variance (diversification benefit).

---

## 4. Per-Trade Entry & Exit Rules

### Entry Conditions (ALL must be true)

| # | Condition | Detail |
|---|-----------|--------|
| 1 | **Not Thursday** | Skip the entire day on Thursdays |
| 2 | **Not June or October** | Skip these calendar months entirely |
| 3 | **Candle body ≥ 3.0 points** | `abs(close - open) >= 3.0` — filters doji/indecision |
| 4 | **Within a scheduled window** | Only trade during one of the 5 windows above |

### Direction

- **Bullish candle** (close > open) → **LONG** entry at candle close
- **Bearish candle** (close < open) → **SHORT** entry at candle close

### Exit

- Exit at the **close** of the final candle in the window (see table above).
- No stop-loss adjustments mid-window — hold for the full duration.
- If a daily loss limit is approaching, **skip the next window** rather than halving
  size mid-trade.

### Position Sizing (Prop Firm Mode)

```
contracts = floor( (daily_loss_limit / (predicted_window_range × point_value)) × safety_buffer )
contracts = clip(contracts, min=1, max=50)
```

- `predicted_window_range`: Use the **Ensemble** volatility model output (see `volatility_predictor.py`).
- `safety_buffer`: **60%** (proven optimal across backtests).
- Use the **more conservative** of the range-based and MAE-based estimates.

**Quick example** (Topstep $50K, daily limit $1,000, predicted range 32 pts):
```
contracts = floor(1000 / (32 × 2) × 0.60) = floor(9.375) = 9
```

**For simplicity / no model:** Use a fixed **7 contracts** per trade as a conservative
baseline. Revisit sizing after 30 live trading days.

---

## 5. Prop Firm Results — Pass Rates & Sizing

### Summary Table

| Firm | Account | Pass Rate (fixed 6ct, $2K DD) | Pass Rate (Ensemble ML, 60% buffer) | Recommended |
|------|---------|-------------------------------|-------------------------------------|-------------|
| Topstep $50K | $50,000 | ~57% | ~86.7% | ML sizing |
| Apex $50K | $50,000 | ~57% | ~86.7% | ML sizing |
| Generic $50K | $50,000 | ~57% | ~72–79% | ML sizing |

> **Fixed-contract sizing** (6ct, from `v5_honest_validation.py`): lower pass rate but
> requires no prediction model.
> **ML-based sizing** (`volatility_predictor.py`, Ensemble + 60% buffer): higher pass rate,
> requires running the prediction module.

### Key Findings

- **86.7% pass rate** (Ensemble + 60% safety buffer, `volatility_predictor.py`).
- **56.8% pass rate** with fixed 6-contract sizing (worst-day-based, from `v5_honest_validation.py`).
- **0% daily loss limit failures** with prediction-based sizing.
- Three-attempt pass probability approaches 99% across most configurations with ML sizing.
- Fixed 6-contract baseline achieves ~57% on first attempt — acceptable but not optimal.

### Cost-to-Fund Expectation

Assuming a ~$165–$200 challenge fee and a ~75% single-attempt pass rate:
```
Expected challenges to pass = 1 / 0.75 ≈ 1.3
Expected cost to get funded ≈ $215 – $260
```

---

## 6. Daily Execution Checklist

Run through this list **every trading day** before placing any orders.

### Pre-Market (Before 9:30 ET)

- [ ] Check the calendar: **Is today Thursday?** → If yes, **no trades today**
- [ ] Check the calendar: **Is this June or October?** → If yes, **no trades this month**
- [ ] Note today's prop firm drawdown status — are you within normal range?
- [ ] (Optional) Run volatility predictor to get today's predicted window range
- [ ] Calculate today's max contracts using the position sizing formula above

### Each Window (repeat for all 5 windows)

- [ ] Is it within ±1 minute of the scheduled entry time?
- [ ] Observe the 1-minute candle that closes at the entry time
- [ ] Measure candle body: `abs(close - open)`
  - [ ] **Body ≥ 3.0 pts?** → Proceed
  - [ ] **Body < 3.0 pts?** → **Skip this window**
- [ ] Determine direction: bullish → LONG, bearish → SHORT
- [ ] Enter at candle close with the pre-calculated contract count
- [ ] Set a mental note of the exit time (no active stop, just time exit)
- [ ] Exit at the close of the final candle in the window

### Post-Trade / End of Day

- [ ] Log the trade: entry time, direction, contracts, P&L
- [ ] Check cumulative daily P&L vs. daily loss limit (must stay above limit)
- [ ] Update drawdown tracking: current equity vs. peak equity
  - [ ] DD > 3%? → Next trade: halve contracts
  - [ ] DD > 6%? → Next trade: reduce to 25% of normal
  - [ ] DD > 8%? → Stop trading for the day entirely
- [ ] Note any anomalies: unusual fills, missed windows, news events

---

## 7. Monitoring Edge Decay

The strategy edge is **real but not eternal**. Market microstructure can shift. Monitor
these signals monthly:

### Decay Warning Signs

| Signal | Threshold | Action |
|--------|-----------|--------|
| Rolling 30-day win rate | Falls >15% below walk-forward baseline | Pause & investigate |
| Rolling 30-day profit factor | Drops below 1.0 | Pause immediately |
| Average trade P&L | 3-month trend turns negative | Re-run walk-forward |
| Fill quality | Avg slippage doubles vs. first month | Review broker/connection |

### Monthly Review Process

1. Export your live trade log.
2. Calculate the last 30 trading days' win rate, profit factor, and Sharpe.
3. Compare to the walk-forward benchmark ranges in Section 2.
4. If **within range**: continue executing.
5. If **below range**: pause live trading and proceed to Section 8.

---

## 8. Periodic Walk-Forward Retesting

Re-run the full walk-forward analysis every **3–6 months** to:

- Confirm the 5 windows still produce statistically significant returns.
- Re-optimise (if needed) with the latest data added to the training set.
- Update the conservative live-trading expectations.

### How to Re-Run

**Option A — Google Colab (recommended for full honest validation):**

Open [v5_honest_validation.py](v5_honest_validation.py) in Google Colab and upload your
updated CSV when prompted — or run it locally:

```bash
python v5_honest_validation.py --csv path/to/updated_data.csv
```

The script runs all 7 sections automatically in ~2–4 minutes and produces a scored
GO / CAUTION / NO-GO recommendation.

**Option B — Local CLI (quick validation using `advanced_validation` module):**

```bash
# Run full validation suite (requires data.csv updated with recent bars)
python run_all.py --csv path/to/updated_data.csv --module validation

# Run prop firm optimiser to refresh sizing guidance
python run_all.py --csv path/to/updated_data.csv --module propfirm
```

### Decision Rules After Retesting

| Walk-Forward Result | Action |
|--------------------|--------|
| Windows still pass (p < 0.05, profit factor > 1.2) | Continue unchanged |
| 1–2 windows weaken but 3+ remain strong | Drop weak windows, continue |
| All windows weaken substantially | Stop live trading; redesign |
| New windows emerge with higher significance | Consider adding to rotation |

---

## 9. Out-of-Sample Cautions

These are important caveats that every trader must internalise before going live:

1. **Past performance ≠ future results.** The walk-forward tests reduce, but do not
   eliminate, the possibility that all edge disappears.
2. **The shrinkage is a minimum estimate.** Live trading adds execution friction, emotional
   bias, and connectivity issues not captured in backtests. Expect the real edge to be
   at or below the walk-forward lower bound.
3. **Regime change risk.** The strategy was designed and tested on data from a specific
   market regime. A fundamental structural change (e.g., exchange rule change, circuit
   breakers, major liquidity shift) could break the edge entirely.
4. **Prop firm rule changes.** Firms occasionally change their daily loss limits, trailing
   drawdown rules, or contract limits. Re-check your firm's current rules monthly.
5. **Do not over-size during a hot streak.** A winning run doesn't mean the edge is larger
   — it means variance temporarily favoured you. Stick to the position sizing formula.
6. **Paper trade first.** Before risking a challenge fee, paper trade for at least
   20–30 sessions to confirm fills and timing match the backtest assumptions.
7. **News and macro events override everything.** On FOMC, CPI, NFP, or other major
   scheduled releases that fall during a window, **skip that window**. The strategy
   edge is built on normal intraday flow, not volatility spikes from surprise events.
8. **Correlation between windows.** On high-conviction trend days, all five windows
   may move in the same direction, amplifying gains — but also amplifying losses.
   If the first two windows both lose, reduce size for the remaining windows that day.
9. **Broker and platform latency matters.** The strategy entries and exits are at
   specific 1-minute candle closes. A slow connection or manual entry can shift
   your fill by 5–15+ seconds, meaningfully changing your average trade result.
   Use limit orders placed at the expected close price where possible.

---

## 10. Layman / Operational Quick-Reference

This condensed cheat sheet is for day-to-day use. Pin it somewhere visible.

```
╔══════════════════════════════════════════════════════════════╗
║         MNQ MULTI-WINDOW STRATEGY — DAILY CHEAT SHEET       ║
╠══════════════════════════════════════════════════════════════╣
║  SKIP THE WHOLE DAY IF:                                      ║
║    • It's Thursday                                           ║
║    • It's June or October                                    ║
╠══════════════════════════════════════════════════════════════╣
║  5 TRADE WINDOWS (ET)          HOLD    CONVICTION            ║
║    12:16 → 12:27               11 min  ●●○ Medium            ║
║    12:51 → 13:00                9 min  ●●○ Medium            ║
║    13:55 → 14:05               10 min  ●●○ Medium            ║
║  ★ 14:50 → 15:01               11 min  ●●● High (pre-close)  ║
║  ★ 15:48 → 15:58               10 min  ●●● High (auction)    ║
╠══════════════════════════════════════════════════════════════╣
║  ENTRY FILTER (per candle):                                  ║
║    |close - open| ≥ 3.0 pts → trade                         ║
║    |close - open| < 3.0 pts → skip this window              ║
╠══════════════════════════════════════════════════════════════╣
║  DIRECTION:                                                  ║
║    Bullish candle (close > open) → LONG                      ║
║    Bearish candle (close < open) → SHORT                     ║
╠══════════════════════════════════════════════════════════════╣
║  SIZING (prop firm):                                         ║
║    Formula: floor((daily_limit / (range × 2)) × 0.60)       ║
║    Simple fallback: 7 contracts                              ║
╠══════════════════════════════════════════════════════════════╣
║  DRAWDOWN RULES:                                             ║
║    DD > 3% from peak → halve contracts                       ║
║    DD > 6% from peak → 25% of normal contracts              ║
║    DD > 8% from peak → STOP for the day                     ║
╠══════════════════════════════════════════════════════════════╣
║  MONTHLY HEALTH CHECK:                                       ║
║    Win rate dropping >15%? → investigate                     ║
║    Profit factor < 1.0?   → pause immediately               ║
║  RE-RUN WALK-FORWARD EVERY 3–6 MONTHS                        ║
╚══════════════════════════════════════════════════════════════╝
```

---

*Last updated: 2026-03-29. For questions or to re-run analysis, see the Python modules
in this repository and the [README](README.md).*
