"""
colab_v2_best_strategy.py — MNQ 14:50 Strategy: v2 Best Strategy (Colab-ready)
================================================================================
Single-cell Colab script. Copy-paste the entire file into one Colab cell, set
CSV_PATH (or mount Google Drive), and run.

Strategy (v2):
  - Entry:     close of 14:50 ET candle
  - Direction: follow candle (bull → LONG, bear → SHORT)
  - Exit:      close of 14:59 ET candle
  - Smart filter: skip Thu, skip Jun/Oct, skip body < 3 pts

Position sizing (the "86.7% innovation"):
  - Predict the FULL 14:50-14:59 window range before the trade (not just entry candle)
  - contracts = floor( daily_limit / (predicted_range × point_value) × safety_buffer )
  - Also predict MAE; use min(range_based, MAE_based) for the final contract count
  - Ensemble model (EWMA 15% + P75 15% + FeatureScaling 20% + GB 50%)
  - 60% safety buffer = winning config

PROP FIRM RULES AND PASS RATE
------------------------------
Pass rate depends heavily on challenge_days and max_drawdown:

  v2-reference (original ~86.7% result):
    challenge_days=10, max_dd=$2000, daily_limit=$1000, target=$3000
    → shorter window → fewer chances to hit drawdown → higher pass rate

  30-day challenge (your specified rules: $3k target, $2k max DD, 30 days):
    challenge_days=30, max_dd=$2000, daily_limit=$1000, target=$3000
    → 3× more trading days → ~3× more chances to breach limits → lower pass rate

  Apex-style ($2.5k max DD):
    max_dd=$2500  (+$500 headroom → easier to survive volatile days)

This script runs BOTH rule sets so you can see the difference.

USAGE IN GOOGLE COLAB
---------------------
Option A (Google Drive):
    from google.colab import drive
    drive.mount('/content/drive')
    CSV_PATH = "/content/drive/MyDrive/data/Dataset_NQ_1min_2022_2025.csv"

Option B (upload directly):
    from google.colab import files
    uploaded = files.upload()
    CSV_PATH = list(uploaded.keys())[0]

Then change CSV_PATH below and run this cell.
"""

# ============================================================
# CONFIGURATION — edit these values
# ============================================================
CSV_PATH    = "Dataset_NQ_1min_2022_2025.csv"  # path to your 1-min bar CSV

# Instrument
INSTRUMENT  = "MNQ"     # "MNQ" ($2/pt) or "NQ" ($20/pt)
POINT_VALUE = 2.0       # $2 for MNQ, $20 for NQ
INITIAL_CAPITAL = 10_000

# Smart filter
SKIP_THURSDAYS = True
SKIP_MONTHS    = {6, 10}    # June, October
MIN_BODY_PTS   = 3.0        # minimum candle body in points

# Safety buffer for position sizing (60% = v2 winning config)
SAFETY_BUFFER  = 0.60

# Prop firm rule sets to compare
PROP_FIRM_V2_REF = {
    "name":             "v2-reference (10-day, $2k DD)",
    "profit_target":    3_000,
    "max_drawdown":     2_000,
    "daily_loss_limit": 1_000,
    "challenge_days":   10,       # ← original v2 setting → ~86.7% pass rate
}

PROP_FIRM_USER = {
    "name":             "Your 30-day challenge ($3k / $2k DD)",
    "profit_target":    3_000,
    "max_drawdown":     2_000,
    "daily_loss_limit": 1_000,
    "challenge_days":   30,       # ← your requested setting
}

# Number of Monte Carlo iterations (increase for more stable results, but slower)
N_MC_RUNS = 1_000

# Max contracts cap
MAX_CONTRACTS = 50
MIN_CONTRACTS = 1

# DD reduction rules (cut size as drawdown grows)
DD_LEVELS = [
    (0.03, 0.50),   # at  3% drawdown: cut contracts by 50%
    (0.06, 0.75),   # at  6% drawdown: cut contracts by 75%
    (0.08, 1.00),   # at  8% drawdown: exit entirely
]

# ============================================================
# IMPORTS
# ============================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    SKLEARN_OK = True
except ImportError:
    print("scikit-learn not found — running without Gradient Boosting (Ensemble will use EWMA+P75+FeatureScaling only)")
    SKLEARN_OK = False

OUT_DIR = "outputs_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# PRINT EFFECTIVE RULE SET
# ============================================================
print("=" * 65)
print("  MNQ 14:50 Strategy — v2 Best Strategy")
print("=" * 65)
print(f"\n  Instrument:      {INSTRUMENT}  (${POINT_VALUE:.0f}/point)")
print(f"  Initial capital: ${INITIAL_CAPITAL:,}")
print(f"\n  Smart Filter:")
print(f"    Skip Thursdays : {SKIP_THURSDAYS}")
print(f"    Skip months    : {sorted(SKIP_MONTHS)}  (6=Jun, 10=Oct)")
print(f"    Min body pts   : {MIN_BODY_PTS}")
print(f"\n  Position Sizing:")
print(f"    Safety buffer  : {SAFETY_BUFFER*100:.0f}%")
print(f"    contracts = min(range_based, MAE_based)")
print(f"    formula   : floor(daily_limit / (pred_range × ${POINT_VALUE:.0f}) × {SAFETY_BUFFER})")
print(f"\n  Prop Firm Rule Sets compared:")
for firm in [PROP_FIRM_V2_REF, PROP_FIRM_USER]:
    print(f"    [{firm['name']}]")
    print(f"      target={firm['profit_target']}  max_dd={firm['max_drawdown']}  "
          f"daily_limit={firm['daily_loss_limit']}  days={firm['challenge_days']}")
print()

# ============================================================
# DATA LOADING
# ============================================================
print("Loading CSV …")
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

ts_col = next(c for c in df.columns if "timestamp" in c.lower())
df = df.rename(columns={ts_col: "timestamp"})
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
df = df.sort_values("timestamp").reset_index(drop=True)
df["date"]    = df["timestamp"].dt.date
df["time"]    = df["timestamp"].dt.strftime("%H:%M")
df["weekday"] = df["timestamp"].dt.dayofweek   # 0=Mon … 3=Thu
df["month"]   = df["timestamp"].dt.month

print(f"  Rows: {len(df):,}  |  Days: {df['date'].nunique()}  |  "
      f"{df['timestamp'].min()} → {df['timestamp'].max()}")

# ============================================================
# PIPELINE STAGE 1 — Extract trades (all days)
# ============================================================
print("\n── Pipeline Stage 1: Extract trades ────────────────────────")
records = []
for date, grp in df.groupby("date"):
    entry_row = grp[grp["time"] == "14:50"]
    exit_row  = grp[grp["time"] == "14:59"]
    if entry_row.empty or exit_row.empty:
        continue
    e  = entry_row.iloc[0]
    x  = exit_row.iloc[0]
    direction = "LONG" if e["close"] >= e["open"] else "SHORT"
    body = abs(e["close"] - e["open"])
    win_grp = grp[(grp["time"] >= "14:50") & (grp["time"] <= "14:59")]
    records.append({
        "date":          date,
        "weekday":       e["weekday"],
        "month":         e["month"],
        "open_1450":     e["open"],
        "close_1450":    e["close"],
        "high_1450":     e["high"],
        "low_1450":      e["low"],
        "volume_1450":   e["volume"],
        "close_1459":    x["close"],
        "direction":     direction,
        "candle_body":   body,
        "pnl_long_pts":  x["close"] - e["close"],
        "pnl_short_pts": e["close"] - x["close"],
        "window_high":   win_grp["high"].max(),
        "window_low":    win_grp["low"].min(),
    })
trades_raw = pd.DataFrame(records)
trades_raw["window_range"] = trades_raw["window_high"] - trades_raw["window_low"]
trades_raw["mae"] = trades_raw.apply(
    lambda r: max(0.0, r["close_1450"] - r["window_low"])
    if r["direction"] == "LONG" else max(0.0, r["window_high"] - r["close_1450"]), axis=1)
trades_raw["pnl_both_pts"] = np.where(
    trades_raw["direction"] == "LONG",
    trades_raw["pnl_long_pts"],
    trades_raw["pnl_short_pts"])
print(f"  Raw potential trades: {len(trades_raw)}")

# ============================================================
# PIPELINE STAGE 2 — Smart filter
# ============================================================
print("\n── Pipeline Stage 2: Smart filter ──────────────────────────")
def _skip(row):
    if SKIP_THURSDAYS and row["weekday"] == 3:
        return True
    if row["month"] in SKIP_MONTHS:
        return True
    if row["candle_body"] < MIN_BODY_PTS:
        return True
    return False

trades_raw["skip"] = trades_raw.apply(_skip, axis=1)
trades = trades_raw[~trades_raw["skip"]].copy().reset_index(drop=True)
n_skipped = len(trades_raw) - len(trades)
print(f"  After smart filter:   {len(trades)}  (skipped {n_skipped})")

# ============================================================
# PIPELINE STAGE 3 — Feature engineering (NO dropna)
# ============================================================
print("\n── Pipeline Stage 3: Feature engineering ───────────────────")

def build_features(df_1min, trd):
    """All features available BEFORE 14:50. Never drops rows (fillna=0)."""
    rows = []
    for _, tr in trd.iterrows():
        d    = tr["date"]
        day  = df_1min[df_1min["date"] == d].sort_values("time")

        def win(s, e):
            return day[(day["time"] >= s) & (day["time"] < e)]

        r5  = np.array((win("14:45","14:50")["high"] - win("14:45","14:50")["low"]).values or [0.0])
        r10 = np.array((win("14:40","14:50")["high"] - win("14:40","14:50")["low"]).values or [0.0])
        r30 = np.array((win("14:20","14:50")["high"] - win("14:20","14:50")["low"]).values or [0.0])
        v10 = np.array(win("14:40","14:50")["volume"].values or [0])
        va  = np.array(win("12:00","14:50")["volume"].values or [0])
        ra  = np.array((win("12:00","14:50")["high"] - win("12:00","14:50")["low"]).values or [0.0])

        def safe(a): return float(np.sum(a)) if len(a) else 0.0
        def safem(a): return float(np.mean(a)) if len(a) else 0.0

        rows.append({
            "last5_avg_range":    safem(r5),
            "last5_max_range":    float(np.max(r5)) if len(r5) else 0.0,
            "last10_avg_range":   safem(r10),
            "last10_max_range":   float(np.max(r10)) if len(r10) else 0.0,
            "last30_avg_range":   safem(r30),
            "aftn_range":         safe(ra),
            "aftn_avg_bar":       safem(ra),
            "vol_accel":          (safe(v10) / max(safe(va), 1)),
            "price_level":        tr["close_1450"],
            "entry_range":        tr["high_1450"] - tr["low_1450"],
            "entry_body":         tr["candle_body"],
            "weekday":            tr["weekday"],
            "month":              tr["month"],
            "window_range":       tr["window_range"],
            "mae":                tr["mae"],
        })
    feat = pd.DataFrame(rows, index=trd.index)

    # Lagged / rolling features — shift(1) = no future leakage
    # min_periods=1 ensures all N samples are kept (no alignment dropna)
    for lag in [1, 2, 3]:
        feat[f"wr_lag{lag}"]  = feat["window_range"].shift(lag)
        feat[f"mae_lag{lag}"] = feat["mae"].shift(lag)

    for w in [5, 10, 20]:
        feat[f"wr_ma{w}"]   = feat["window_range"].shift(1).rolling(w, min_periods=1).mean()
        feat[f"wr_std{w}"]  = feat["window_range"].shift(1).rolling(w, min_periods=1).std().fillna(0)

    for span in [5, 10]:
        feat[f"wr_ewm{span}"] = feat["window_range"].shift(1).ewm(span=span, min_periods=1).mean()

    feat["mae_ma5"]  = feat["mae"].shift(1).rolling(5,  min_periods=1).mean()
    feat["mae_ma10"] = feat["mae"].shift(1).rolling(10, min_periods=1).mean()

    # fillna(0) fills only the first few lag rows; real data is never NaN
    return feat.fillna(0)

feat_df = build_features(df, trades)
print(f"  After feature build:  {len(feat_df)}  (no alignment drop — all {len(trades)} samples)")

if len(feat_df) != len(trades):
    print(f"  WARNING: mismatch! {len(feat_df)} vs {len(trades)}")

# ============================================================
# PIPELINE STAGE 4 — Predictions (all models produce N predictions)
# ============================================================
print("\n── Pipeline Stage 4: Predictions ───────────────────────────")

FCOLS = [
    "last5_avg_range","last5_max_range","last10_avg_range","last10_max_range",
    "last30_avg_range","aftn_range","aftn_avg_bar","vol_accel",
    "price_level","entry_range","entry_body","weekday","month",
    "wr_lag1","wr_lag2","wr_lag3","mae_lag1",
    "wr_ma5","wr_ma10","wr_ma20","wr_std5","wr_ewm5","wr_ewm10",
    "mae_ma5","mae_ma10",
]
FCOLS = [c for c in FCOLS if c in feat_df.columns]

def _ewma_pred(feat, col="window_range", span=10):
    return (feat[col].shift(1).ewm(span=span, min_periods=1)
            .mean().fillna(feat["entry_range"] * 3.5).values)

def _p75_pred(feat, col="window_range", window=30):
    return (feat[col].shift(1).rolling(window, min_periods=1)
            .quantile(0.75).fillna(feat["entry_range"] * 3.5).values)

def _feat_scale_pred(feat):
    mult = ((feat["window_range"] / feat["last5_avg_range"].clip(lower=0.1))
            .shift(1).rolling(20, min_periods=1).mean().fillna(3.5))
    return (feat["last5_avg_range"] * mult).values

def _gb_pred(feat, target="window_range", min_train=30):
    if not SKLEARN_OK:
        return _ewma_pred(feat, col=target)
    y   = feat[target].values
    X   = feat[FCOLS].values
    out = np.full(len(feat), np.nan)
    for i in range(min_train, len(feat)):
        try:
            m = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                          learning_rate=0.1, random_state=42)
            m.fit(X[:i], y[:i])
            out[i] = m.predict(X[i:i+1])[0]
        except Exception:
            out[i] = float(np.nanmean(y[:i]))
    fallback = float(np.nanmedian(y[:min_train]))
    out[:min_train] = fallback
    return np.clip(out, 1.0, 500.0)

def _ensemble(e, p75, fs, gb):
    return 0.15 * e + 0.15 * p75 + 0.20 * fs + 0.50 * gb

# window_range predictions
p_ewma   = _ewma_pred(feat_df)
p_p75    = _p75_pred(feat_df)
p_fscale = _feat_scale_pred(feat_df)
print("  Running GB for window_range (walk-forward, ~1 min) …")
p_gb     = _gb_pred(feat_df, "window_range")
p_ens    = _ensemble(p_ewma, p_p75, p_fscale, p_gb)

# MAE predictions (same architecture, different target)
p_mae_ewma  = _ewma_pred(feat_df, "mae", span=10)
p_mae_p75   = _p75_pred(feat_df, "mae", window=30)
print("  Running GB for MAE (walk-forward) …")
p_mae_gb    = _gb_pred(feat_df, "mae")
p_mae_ens   = _ensemble(p_mae_ewma, p_mae_p75, feat_df["mae_ma5"].values, p_mae_gb)

# Validate no NaN (alignment safety check)
for name, arr in [("ewma", p_ewma),("p75", p_p75),("feat_scale", p_fscale),
                   ("gb", p_gb),("ensemble", p_ens)]:
    n_nan = int(np.isnan(arr).sum())
    status = "OK" if n_nan == 0 else f"WARNING {n_nan} NaN"
    print(f"  {name:<14} predictions: {len(arr)}  NaN={n_nan}  [{status}]")

n_aligned = int((~np.isnan(p_ens) & ~np.isnan(p_mae_ens)).sum())
print(f"\n  Samples after alignment check: {n_aligned}  (should equal {len(trades)})")

# ============================================================
# PIPELINE STAGE 5 — Prediction distributions
# ============================================================
print("\n── Predicted window_range / MAE distribution ────────────────")
for pct in [10, 25, 50, 75, 90]:
    print(f"  {pct:>3}th pct: range={np.nanpercentile(p_ens, pct):.1f} pts  "
          f"mae={np.nanpercentile(p_mae_ens, pct):.1f} pts")
print(f"  mean  range={np.nanmean(p_ens):.1f}  mae={np.nanmean(p_mae_ens):.1f}")

# ============================================================
# PIPELINE STAGE 6 — Contract sizing: min(range_based, MAE_based)
# ============================================================
print("\n── Contract Sizing ──────────────────────────────────────────")

def _contracts(pred_range, daily_lim, buf):
    if pred_range <= 0:
        return MIN_CONTRACTS
    c = (daily_lim / (pred_range * POINT_VALUE)) * buf
    return int(np.clip(c, MIN_CONTRACTS, MAX_CONTRACTS))

def compute_contracts(preds_range, preds_mae, daily_lim, buf):
    c_wr  = np.array([_contracts(float(p), daily_lim, buf) for p in preds_range])
    c_mae = np.array([_contracts(float(p), daily_lim, buf) for p in preds_mae])
    return np.minimum(c_wr, c_mae)

# Using v2-reference firm daily limit for sizing (both firms share same daily_limit here)
daily_lim_ref = PROP_FIRM_V2_REF["daily_loss_limit"]
contracts_final = compute_contracts(p_ens, p_mae_ens, daily_lim_ref, SAFETY_BUFFER)

print(f"  Safety buffer: {SAFETY_BUFFER*100:.0f}%   daily_limit: ${daily_lim_ref:,}")
print(f"  Avg final contracts: {contracts_final.mean():.1f}")
for pct in [10, 25, 50, 75, 90]:
    print(f"  {pct:>3}th pct: {int(np.percentile(contracts_final, pct))} contracts")

# ============================================================
# PIPELINE STAGE 7 — Daily breach check (historical)
# ============================================================
print("\n── Daily-Limit Breach Check (historical) ────────────────────")
pnl_pts_arr = trades["pnl_both_pts"].values
for firm in [PROP_FIRM_V2_REF, PROP_FIRM_USER]:
    daily_lim = firm["daily_loss_limit"]
    c_arr = compute_contracts(p_ens, p_mae_ens, daily_lim, SAFETY_BUFFER)
    breaches = [(i, float(pnl_pts_arr[i] * POINT_VALUE * int(c_arr[i])))
                for i in range(len(pnl_pts_arr))
                if pnl_pts_arr[i] * POINT_VALUE * int(c_arr[i]) < -daily_lim]
    pct = len(breaches) / max(len(trades), 1) * 100
    print(f"  [{firm['name']}]  breaches={len(breaches)}/{len(trades)} ({pct:.1f}%)")
    if breaches:
        worst = min(breaches, key=lambda x: x[1])
        print(f"    Worst day: trade #{worst[0]}  P&L=${worst[1]:,.0f}")

# ============================================================
# PIPELINE STAGE 8 — Monte Carlo prop firm simulation
# ============================================================
print("\n── Monte Carlo Prop Firm Simulation ─────────────────────────")

def prop_firm_mc(trd, c_arr, firm, n_runs=N_MC_RUNS):
    pnl_arr  = trd["pnl_both_pts"].values.astype(float)
    target   = firm["profit_target"]
    max_dd   = firm["max_drawdown"]
    daily_lim = firm["daily_loss_limit"]
    days     = firm["challenge_days"]
    rng      = np.random.default_rng(42)

    passed = fail_dd = fail_daily = fail_target = 0
    days_list = []
    for _ in range(n_runs):
        equity = 0.0
        peak   = 0.0
        failed = False
        idxs   = rng.integers(0, len(trd), size=days)
        for d in range(days):
            idx = idxs[d]
            c   = int(c_arr[idx])
            pts = pnl_arr[idx]

            # DD reduction
            dd_pct = (equity - peak) / max(abs(peak), 1e-6) if peak != 0 else 0
            for thresh, cut in DD_LEVELS:
                if abs(dd_pct) >= thresh:
                    c = max(1, int(c * (1 - cut)))

            daily_pnl = pts * POINT_VALUE * c
            equity   += daily_pnl
            peak      = max(peak, equity)

            if equity < -max_dd:
                failed = True; fail_dd += 1; break
            if daily_pnl < -daily_lim:
                failed = True; fail_daily += 1; break

        if not failed:
            if equity >= target:
                passed += 1; days_list.append(d + 1)
            else:
                fail_target += 1

    nr = max(n_runs, 1)
    return {
        "pass_rate":   passed / nr,
        "fail_dd":     fail_dd / nr,
        "fail_daily":  fail_daily / nr,
        "fail_target": fail_target / nr,
        "avg_days":    float(np.mean(days_list)) if days_list else float(days),
    }

# Fixed 9-contract baseline (for comparison)
def mc_baseline(trd, n_contracts, firm, n_runs=N_MC_RUNS):
    c_arr = np.full(len(trd), n_contracts, dtype=int)
    return prop_firm_mc(trd, c_arr, firm, n_runs)

# Buffer sweep
def buffer_sweep(trd, preds_range, preds_mae, firm, n_mc=500):
    results = []
    for buf in np.arange(0.30, 1.01, 0.10):
        c_arr = compute_contracts(preds_range, preds_mae, firm["daily_loss_limit"], buf)
        res   = prop_firm_mc(trd, c_arr, firm, n_mc)
        results.append({"buffer": round(float(buf), 2),
                         "pass_rate": res["pass_rate"],
                         "avg_contracts": float(c_arr.mean())})
    return results

mc_results = {}
for firm in [PROP_FIRM_V2_REF, PROP_FIRM_USER]:
    c_arr   = compute_contracts(p_ens, p_mae_ens, firm["daily_loss_limit"], SAFETY_BUFFER)
    res     = prop_firm_mc(trades, c_arr, firm, N_MC_RUNS)
    base_res = mc_baseline(trades, 9, firm, N_MC_RUNS)
    mc_results[firm["name"]] = {"firm": firm, "mc": res, "baseline": base_res,
                                  "contracts": c_arr}

    print(f"\n  [{firm['name']}]  (challenge_days={firm['challenge_days']})")
    print(f"    Ensemble 60% buffer   → pass rate: {res['pass_rate']*100:.1f}%  "
          f"fail_dd={res['fail_dd']*100:.1f}%  fail_daily={res['fail_daily']*100:.1f}%  "
          f"fail_target={res['fail_target']*100:.1f}%")
    print(f"    Fixed 9-ct baseline   → pass rate: {base_res['pass_rate']*100:.1f}%  "
          f"fail_dd={base_res['fail_dd']*100:.1f}%  fail_daily={base_res['fail_daily']*100:.1f}%")

# ============================================================
# PIPELINE STAGE 9 — Buffer sweep for both firms
# ============================================================
print("\n── Buffer Sweep ─────────────────────────────────────────────")
sweep_results = {}
for firm in [PROP_FIRM_V2_REF, PROP_FIRM_USER]:
    sw = buffer_sweep(trades, p_ens, p_mae_ens, firm, n_mc=500)
    sweep_results[firm["name"]] = sw
    best = max(sw, key=lambda r: r["pass_rate"])
    print(f"\n  [{firm['name']}]")
    for r in sw:
        marker = " ◄ best" if r["buffer"] == best["buffer"] else ""
        print(f"    Buffer {r['buffer']*100:.0f}%: pass_rate={r['pass_rate']*100:.1f}%  "
              f"avg_cts={r['avg_contracts']:.1f}{marker}")

# ============================================================
# PIPELINE STAGE 10 — Core strategy backtest (1 contract fixed)
# ============================================================
print("\n── Core Backtest (1 contract, for reference) ────────────────")

def run_fixed_backtest(trd, n_contracts=1):
    pnl_usd = trd["pnl_both_pts"].values * POINT_VALUE * n_contracts
    equity  = INITIAL_CAPITAL + np.cumsum(pnl_usd)
    wins    = pnl_usd[pnl_usd > 0]
    losses  = pnl_usd[pnl_usd < 0]
    eq_full = np.concatenate([[INITIAL_CAPITAL], equity])
    peak    = np.maximum.accumulate(eq_full)
    dd      = eq_full - peak
    return {
        "n_trades":    len(trd),
        "win_rate":    float(np.mean(pnl_usd > 0)) * 100,
        "total_pnl":   float(pnl_usd.sum()),
        "final_cap":   float(equity[-1]),
        "total_ret":   float((equity[-1] / INITIAL_CAPITAL - 1) * 100),
        "max_dd":      float(dd.min()),
        "profit_fac":  float(wins.sum() / abs(losses.sum())) if len(losses) else 999.0,
        "avg_win":     float(wins.mean())   if len(wins)   else 0.0,
        "avg_loss":    float(losses.mean()) if len(losses) else 0.0,
    }

bt = run_fixed_backtest(trades, 1)
pf_display = "N/A (no losses)" if bt['profit_fac'] >= 999 else f"{bt['profit_fac']:.2f}"
print(f"  Trades: {bt['n_trades']}  Win rate: {bt['win_rate']:.1f}%  "
      f"P&L: ${bt['total_pnl']:,.0f}  Max DD: ${bt['max_dd']:,.0f}  PF: {pf_display}")
print(f"  Final capital: ${bt['final_cap']:,.0f}  Total return: {bt['total_ret']:.1f}%")

# ============================================================
# CHARTS
# ============================================================
print("\n── Generating charts ────────────────────────────────────────")

# 1. Predicted vs Actual window_range
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
mask = ~np.isnan(feat_df["window_range"].values) & ~np.isnan(p_ens)
ax.scatter(feat_df["window_range"].values[mask], p_ens[mask],
           alpha=0.3, s=8, color="#42a5f5")
mn = min(feat_df["window_range"].values[mask].min(), p_ens[mask].min())
mx = max(feat_df["window_range"].values[mask].max(), p_ens[mask].max())
ax.plot([mn, mx], [mn, mx], "r--", alpha=0.7)
ax.set_title("Ensemble: Predicted vs Actual window_range")
ax.set_xlabel("Actual (pts)")
ax.set_ylabel("Predicted (pts)")
ax.grid(alpha=0.3)

# 2. Pass rate comparison: v2-ref vs 30-day
ax = axes[1]
colors_map = {"v2-reference (10-day, $2k DD)": "#42a5f5",
              "Your 30-day challenge ($3k / $2k DD)": "#ef5350"}
for fname, sw in sweep_results.items():
    bufs = [r["buffer"] * 100 for r in sw]
    prs  = [r["pass_rate"] * 100 for r in sw]
    c    = colors_map.get(fname, "#ffa726")
    ax.plot(bufs, prs, "o-", label=fname[:30], color=c, lw=2)
ax.axhline(86.7, color="#42a5f5", linestyle=":", alpha=0.5, label="Original v2 ~86.7%")
ax.axvline(60,   color="gray",    linestyle="--", alpha=0.4, label="60% buffer")
ax.set_title("Pass Rate vs Safety Buffer (both rule sets)")
ax.set_xlabel("Safety Buffer (%)")
ax.set_ylabel("Pass Rate (%)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "v2_pass_rate_comparison.png")
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {chart_path}")

# 3. Contract distribution histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(contracts_final, bins=range(1, MAX_CONTRACTS + 2), edgecolor="black",
        color="#42a5f5", alpha=0.7)
ax.axvline(contracts_final.mean(), color="red", linestyle="--",
           label=f"Mean = {contracts_final.mean():.1f}")
ax.set_title("Contract Distribution (Ensemble 60% buffer)")
ax.set_xlabel("Contracts")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
hist_path = os.path.join(OUT_DIR, "contract_distribution.png")
fig.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {hist_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"\n  Dataset:  {len(df):,} rows  |  {df['date'].nunique()} trading days")
print(f"\n  Pipeline:")
print(f"    Raw trades:            {len(trades_raw)}")
print(f"    After smart filter:    {len(trades)}")
print(f"    Feature samples:       {len(feat_df)}  ← all preserved (no alignment drop)")
print(f"\n  1-contract backtest (both dir): win={bt['win_rate']:.1f}%  PF={bt['profit_fac']:.2f}  "
      f"P&L=${bt['total_pnl']:,.0f}")

print(f"\n  {'Firm':<38} {'Ensemble@60%':>14} {'Fixed-9 base':>14}")
print(f"  {'-'*38} {'-'*14} {'-'*14}")
for fname, d in mc_results.items():
    ens_pct  = d['mc']['pass_rate'] * 100
    base_pct = d['baseline']['pass_rate'] * 100
    days     = d['firm']['challenge_days']
    print(f"  {fname[:38]:<38} {ens_pct:>13.1f}% {base_pct:>13.1f}%  [{days}-day]")

print(f"""
  WHY DOES PASS RATE CHANGE?
  --------------------------
  The 86.7% result used a 10-day challenge window.
  Your 30-day challenge runs 3× more trades per attempt:
    → 3× more chances to hit the $2,000 max-drawdown limit
    → 3× more chances to breach the $1,000 daily-loss limit
  Each extra day is an independent risk event, so longer
  challenges are inherently harder to pass.

  To reproduce the original 86.7%:
    Use challenge_days=10  (v2-reference firm above)

  To evaluate YOUR 30-day challenge:
    Use challenge_days=30  (Your 30-day challenge firm above)

  Both sets of rules are shown in the table above.
""")
print("=" * 65)
print(f"  Charts saved to: {OUT_DIR}/")
print("=" * 65)
