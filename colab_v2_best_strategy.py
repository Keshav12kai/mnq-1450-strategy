"""
colab_v2_best_strategy.py — MNQ 14:50 v2 Best Strategy
=======================================================

COLAB USAGE
-----------
Option A – Google Drive (recommended for large files):
    1. Run in a Colab cell:
           from google.colab import drive
           drive.mount('/content/drive')
    2. Set CSV_PATH below, then paste & run this file in the next cell.

Option B – Direct upload:
    1. Run in a Colab cell:
           from google.colab import files
           uploaded = files.upload()   # pick your CSV
    2. Set CSV_PATH to the uploaded filename, then paste & run this file.

LOCAL USAGE
-----------
    python colab_v2_best_strategy.py --csv data.csv
    python colab_v2_best_strategy.py --csv data.csv \\
        --daily-limit 1000 --target 3000 --max-dd 2000 --days 30 \\
        --point-value 2

STRATEGY (v2 best)
------------------
  Entry  : 14:50 ET candle CLOSE, direction = candle direction
  Exit   : 14:59 ET candle CLOSE
  Filter : skip Thursday | skip June & October | body >= 3 pts
  Sizing : contracts = min(range-based, mae-based)  ×  safety_buffer
           range-based = daily_limit / (pred_window_range × point_value)
           mae-based   = daily_limit / (pred_mae          × point_value)
  Model  : Ensemble — 15% EWMA + 15% P75 + 20% FeatureScaling + 50% GradBoost
           GradBoost is walk-forward only (no lookahead)

DEFAULT PROP FIRM PARAMETERS (MNQ, user spec)
---------------------------------------------
  Instrument     MNQ  ($2 / point)
  Profit target  $3,000
  Max drawdown   $2,000
  Challenge days 30
  Daily limit    $1,000  (configurable via --daily-limit)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
#  PARAMETERS — edit these when using copy-paste in Colab
# ═════════════════════════════════════════════════════════════════════════════

CSV_PATH      = "Dataset_NQ_1min_2022_2025.csv"  # ← change to your file path

POINT_VALUE   = 2.0        # MNQ = $2/pt  |  NQ = $20/pt
DAILY_LIMIT   = 1_000.0    # prop firm daily loss limit  ($)
PROFIT_TARGET = 3_000.0    # prop firm profit target      ($)
MAX_DRAWDOWN  = 2_000.0    # prop firm max drawdown       ($)
CHALLENGE_DAYS = 30        # prop firm challenge days

INITIAL_CAPITAL = 10_000.0 # starting capital for equity curve

# Smart filter
SKIP_THURSDAY  = True
SKIP_MONTHS    = {6, 10}   # June, October
MIN_BODY_PTS   = 3.0       # minimum candle body (points)

# Baseline comparison: fixed N contracts with smart filter
BASELINE_CONTRACTS = 9

# Output
OUT_DIR        = "outputs_v2"
CHART_DPI      = 150
MC_RUNS        = 1_000     # Monte Carlo runs per buffer in sweep

# ── Colab detection ───────────────────────────────────────────────────────────
try:
    import google.colab          # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    """Load 1-minute bars, normalise columns, add helper columns."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Accept both 'timestamp ET' and 'timestamp'
    ts_col = next((c for c in df.columns if "timestamp" in c.lower()), None)
    if ts_col is None:
        raise ValueError("CSV must contain a 'timestamp ET' column.")
    df = df.rename(columns={ts_col: "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"]    = df["timestamp"].dt.date
    df["time"]    = df["timestamp"].dt.strftime("%H:%M")
    df["weekday"] = df["timestamp"].dt.dayofweek   # 0=Mon … 3=Thu … 4=Fri
    df["month"]   = df["timestamp"].dt.month
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  TRADE EXTRACTION — v2 targets (window_range + MAE)
# ═════════════════════════════════════════════════════════════════════════════

def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per trading day:
      - entry = close of 14:50 candle
      - exit  = close of 14:59 candle
      - window_range = high − low over the 14:50-14:59 window  (v2 target 1)
      - mae          = max adverse excursion from entry close    (v2 target 2)
      - skip         = True if smart filter applies
    """
    records = []
    for date, grp in df.groupby("date"):
        entry_row = grp[grp["time"] == "14:50"]
        exit_row  = grp[grp["time"] == "14:59"]
        if entry_row.empty or exit_row.empty:
            continue

        e = entry_row.iloc[0]
        x = exit_row.iloc[0]

        window = grp[(grp["time"] >= "14:50") & (grp["time"] <= "14:59")]
        w_high = float(window["high"].max())
        w_low  = float(window["low"].min())

        direction  = "LONG"  if float(e["close"]) >= float(e["open"]) else "SHORT"
        body       = abs(float(e["close"]) - float(e["open"]))
        pnl_long   = float(x["close"]) - float(e["close"])
        pnl_short  = float(e["close"]) - float(x["close"])

        # MAE: max adverse excursion from the 14:50 close
        mae = (float(e["close"]) - w_low) if direction == "LONG" else (w_high - float(e["close"]))
        mae = max(0.0, mae)

        # Smart filter
        skip = False
        if SKIP_THURSDAY and int(e["weekday"]) == 3:
            skip = True
        if int(e["month"]) in SKIP_MONTHS:
            skip = True
        if body < MIN_BODY_PTS:
            skip = True

        records.append({
            "date":          pd.Timestamp(date),
            "weekday":       int(e["weekday"]),
            "month":         int(e["month"]),
            "open_1450":     float(e["open"]),
            "close_1450":    float(e["close"]),
            "high_1450":     float(e["high"]),
            "low_1450":      float(e["low"]),
            "volume_1450":   float(e["volume"]),
            "close_1459":    float(x["close"]),
            "direction":     direction,
            "candle_body":   body,
            "pnl_long_pts":  pnl_long,
            "pnl_short_pts": pnl_short,
            "window_high":   w_high,
            "window_low":    w_low,
            "window_range":  w_high - w_low,
            "mae":           mae,
            "skip":          skip,
        })

    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — pre-14:50 only (no lookahead)
# ═════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Build features available BEFORE 14:50 for each trading day.
    Aligned with trades index.
    """
    feature_rows = []

    for _, tr in trades.iterrows():
        date   = tr["date"].date() if hasattr(tr["date"], "date") else tr["date"]
        day_df = df[df["date"] == date].sort_values("time").reset_index(drop=True)

        def _win(start, end):
            return day_df[(day_df["time"] >= start) & (day_df["time"] < end)]

        last5   = _win("14:45", "14:50")
        last10  = _win("14:40", "14:50")
        last30  = _win("14:20", "14:50")
        morning = _win("09:30", "12:00")
        aftn    = _win("12:00", "14:50")
        full    = _win("09:30", "14:50")

        def _rng(w): return (w["high"] - w["low"]).values if len(w) else np.array([0.0])
        def _vol(w): return  w["volume"].values           if len(w) else np.array([0.0])

        r5, v5   = _rng(last5),   _vol(last5)
        r10, v10 = _rng(last10),  _vol(last10)
        r30, v30 = _rng(last30),  _vol(last30)
        rm, vm   = _rng(morning), _vol(morning)
        ra, va   = _rng(aftn),    _vol(aftn)
        rf, vf   = _rng(full),    _vol(full)

        # VWAP distance (optional column)
        vwap_dist = 0.0
        if "Vwap_RTH" in day_df.columns:
            vr = day_df[day_df["time"] == "14:49"]
            if not vr.empty and not pd.isna(vr["Vwap_RTH"].iloc[0]):
                vwap_dist = abs(tr["close_1450"] - float(vr["Vwap_RTH"].iloc[0]))

        feature_rows.append({
            "last5_avg_range":    r5.mean(),
            "last5_max_range":    r5.max(),
            "last5_avg_vol":      v5.mean(),
            "last5_total_range":  r5.sum(),
            "last10_avg_range":   r10.mean(),
            "last10_max_range":   r10.max(),
            "last10_avg_vol":     v10.mean(),
            "last10_total_range": r10.sum(),
            "last30_avg_range":   r30.mean(),
            "last30_max_range":   r30.max(),
            "last30_avg_vol":     v30.mean(),
            "last30_total_range": r30.sum(),
            "morning_range":      rm.sum(),
            "morning_avg_bar":    rm.mean(),
            "morning_vol":        vm.sum(),
            "aftn_range":         ra.sum(),
            "aftn_avg_bar":       ra.mean(),
            "aftn_vol":           va.sum(),
            "full_range":         rf.sum(),
            "full_vol":           vf.sum(),
            "vol_accel":          v10.sum() / max(va.sum(), 1),
            "range_accel":        r10.mean() / max(ra.mean(), 1e-6),
            "vwap_dist":          vwap_dist,
            "price_level":        tr["close_1450"],
            "entry_range":        tr["high_1450"] - tr["low_1450"],
            "entry_body":         tr["candle_body"],
            "weekday":            tr["weekday"],
            "month":              tr["month"],
            # targets (included for lag/rolling creation)
            "window_range":       tr["window_range"],
            "mae":                tr["mae"],
        })

    feat = pd.DataFrame(feature_rows, index=trades.index)

    # Lagged features (shift to avoid lookahead)
    for lag in [1, 2, 3]:
        for col in ["window_range", "entry_range", "mae", "morning_range", "aftn_range"]:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    # Rolling statistics on window_range (shifted by 1)
    for w in [5, 10, 20]:
        wr_s = feat["window_range"].shift(1)
        feat[f"wr_ma{w}"]  = wr_s.rolling(w, min_periods=1).mean()
        feat[f"wr_std{w}"] = wr_s.rolling(w, min_periods=1).std().fillna(0)

    # EWM features
    for span in [5, 10]:
        feat[f"wr_ewm{span}"] = feat["window_range"].shift(1).ewm(span=span).mean()

    # MAE rolling
    feat["mae_ma5"]  = feat["mae"].shift(1).rolling(5,  min_periods=1).mean()
    feat["mae_ma10"] = feat["mae"].shift(1).rolling(10, min_periods=1).mean()

    return feat.fillna(0)


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICTION MODELS
# ═════════════════════════════════════════════════════════════════════════════

# Feature columns used by GradientBoosting (excludes targets)
_FEATURE_COLS = [
    "last5_avg_range", "last5_max_range", "last5_avg_vol", "last5_total_range",
    "last10_avg_range", "last10_max_range", "last10_avg_vol", "last10_total_range",
    "last30_avg_range", "last30_max_range", "last30_avg_vol", "last30_total_range",
    "morning_range", "morning_avg_bar", "morning_vol",
    "aftn_range", "aftn_avg_bar", "aftn_vol",
    "full_range", "full_vol",
    "vol_accel", "range_accel", "vwap_dist", "price_level",
    "entry_range", "entry_body", "weekday", "month",
    "window_range_lag1", "window_range_lag2", "window_range_lag3",
    "entry_range_lag1", "mae_lag1", "morning_range_lag1", "aftn_range_lag1",
    "wr_ma5", "wr_ma10", "wr_ma20", "wr_std5", "wr_ewm5", "wr_ewm10",
    "mae_ma5", "mae_ma10",
]


def _predict_ewma(feat: pd.DataFrame, span: int = 10) -> np.ndarray:
    return (feat["window_range"].shift(1)
            .ewm(span=span).mean()
            .fillna(feat["entry_range"] * 3.5)
            .values)


def _predict_p75(feat: pd.DataFrame, window: int = 30) -> np.ndarray:
    return (feat["window_range"].shift(1)
            .rolling(window, min_periods=5).quantile(0.75)
            .fillna(feat["entry_range"] * 3.5)
            .values)


def _predict_feature_scaling(feat: pd.DataFrame) -> np.ndarray:
    hist_mult = (
        (feat["window_range"] / feat["last5_avg_range"].clip(lower=0.1))
        .shift(1).rolling(20, min_periods=5).mean().fillna(3.5)
    )
    return (feat["last5_avg_range"] * hist_mult).values


def _predict_gradient_boosting(feat: pd.DataFrame,
                                target: str = "window_range") -> np.ndarray:
    """Walk-forward GradientBoosting — strictly no lookahead."""
    avail = [c for c in _FEATURE_COLS if c in feat.columns and c != target]
    y = feat[target].values
    X = feat[avail].values
    preds     = np.full(len(feat), np.nan)
    min_train = 30

    for i in range(min_train, len(feat)):
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        try:
            model.fit(X[:i], y[:i])
            preds[i] = model.predict(X[i: i + 1])[0]
        except Exception:
            preds[i] = float(np.nanmean(y[:i]))

    fallback = float(np.nanmedian(y[:min_train])) if min_train <= len(y) else float(np.nanmean(y))
    preds[:min_train] = fallback
    return np.clip(preds, 1.0, 500.0)


def _predict_ensemble(ewma, p75, fscale, gb) -> np.ndarray:
    """15% EWMA + 15% P75 + 20% FeatureScaling + 50% GradientBoosting."""
    return 0.15 * ewma + 0.15 * p75 + 0.20 * fscale + 0.50 * gb


def run_predictions(feat: pd.DataFrame):
    """Run all 5 models for window_range and MAE.  Returns dict of arrays."""
    print("  Running prediction models …")

    # window_range predictions
    p_ewma   = _predict_ewma(feat)
    p_p75    = _predict_p75(feat)
    p_fscale = _predict_feature_scaling(feat)
    print("    [GB walk-forward — this may take a minute] …")
    p_gb     = _predict_gradient_boosting(feat, "window_range")
    p_ens    = _predict_ensemble(p_ewma, p_p75, p_fscale, p_gb)

    # MAE predictions (reuse same ensemble structure)
    mae_ewma   = (feat["mae"].shift(1).ewm(span=10).mean().fillna(feat["entry_range"] * 1.5).values)
    mae_p75    = (feat["mae"].shift(1).rolling(30, min_periods=5).quantile(0.75).fillna(feat["entry_range"] * 1.5).values)
    mae_fscale = feat["mae_ma5"].values   # simple fallback feature
    print("    [GB walk-forward for MAE] …")
    mae_gb  = _predict_gradient_boosting(feat, "mae")
    mae_ens = _predict_ensemble(mae_ewma, mae_p75, mae_fscale, mae_gb)

    return {
        "wr_ewma":   p_ewma,
        "wr_p75":    p_p75,
        "wr_fscale": p_fscale,
        "wr_gb":     p_gb,
        "wr_ens":    p_ens,
        "mae_ens":   mae_ens,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  POSITION SIZING
# ═════════════════════════════════════════════════════════════════════════════

def size_contracts(pred_range: float, daily_limit: float,
                   buffer: float, point_value: float,
                   min_c: int = 1, max_c: int = 50) -> int:
    """contracts = floor(daily_limit / (pred_range × pv) × buffer)."""
    if pred_range <= 0:
        return min_c
    c = (daily_limit / (pred_range * point_value)) * buffer
    return int(np.clip(c, min_c, max_c))


def build_contracts_arr(preds: dict, daily_limit: float,
                        buffer: float, point_value: float) -> np.ndarray:
    """
    For each trade compute:
      contracts = min(range-based, mae-based)   ×  buffer  (already baked in)
    """
    n = len(preds["wr_ens"])
    arr = np.empty(n, dtype=int)
    for i in range(n):
        c_wr  = size_contracts(float(preds["wr_ens"][i]),  daily_limit, buffer, point_value)
        c_mae = size_contracts(float(preds["mae_ens"][i]), daily_limit, buffer, point_value)
        arr[i] = min(c_wr, c_mae)
    return arr


# ═════════════════════════════════════════════════════════════════════════════
#  PROP FIRM MONTE CARLO BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

DD_LEVELS = [
    (0.03, 0.50),   # at  3% drawdown → cut contracts by 50 %
    (0.06, 0.75),   # at  6% drawdown → cut contracts by 75 %
    (0.08, 1.00),   # at  8% drawdown → exit entirely
]


def prop_firm_mc(pnl_pts: np.ndarray, contracts_arr: np.ndarray,
                 profit_target: float, max_dd: float,
                 daily_limit: float, challenge_days: int,
                 point_value: float, n_runs: int = 1_000) -> dict:
    """
    Monte Carlo prop-firm challenge simulation.
    Samples challenge_days trades (with replacement), applies DD rules.
    """
    rng = np.random.default_rng(42)
    passed = fail_dd = fail_daily = fail_target = 0
    days_list = []

    for _ in range(n_runs):
        equity = 0.0
        peak   = 0.0
        failed = False
        idxs   = rng.integers(0, len(pnl_pts), size=challenge_days)

        for d in range(challenge_days):
            idx = int(idxs[d])
            c   = int(contracts_arr[idx])

            # DD reduction
            dd_frac = (equity - peak) / max(abs(peak), 1e-6) if peak != 0 else 0.0
            for thresh, cut in DD_LEVELS:
                if abs(dd_frac) >= thresh:
                    c = max(1, int(c * (1 - cut)))

            daily_pnl = float(pnl_pts[idx]) * point_value * c
            equity   += daily_pnl
            peak      = max(peak, equity)

            if equity < -max_dd:
                failed = True; fail_dd += 1; break
            if daily_pnl < -daily_limit:
                failed = True; fail_daily += 1; break

        if not failed:
            if equity >= profit_target:
                passed += 1
                days_list.append(d + 1)
            else:
                fail_target += 1

    nr = max(n_runs, 1)
    return {
        "pass_rate":   passed / nr,
        "avg_days":    float(np.mean(days_list)) if days_list else float(challenge_days),
        "fastest":     int(min(days_list)) if days_list else challenge_days,
        "fail_dd_pct": fail_dd    / nr,
        "fail_daily_pct": fail_daily  / nr,
        "fail_target_pct": fail_target / nr,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SAFETY BUFFER SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def safety_buffer_sweep(pnl_pts: np.ndarray, preds: dict,
                        profit_target: float, max_dd: float,
                        daily_limit: float, challenge_days: int,
                        point_value: float, n_mc: int = 1_000) -> pd.DataFrame:
    """Test safety buffers 30 %–100 % in 10 % steps.  Return pass-rate table."""
    rows = []
    for buf in np.arange(0.30, 1.01, 0.10):
        buf = round(float(buf), 2)
        c_arr = build_contracts_arr(preds, daily_limit, buf, point_value)
        res   = prop_firm_mc(pnl_pts, c_arr, profit_target, max_dd,
                             daily_limit, challenge_days, point_value, n_mc)
        rows.append({
            "buffer_%":          int(round(buf * 100)),
            "pass_rate_%":       round(res["pass_rate"] * 100, 1),
            "avg_contracts":     round(float(c_arr.mean()), 1),
            "avg_days_to_pass":  round(res["avg_days"], 1),
            "fail_dd_%":         round(res["fail_dd_pct"]    * 100, 1),
            "fail_daily_%":      round(res["fail_daily_pct"] * 100, 1),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY EQUITY-CURVE BACKTEST (per-trade)
# ═════════════════════════════════════════════════════════════════════════════

def _pnl_direction(trades: pd.DataFrame) -> np.ndarray:
    """Return per-trade P&L in points following the trade direction."""
    return np.where(
        trades["direction"] == "LONG",
        trades["pnl_long_pts"].values,
        trades["pnl_short_pts"].values,
    ).astype(float)


def backtest_dynamic(trades: pd.DataFrame, contracts_arr: np.ndarray,
                     point_value: float, initial_capital: float,
                     daily_limit: float) -> dict:
    """Full equity-curve backtest with dynamic contract sizing and DD rules."""
    pnl_pts = _pnl_direction(trades)
    equity  = initial_capital
    peak    = initial_capital
    equity_curve = [equity]
    daily_pnl_list = []
    n_daily_breach = 0

    for i in range(len(trades)):
        c = int(contracts_arr[i])

        # DD reduction
        dd_frac = (equity - peak) / max(abs(peak), 1e-6)
        for thresh, cut in DD_LEVELS:
            if abs(dd_frac) >= thresh:
                c = max(1, int(c * (1 - cut)))

        dp = float(pnl_pts[i]) * point_value * c
        equity    += dp
        peak       = max(peak, equity)
        daily_pnl_list.append(dp)
        equity_curve.append(equity)
        if dp < -daily_limit:
            n_daily_breach += 1

    return {
        "equity_curve":   np.array(equity_curve),
        "daily_pnl":      np.array(daily_pnl_list),
        "n_daily_breach": n_daily_breach,
    }


def backtest_fixed(trades: pd.DataFrame, contracts: int,
                   point_value: float, initial_capital: float,
                   daily_limit: float) -> dict:
    """Fixed-contract equity-curve backtest (baseline)."""
    arr = np.full(len(trades), contracts, dtype=int)
    return backtest_dynamic(trades, arr, point_value, initial_capital, daily_limit)


# ═════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════════════

def _sharpe(equity: np.ndarray) -> float:
    rets = np.diff(equity) / np.where(equity[:-1] == 0, 1, equity[:-1])
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2 or np.std(rets) == 0:
        return 0.0
    return float(np.sqrt(252) * np.mean(rets) / np.std(rets))


def _sortino(equity: np.ndarray) -> float:
    rets = np.diff(equity) / np.where(equity[:-1] == 0, 1, equity[:-1])
    rets = rets[np.isfinite(rets)]
    neg  = rets[rets < 0]
    if len(rets) < 2 or len(neg) == 0:
        return 0.0
    return float(np.sqrt(252) * np.mean(rets) / np.std(neg))


def _max_dd(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    dd   = equity - peak
    dd_pct = dd / np.where(peak == 0, 1, peak) * 100
    return float(dd.min()), float(dd_pct.min())


def _profit_factor(pnl: np.ndarray) -> float:
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)


def _cagr(initial: float, final: float,
          start: pd.Timestamp, end: pd.Timestamp) -> float:
    days  = (end - start).days
    years = max(days / 365.25, 1e-9)
    return float((final / initial) ** (1 / years) - 1) * 100


def compute_metrics(label: str, equity: np.ndarray, daily_pnl: np.ndarray,
                    n_daily_breach: int, trades: pd.DataFrame,
                    initial_capital: float) -> dict:
    pnl       = daily_pnl
    wins      = pnl[pnl > 0]
    losses    = pnl[pnl < 0]
    final_cap = float(equity[-1])
    dd_usd, dd_pct = _max_dd(equity)
    start = trades["date"].min()
    end   = trades["date"].max()

    return {
        "label":            label,
        "trades":           int(len(pnl)),
        "win_rate_%":       round(float(np.mean(pnl > 0) * 100), 1),
        "profit_factor":    round(_profit_factor(pnl), 2),
        "sharpe":           round(_sharpe(equity), 2),
        "sortino":          round(_sortino(equity), 2),
        "total_return_%":   round((final_cap / initial_capital - 1) * 100, 1),
        "cagr_%":           round(_cagr(initial_capital, final_cap, start, end), 1),
        "final_capital":    round(final_cap, 2),
        "max_dd_$":         round(dd_usd, 2),
        "max_dd_%":         round(dd_pct, 2),
        "avg_trade_$":      round(float(np.mean(pnl)), 2),
        "avg_win_$":        round(float(np.mean(wins))   if len(wins)   else 0.0, 2),
        "avg_loss_$":       round(float(np.mean(losses)) if len(losses) else 0.0, 2),
        "daily_breach_n":   n_daily_breach,
        "daily_breach_%":   round(n_daily_breach / max(len(pnl), 1) * 100, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(actual: np.ndarray, pred: np.ndarray, name: str) -> dict:
    mask = np.isfinite(actual) & np.isfinite(pred)
    a, p = actual[mask], pred[mask]
    if len(a) < 2:
        return {}
    mae   = mean_absolute_error(a, p)
    rmse  = float(np.sqrt(mean_squared_error(a, p)))
    mape  = float(np.mean(np.abs((a - p) / np.clip(a, 1e-6, None))) * 100)
    corr  = float(np.corrcoef(a, p)[0, 1])
    med   = np.median(a)
    reg   = float(((a >= med) == (p >= med)).mean() * 100)
    return {"model": name, "mae": mae, "rmse": rmse,
            "mape_%": mape, "corr": corr, "regime_acc_%": reg, "n": len(a)}


# ═════════════════════════════════════════════════════════════════════════════
#  CHARTS & LOGS
# ═════════════════════════════════════════════════════════════════════════════

def _dark_fig(*args, **kwargs):
    plt.style.use("dark_background")
    return plt.subplots(*args, **kwargs)


def plot_equity_comparison(dyn_equity: np.ndarray, base_equity: np.ndarray,
                           output_dir: str):
    fig, ax = _dark_fig(figsize=(12, 5))
    ax.plot(dyn_equity,  label="v2 Dynamic (Ensemble + buffer)", color="#42a5f5", lw=1.5)
    ax.plot(base_equity, label=f"Baseline fixed {BASELINE_CONTRACTS} MNQ",
            color="#ffa726", lw=1.5, alpha=0.8)
    ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Equity Curve: v2 Dynamic vs Baseline")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Capital ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "v2_equity_comparison.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pass_rate_sweep(sweep_df: pd.DataFrame, baseline_pass: float,
                         best_buf: int, output_dir: str):
    fig, ax = _dark_fig(figsize=(9, 5))
    ax.plot(sweep_df["buffer_%"], sweep_df["pass_rate_%"],
            "o-", color="#42a5f5", lw=2, label="v2 Ensemble")
    ax.axhline(baseline_pass * 100, color="#ef5350", linestyle="--",
               label=f"Baseline fixed {BASELINE_CONTRACTS}-contract ({baseline_pass*100:.1f}%)")
    ax.axvline(best_buf, color="#26a69a", linestyle=":", alpha=0.7,
               label=f"Optimal buffer ({best_buf}%)")
    ax.set_title("Prop Firm Pass Rate vs Safety Buffer")
    ax.set_xlabel("Safety Buffer (%)")
    ax.set_ylabel("Pass Rate (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "v2_pass_rate_sweep.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_prediction_scatter(feat: pd.DataFrame, preds: dict, output_dir: str):
    fig, axes = _dark_fig(1, 2, figsize=(13, 5))
    actual_wr  = feat["window_range"].values
    actual_mae = feat["mae"].values
    colors = ["#42a5f5", "#ffa726", "#26a69a", "#ef5350", "#ab47bc"]

    for ax, actual, key, title in [
        (axes[0], actual_wr,  "wr_ens",  "window_range: Predicted vs Actual"),
        (axes[1], actual_mae, "mae_ens", "MAE: Predicted vs Actual"),
    ]:
        p    = preds[key]
        mask = np.isfinite(actual) & np.isfinite(p)
        ax.scatter(actual[mask], p[mask], alpha=0.25, s=6, color=colors[0])
        mn = min(actual[mask].min(), p[mask].min())
        mx = max(actual[mask].max(), p[mask].max())
        ax.plot([mn, mx], [mn, mx], "r--", alpha=0.7, label="Perfect")
        ax.set_title(title)
        ax.set_xlabel("Actual (pts)")
        ax.set_ylabel("Predicted (pts)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "v2_prediction_scatter.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_mae_comparison(feat: pd.DataFrame, preds: dict, output_dir: str):
    fig, ax = _dark_fig(figsize=(9, 5))
    actual = feat["window_range"].values
    evals  = {
        "EWMA":          evaluate_model(actual, preds["wr_ewma"],   "EWMA"),
        "P75":           evaluate_model(actual, preds["wr_p75"],    "P75"),
        "Feat.Scaling":  evaluate_model(actual, preds["wr_fscale"], "Feat.Scaling"),
        "GradBoost":     evaluate_model(actual, preds["wr_gb"],     "GradBoost"),
        "Ensemble":      evaluate_model(actual, preds["wr_ens"],    "Ensemble"),
    }
    names = list(evals.keys())
    maes  = [evals[n].get("mae", 0) for n in names]
    colors = ["#42a5f5", "#26a69a", "#ffa726", "#ef5350", "#ab47bc"]
    ax.barh(names, maes, color=colors)
    ax.set_title("Model MAE Comparison (window_range)")
    ax.set_xlabel("MAE (pts)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(output_dir, "v2_model_mae.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run(csv_path: str,
        point_value: float      = POINT_VALUE,
        daily_limit: float      = DAILY_LIMIT,
        profit_target: float    = PROFIT_TARGET,
        max_drawdown: float     = MAX_DRAWDOWN,
        challenge_days: int     = CHALLENGE_DAYS,
        initial_capital: float  = INITIAL_CAPITAL,
        output_dir: str         = OUT_DIR) -> dict:
    """
    Full v2 pipeline.  Returns a dict with key results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  v2 Best Strategy Pipeline")
    print(f"{'='*65}")
    print(f"  CSV:           {csv_path}")
    print(f"  Instrument:    MNQ  (${point_value}/pt)")
    print(f"  Profit target: ${profit_target:,.0f}   Max DD: ${max_drawdown:,.0f}")
    print(f"  Daily limit:   ${daily_limit:,.0f}      Challenge: {challenge_days} days")
    print(f"  Output dir:    {output_dir}")

    print("\nLoading CSV …")
    df = load_data(csv_path)
    print(f"  Rows: {len(df):,}  |  Days: {df['date'].nunique():,}  |  "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── 2. Extract trades ─────────────────────────────────────────────────────
    print("\nExtracting trades (v2 targets: window_range + MAE) …")
    trades_all = extract_trades(df)
    trades     = trades_all[~trades_all["skip"]].reset_index(drop=True)
    print(f"  All potential days : {len(trades_all):,}")
    print(f"  After smart filter : {len(trades):,}")

    if len(trades) < 60:
        raise RuntimeError("Not enough trades after smart filter (need ≥ 60).")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\nBuilding features (pre-14:50 only) …")
    feat = build_features(df, trades)

    # ── 4. Predictions ────────────────────────────────────────────────────────
    preds = run_predictions(feat)

    # ── 5. Model evaluation ───────────────────────────────────────────────────
    actual_wr = feat["window_range"].values
    print("\n── Model Evaluation (window_range) ──────────────────────────")
    for key, label in [("wr_ewma", "EWMA"), ("wr_p75", "P75"),
                       ("wr_fscale", "Feat.Scaling"),
                       ("wr_gb", "GradBoost"), ("wr_ens", "Ensemble")]:
        ev = evaluate_model(actual_wr, preds[key], label)
        if ev:
            print(f"  {label:<16}: MAE={ev['mae']:.2f}  RMSE={ev['rmse']:.2f}"
                  f"  MAPE={ev['mape_%']:.1f}%  Corr={ev['corr']:.3f}"
                  f"  RegAcc={ev['regime_acc_%']:.1f}%")

    # ── 6. Safety buffer sweep ────────────────────────────────────────────────
    print("\n── Safety Buffer Sweep ───────────────────────────────────────")
    pnl_pts = _pnl_direction(trades)
    sweep_df = safety_buffer_sweep(
        pnl_pts, preds, profit_target, max_drawdown,
        daily_limit, challenge_days, point_value, MC_RUNS
    )
    print(sweep_df.to_string(index=False))

    # Best buffer = highest pass rate
    best_row = sweep_df.loc[sweep_df["pass_rate_%"].idxmax()]
    best_buf_frac = best_row["buffer_%"] / 100.0
    best_buf_pct  = int(best_row["buffer_%"])
    print(f"\n  ★ Optimal buffer: {best_buf_pct}%  "
          f"(pass rate {best_row['pass_rate_%']:.1f}%)")

    # ── 7. Build dynamic contract arrays ─────────────────────────────────────
    contracts_dyn = build_contracts_arr(preds, daily_limit, best_buf_frac, point_value)
    contracts_base = np.full(len(trades), BASELINE_CONTRACTS, dtype=int)

    # ── 8. Equity-curve backtests ─────────────────────────────────────────────
    print("\n── Equity-Curve Backtest ─────────────────────────────────────")
    res_dyn  = backtest_dynamic(trades, contracts_dyn,  point_value, initial_capital, daily_limit)
    res_base = backtest_fixed  (trades, BASELINE_CONTRACTS, point_value, initial_capital, daily_limit)

    metrics_dyn  = compute_metrics(
        f"v2 Dynamic (buf={best_buf_pct}%)",
        res_dyn["equity_curve"],  res_dyn["daily_pnl"],
        res_dyn["n_daily_breach"], trades, initial_capital
    )
    metrics_base = compute_metrics(
        f"Baseline fixed {BASELINE_CONTRACTS} MNQ",
        res_base["equity_curve"], res_base["daily_pnl"],
        res_base["n_daily_breach"], trades, initial_capital
    )

    # ── 9. Prop-firm MC for best buffer (dynamic) & baseline ─────────────────
    print("\n── Prop Firm MC (best buffer) ────────────────────────────────")
    mc_dyn  = prop_firm_mc(pnl_pts, contracts_dyn,  profit_target, max_drawdown,
                           daily_limit, challenge_days, point_value, MC_RUNS)
    mc_base = prop_firm_mc(pnl_pts, contracts_base, profit_target, max_drawdown,
                           daily_limit, challenge_days, point_value, MC_RUNS)
    print(f"  v2 Dynamic  : pass_rate={mc_dyn['pass_rate']*100:.1f}%  "
          f"avg_days={mc_dyn['avg_days']:.1f}  "
          f"fail_daily={mc_dyn['fail_daily_pct']*100:.1f}%")
    print(f"  Baseline    : pass_rate={mc_base['pass_rate']*100:.1f}%  "
          f"avg_days={mc_base['avg_days']:.1f}  "
          f"fail_daily={mc_base['fail_daily_pct']*100:.1f}%")

    # ── 10. Print comparison table ────────────────────────────────────────────
    print("\n── Performance Comparison ────────────────────────────────────")
    comp_rows = []
    for m, mc in [(metrics_dyn, mc_dyn), (metrics_base, mc_base)]:
        comp_rows.append({
            "Strategy":       m["label"],
            "Trades":         m["trades"],
            "Win%":           m["win_rate_%"],
            "PF":             m["profit_factor"],
            "Sharpe":         m["sharpe"],
            "MaxDD$":         m["max_dd_$"],
            "TotalRet%":      m["total_return_%"],
            "CAGR%":          m["cagr_%"],
            "DailyBreach%":   m["daily_breach_%"],
            "PassRate%":      round(mc["pass_rate"] * 100, 1),
            "AvgDaysToPass":  round(mc["avg_days"], 1),
        })
    comp = pd.DataFrame(comp_rows).set_index("Strategy")
    try:
        from IPython.display import display
        display(comp)
    except Exception:
        print(comp.to_string())

    # ── 11. Save trade log ────────────────────────────────────────────────────
    log = trades.copy()
    log["contracts_dyn"]  = contracts_dyn
    log["contracts_base"] = contracts_base
    log["pnl_dyn_$"]      = res_dyn["daily_pnl"]
    log["pnl_base_$"]     = res_base["daily_pnl"]
    log_path = os.path.join(output_dir, "v2_trade_log.csv")
    log.to_csv(log_path, index=False)
    print(f"\n  Trade log saved: {log_path}")

    # ── 12. Charts ────────────────────────────────────────────────────────────
    print("\nGenerating charts …")
    plot_equity_comparison(res_dyn["equity_curve"], res_base["equity_curve"], output_dir)
    plot_pass_rate_sweep(sweep_df, mc_base["pass_rate"], best_buf_pct, output_dir)
    plot_prediction_scatter(feat, preds, output_dir)
    plot_model_mae_comparison(feat, preds, output_dir)

    print(f"\n{'='*65}")
    print("  v2 Pipeline COMPLETE")
    print(f"  Output saved to: {os.path.abspath(output_dir)}/")
    print(f"{'='*65}\n")

    return {
        "trades":        trades,
        "feat":          feat,
        "preds":         preds,
        "sweep_df":      sweep_df,
        "best_buf_pct":  best_buf_pct,
        "contracts_dyn": contracts_dyn,
        "metrics_dyn":   metrics_dyn,
        "metrics_base":  metrics_base,
        "mc_dyn":        mc_dyn,
        "mc_base":       mc_base,
        "comp":          comp,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  COLAB DATA-LOADING HELPER  (prints instructions when IN_COLAB = True)
# ═════════════════════════════════════════════════════════════════════════════

def colab_setup():
    """Print Colab data-loading instructions and detect mounted Drive path."""
    if not IN_COLAB:
        return

    # Check if Drive is already mounted
    if os.path.isdir("/content/drive/MyDrive"):
        print("✓ Google Drive is mounted at /content/drive")
        print("  Set CSV_PATH at the top of this script to your file in Drive,")
        print("  e.g.:  CSV_PATH = '/content/drive/MyDrive/Dataset_NQ_1min_2022_2025.csv'")
    else:
        print("Google Drive is NOT mounted.")
        print("To mount:  run the cell below FIRST, then re-run this script.\n")
        print("    from google.colab import drive")
        print("    drive.mount('/content/drive')\n")
        print("Alternatively, upload your file directly:")
        print("    from google.colab import files")
        print("    uploaded = files.upload()")
        print("    CSV_PATH = list(uploaded.keys())[0]")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MNQ 14:50 v2 Best Strategy — Colab & Local Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",         type=str,   default=CSV_PATH,
                   help="Path to 1-minute bar CSV")
    p.add_argument("--daily-limit", type=float, default=DAILY_LIMIT,
                   dest="daily_limit",
                   help="Prop firm daily loss limit ($)")
    p.add_argument("--target",      type=float, default=PROFIT_TARGET,
                   help="Prop firm profit target ($)")
    p.add_argument("--max-dd",      type=float, default=MAX_DRAWDOWN,
                   dest="max_drawdown",
                   help="Prop firm max drawdown ($)")
    p.add_argument("--days",        type=int,   default=CHALLENGE_DAYS,
                   help="Prop firm challenge days")
    p.add_argument("--point-value", type=float, default=POINT_VALUE,
                   dest="point_value",
                   help="Dollar value per point  (MNQ=2, NQ=20)")
    p.add_argument("--capital",     type=float, default=INITIAL_CAPITAL,
                   help="Initial capital for equity curve ($)")
    p.add_argument("--output-dir",  type=str,   default=OUT_DIR,
                   dest="output_dir",
                   help="Directory for output charts and logs")
    return p.parse_args(argv)


if __name__ == "__main__":
    colab_setup()
    args = _parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        if IN_COLAB:
            colab_setup()
        sys.exit(1)

    run(
        csv_path       = args.csv,
        point_value    = args.point_value,
        daily_limit    = args.daily_limit,
        profit_target  = args.target,
        max_drawdown   = args.max_drawdown,
        challenge_days = args.days,
        initial_capital= args.capital,
        output_dir     = args.output_dir,
    )
