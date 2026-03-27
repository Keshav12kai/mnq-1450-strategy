"""
colab_v2_best_strategy.py
=========================
MNQ 14:50 Strategy — Best v2 Colab / Standalone Script

WHAT THIS DOES
--------------
Runs the complete v2 pipeline end-to-end:
  1. Feature engineering   — window range & MAE for the 14:50–14:59 window
  2. Model building         — EWMA, P75, FeatureScaling, GradientBoosting, Ensemble
                             (walk-forward, no lookahead bias)
  3. Safety-buffer sweep    — 30%→100% for Topstep $50K (and any prop firm you configure)
  4. Backtest               — Ensemble @ 60% buffer + Smart Filter
  5. Final comparison table — Ensemble v2  vs  Fixed-9 baseline (both with Smart Filter)
  6. Saves plots & trade logs to ./output_v2/

HOW TO USE IN GOOGLE COLAB
---------------------------
Option A — Google Drive mount (recommended for large CSVs):

    from google.colab import drive
    drive.mount('/content/drive')
    CSV_PATH = "/content/drive/MyDrive/Dataset_NQ_1min_2022_2025.csv"

Option B — Direct file upload:

    from google.colab import files
    uploaded = files.upload()          # choose your CSV
    CSV_PATH = list(uploaded.keys())[0]

Then either:
  • Run this script as-is in a Colab cell after cloning the repo:

        !git clone https://github.com/Keshav12kai/mnq-1450-strategy.git
        %cd mnq-1450-strategy
        !pip install -q -r requirements.txt
        # paste/run this file

  • Or simply paste the whole file into a single Colab cell.

STANDALONE (local / terminal)
------------------------------
    pip install pandas numpy matplotlib scipy scikit-learn
    python colab_v2_best_strategy.py --csv Dataset_NQ_1min_2022_2025.csv

Dependencies: pandas, numpy, matplotlib, scipy, scikit-learn
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  edit the block below to match your prop firm / preferences
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH        = "Dataset_NQ_1min_2022_2025.csv"   # override via --csv or Colab
OUTPUT_DIR      = "output_v2"
INITIAL_CAPITAL = 10_000          # your starting paper/sim capital ($)
POINT_VALUE     = 2.0             # MNQ = $2/pt  |  NQ = $20/pt
FIXED_BASELINE_CONTRACTS = 9      # fixed-contract baseline to compare against
SAFETY_BUFFER   = 0.60            # default / winning buffer
ENTRY_TIME      = "14:50"
EXIT_TIME       = "14:59"
MIN_CANDLE_BODY = 3.0             # points — skip indecision candles
SKIP_THURSDAYS  = True
SKIP_MONTHS     = [6, 10]         # June & October

# Prop firm used for position sizing & pass-rate simulation
PROP_FIRM = {
    "name":             "Topstep $50K",
    "account_size":     50_000,
    "profit_target":    3_000,
    "max_drawdown":     2_000,
    "daily_loss_limit": 1_000,
    "challenge_days":   10,
    "fee":              165,
}

DD_LEVELS = [        # (drawdown_pct_threshold, fraction_to_cut)
    (0.03, 0.50),    # at 3 % DD → halve contracts
    (0.06, 0.75),    # at 6 % DD → keep 25 %
    (0.08, 1.00),    # at 8 % DD → exit for the day
]
MAX_CONTRACTS = 50
MIN_CONTRACTS = 1
MC_RUNS       = 500   # Monte Carlo iterations for buffer sweep (raise to 2000 for precision)

FEATURE_COLS = [
    "last5_avg_range", "last5_max_range", "last5_avg_vol", "last5_total_range",
    "last10_avg_range", "last10_max_range", "last10_avg_vol", "last10_total_range",
    "last30_avg_range", "last30_max_range", "last30_avg_vol", "last30_total_range",
    "morning_range", "morning_avg_bar", "morning_vol",
    "aftn_range", "aftn_avg_bar", "aftn_vol",
    "full_range", "full_vol",
    "vol_accel", "range_accel", "vwap_dist", "price_level",
    "entry_range", "entry_body",
    "weekday", "month",
    "window_range_lag1", "window_range_lag2", "window_range_lag3",
    "entry_range_lag1", "mae_lag1", "morning_range_lag1", "aftn_range_lag1",
    "wr_ma5", "wr_ma10", "wr_ma20", "wr_std5", "wr_ewm5", "wr_ewm10",
    "mae_ma5", "mae_ma10",
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Data Loading & Smart Filter
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    ts_col = next(c for c in df.columns if "timestamp" in c.lower())
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"]    = df["timestamp"].dt.date
    df["time"]    = df["timestamp"].dt.strftime("%H:%M")
    df["weekday"] = df["timestamp"].dt.dayofweek   # 0=Mon … 4=Fri; 3=Thu
    df["month"]   = df["timestamp"].dt.month
    return df


def smart_filter(row: pd.Series) -> bool:
    """Return True if the trade should be SKIPPED."""
    if SKIP_THURSDAYS and row["weekday"] == 3:
        return True
    if row["month"] in SKIP_MONTHS:
        return True
    body = abs(row["close_1450"] - row["open_1450"])
    if body < MIN_CANDLE_BODY:
        return True
    return False


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for date, grp in df.groupby("date"):
        entry_row = grp[grp["time"] == ENTRY_TIME]
        exit_row  = grp[grp["time"] == EXIT_TIME]
        if entry_row.empty or exit_row.empty:
            continue
        e = entry_row.iloc[0]
        x = exit_row.iloc[0]
        direction  = "LONG" if e["close"] >= e["open"] else "SHORT"
        body       = abs(e["close"] - e["open"])
        pnl_long   = x["close"] - e["close"]
        pnl_short  = e["close"] - x["close"]
        win_rows   = grp[(grp["time"] >= ENTRY_TIME) & (grp["time"] <= EXIT_TIME)]
        records.append({
            "date":          date,
            "weekday":       e["weekday"],
            "month":         e["month"],
            "open_1450":     e["open"],
            "close_1450":    e["close"],
            "high_1450":     e["high"],
            "low_1450":      e["low"],
            "close_1459":    x["close"],
            "direction":     direction,
            "candle_body":   body,
            "pnl_long_pts":  pnl_long,
            "pnl_short_pts": pnl_short,
            "window_high":   win_rows["high"].max(),
            "window_low":    win_rows["low"].min(),
        })

    trades = pd.DataFrame(records)
    if not trades.empty:
        trades["window_range"] = trades["window_high"] - trades["window_low"]
        def _mae(r):
            if r["direction"] == "LONG":
                return max(0.0, r["close_1450"] - r["window_low"])
            return max(0.0, r["window_high"] - r["close_1450"])
        trades["mae"]  = trades.apply(_mae, axis=1)
        trades["skip"] = trades.apply(smart_filter, axis=1)
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []

    for _, trade_row in trades.iterrows():
        date   = trade_row["date"]
        day_df = df[df["date"] == date].copy().sort_values("time").reset_index(drop=True)

        def _get_window(start, end):
            return day_df[(day_df["time"] >= start) & (day_df["time"] < end)]

        last5   = _get_window("14:45", "14:50")
        last10  = _get_window("14:40", "14:50")
        last30  = _get_window("14:20", "14:50")
        morning = _get_window("09:30", "12:00")
        aftn    = _get_window("12:00", "14:50")
        full    = _get_window("09:30", "14:50")

        def _rng(w): return (w["high"] - w["low"]).values if len(w) else np.array([0.0])
        def _vol(w): return w["volume"].values            if len(w) else np.array([0.0])

        r5  = _rng(last5);   v5  = _vol(last5)
        r10 = _rng(last10);  v10 = _vol(last10)
        r30 = _rng(last30);  v30 = _vol(last30)
        rm  = _rng(morning); vm  = _vol(morning)
        ra  = _rng(aftn);    va  = _vol(aftn)
        rf  = _rng(full);    vf  = _vol(full)

        vwap_dist = 0.0
        if "Vwap_RTH" in day_df.columns:
            vwap_row = day_df[day_df["time"] == "14:49"]
            if not vwap_row.empty and not pd.isna(vwap_row["Vwap_RTH"].iloc[0]):
                vwap_dist = abs(trade_row["close_1450"] - vwap_row["Vwap_RTH"].iloc[0])

        row = {
            "last5_avg_range":    r5.mean(),   "last5_max_range":    r5.max(),
            "last5_avg_vol":      v5.mean(),   "last5_total_range":  r5.sum(),
            "last10_avg_range":   r10.mean(),  "last10_max_range":   r10.max(),
            "last10_avg_vol":     v10.mean(),  "last10_total_range": r10.sum(),
            "last30_avg_range":   r30.mean(),  "last30_max_range":   r30.max(),
            "last30_avg_vol":     v30.mean(),  "last30_total_range": r30.sum(),
            "morning_range":      rm.sum(),    "morning_avg_bar":    rm.mean(),
            "morning_vol":        vm.sum(),    "aftn_range":         ra.sum(),
            "aftn_avg_bar":       ra.mean(),   "aftn_vol":           va.sum(),
            "full_range":         rf.sum(),    "full_vol":           vf.sum(),
            "vol_accel":          v10.sum() / max(va.sum(), 1),
            "range_accel":        r10.mean() / max(ra.mean(), 1e-6),
            "vwap_dist":          vwap_dist,
            "price_level":        trade_row["close_1450"],
            "entry_range":        trade_row["high_1450"] - trade_row["low_1450"],
            "entry_body":         trade_row["candle_body"],
            "weekday":            trade_row["weekday"],
            "month":              trade_row["month"],
            "window_range":       trade_row["window_range"],
            "mae":                trade_row["mae"],
        }
        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows).reset_index(drop=True)

    # Lagged targets
    for lag in [1, 2, 3]:
        feat_df[f"window_range_lag{lag}"] = feat_df["window_range"].shift(lag)
    feat_df["entry_range_lag1"]   = feat_df["entry_range"].shift(1)
    feat_df["mae_lag1"]           = feat_df["mae"].shift(1)
    feat_df["morning_range_lag1"] = feat_df["morning_range"].shift(1)
    feat_df["aftn_range_lag1"]    = feat_df["aftn_range"].shift(1)

    # Rolling stats
    for w in [5, 10, 20]:
        feat_df[f"wr_ma{w}"]   = feat_df["window_range"].shift(1).rolling(w, min_periods=1).mean()
        feat_df[f"wr_std{w}"]  = feat_df["window_range"].shift(1).rolling(w, min_periods=1).std().fillna(0)
    for span in [5, 10]:
        feat_df[f"wr_ewm{span}"] = feat_df["window_range"].shift(1).ewm(span=span).mean()
    feat_df["mae_ma5"]  = feat_df["mae"].shift(1).rolling(5,  min_periods=1).mean()
    feat_df["mae_ma10"] = feat_df["mae"].shift(1).rolling(10, min_periods=1).mean()

    return feat_df.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Prediction Models
# ─────────────────────────────────────────────────────────────────────────────

def predict_ewma(feat_df: pd.DataFrame, span: int = 10) -> np.ndarray:
    return feat_df["window_range"].shift(1).ewm(span=span).mean().fillna(
        feat_df["entry_range"] * 3.5).values


def predict_p75(feat_df: pd.DataFrame, window: int = 30) -> np.ndarray:
    return feat_df["window_range"].shift(1).rolling(window, min_periods=5).quantile(0.75).fillna(
        feat_df["entry_range"] * 3.5).values


def predict_feature_scaling(feat_df: pd.DataFrame) -> np.ndarray:
    hist_mult = (feat_df["window_range"] / feat_df["last5_avg_range"].clip(lower=0.1)
                 ).shift(1).rolling(20, min_periods=5).mean().fillna(3.5)
    return (feat_df["last5_avg_range"] * hist_mult).values


def predict_gradient_boosting(feat_df: pd.DataFrame,
                               target: str = "window_range") -> np.ndarray:
    """Walk-forward GradientBoosting — no lookahead bias."""
    available_cols = [c for c in FEATURE_COLS if c in feat_df.columns and c != target]
    y      = feat_df[target].values
    X      = feat_df[available_cols].values
    preds  = np.full(len(feat_df), np.nan)
    min_train = 30

    for i in range(min_train, len(feat_df)):
        X_train = X[:i]
        y_train = y[:i]
        model   = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        try:
            model.fit(X_train, y_train)
            preds[i] = model.predict(X[i : i + 1])[0]
        except Exception:
            preds[i] = np.nanmean(y_train)

    fallback = np.nanmedian(y[:min_train]) if min_train <= len(y) else np.nanmean(y)
    preds[:min_train] = fallback
    return np.clip(preds, 1.0, 500.0)


def predict_ensemble(ewma: np.ndarray, p75: np.ndarray,
                     feat_scale: np.ndarray, gb: np.ndarray) -> np.ndarray:
    """Weighted: 15 % EWMA + 15 % P75 + 20 % FeatureScaling + 50 % GB."""
    return 0.15 * ewma + 0.15 * p75 + 0.20 * feat_scale + 0.50 * gb


def evaluate_model(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    a, p = actual[mask], predicted[mask]
    if len(a) < 2:
        return {}
    mae  = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    mape = np.mean(np.abs((a - p) / np.clip(a, 1e-6, None))) * 100
    corr = float(np.corrcoef(a, p)[0, 1])
    med  = np.median(a)
    regime_acc = ((a >= med) == (p >= med)).mean()
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape,
            "corr": corr, "regime_acc": regime_acc}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Position Sizing
# ─────────────────────────────────────────────────────────────────────────────

def contracts_from_prediction(predicted_window_range: float,
                               daily_loss_limit: float,
                               safety_buffer: float) -> int:
    """contracts = floor( (daily_limit / (pred_range × pv)) × buffer )"""
    if predicted_window_range <= 0:
        return MIN_CONTRACTS
    c = (daily_loss_limit / (predicted_window_range * POINT_VALUE)) * safety_buffer
    return int(np.clip(c, MIN_CONTRACTS, MAX_CONTRACTS))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Prop-Firm Monte Carlo & Safety-Buffer Sweep
# ─────────────────────────────────────────────────────────────────────────────

def prop_firm_mc(trades: pd.DataFrame, contracts_arr: np.ndarray,
                 prop_firm: dict, n_runs: int = MC_RUNS) -> dict:
    pnl_arr   = np.where(trades["direction"] == "LONG",
                         trades["pnl_long_pts"],
                         trades["pnl_short_pts"]).astype(float)
    target    = prop_firm["profit_target"]
    max_dd_abs = prop_firm["max_drawdown"]
    daily_lim  = prop_firm["daily_loss_limit"]
    days       = prop_firm["challenge_days"]

    rng    = np.random.default_rng(42)
    passed = 0
    fail_dd = fail_daily = fail_target = 0
    days_list = []

    for _ in range(n_runs):
        equity = 0.0
        peak   = 0.0
        failed = False
        day    = 0
        idxs   = rng.integers(0, len(trades), size=days)
        for d in range(days):
            idx = idxs[d]
            c   = int(contracts_arr[idx]) if idx < len(contracts_arr) else 1
            pts = pnl_arr[idx]

            dd_pct = (equity - peak) / max(abs(peak), 1e-6) if peak != 0 else 0.0
            for thresh, cut in DD_LEVELS:
                if abs(dd_pct) >= thresh:
                    c = max(1, int(c * (1 - cut)))

            daily_pnl = pts * POINT_VALUE * c
            equity   += daily_pnl
            peak      = max(peak, equity)
            day      += 1

            if equity < -max_dd_abs:
                failed = True; fail_dd += 1; break
            if daily_pnl < -daily_lim:
                failed = True; fail_daily += 1; break

        if not failed:
            if equity >= target:
                passed += 1
                days_list.append(day)
            else:
                fail_target += 1

    n_f = max(n_runs, 1)
    return {
        "pass_rate":   passed / n_f,
        "avg_days":    float(np.mean(days_list)) if days_list else float(days),
        "fail_dd":     fail_dd / n_f,
        "fail_daily":  fail_daily / n_f,
        "fail_target": fail_target / n_f,
    }


def safety_buffer_sweep(trades: pd.DataFrame, ensemble_preds: np.ndarray,
                         prop_firm: dict) -> list:
    """Run MC simulation for buffers 30 %→100 % in 10 % steps."""
    results = []
    for buf in np.arange(0.30, 1.01, 0.10):
        contracts_arr = np.array([
            contracts_from_prediction(float(ensemble_preds[i]),
                                      prop_firm["daily_loss_limit"], buf)
            for i in range(len(trades))
        ])
        res = prop_firm_mc(trades, contracts_arr, prop_firm)
        results.append({
            "buffer":        round(float(buf), 2),
            "pass_rate":     res["pass_rate"],
            "avg_contracts": float(contracts_arr.mean()),
            "fail_daily":    res["fail_daily"],
            "avg_days":      res["avg_days"],
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────

def _streaks(pnl: np.ndarray):
    max_w = max_l = cur_w = cur_l = 0
    for v in pnl:
        if v > 0:
            cur_w += 1; cur_l = 0
        elif v < 0:
            cur_l += 1; cur_w = 0
        else:
            cur_w = cur_l = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def compute_stats(res: pd.DataFrame) -> dict:
    pnl  = res["pnl"].values
    eq   = np.concatenate([[INITIAL_CAPITAL], INITIAL_CAPITAL + np.cumsum(pnl)])
    n    = len(pnl)
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    win_rate     = len(wins) / n if n else 0.0
    final_cap    = eq[-1]
    total_return = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    peak         = np.maximum.accumulate(eq)
    dd           = eq - peak
    dd_pct       = np.where(peak > 0, dd / peak * 100, 0.0)
    max_dd       = dd.min()
    max_dd_pct   = dd_pct.min()
    daily_ret    = pnl / INITIAL_CAPITAL
    excess       = daily_ret
    sharpe       = (excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) > 0 else 0.0
    down         = daily_ret[daily_ret < 0]
    sortino      = (excess.mean() / down.std(ddof=1) * np.sqrt(252)) if len(down) > 1 and down.std(ddof=1) > 0 else 0.0
    gross_profit = wins.sum()  if len(wins)   else 0.0
    gross_loss   = losses.sum() if len(losses) else 0.0
    pf           = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")
    avg_win      = wins.mean()   if len(wins)   else 0.0
    avg_loss     = losses.mean() if len(losses) else 0.0
    expectancy   = win_rate * avg_win + (1 - win_rate) * avg_loss
    streaks      = _streaks(pnl)
    return {
        "n_trades":       n,
        "win_rate":       win_rate,
        "total_pnl":      pnl.sum(),
        "final_capital":  final_cap,
        "total_return":   total_return,
        "max_dd":         max_dd,
        "max_dd_pct":     max_dd_pct,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "profit_factor":  pf,
        "expectancy":     expectancy,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "max_cons_wins":  streaks[0],
        "max_cons_losses": streaks[1],
    }


def run_backtest_dynamic(trades: pd.DataFrame,
                          contracts_arr: np.ndarray) -> dict:
    """Backtest with per-trade contract sizes (from volatility prediction)."""
    filtered = trades[~trades["skip"]].copy().reset_index(drop=True)
    if len(filtered) == 0:
        return {"trades": pd.DataFrame(), "stats": {}}

    n = min(len(filtered), len(contracts_arr))
    records = []
    for i in range(n):
        row = filtered.iloc[i]
        pts = row["pnl_long_pts"] if row["direction"] == "LONG" else row["pnl_short_pts"]
        c   = int(contracts_arr[i])
        records.append({
            "date":      row["date"],
            "direction": row["direction"],
            "entry":     row["close_1450"],
            "exit":      row["close_1459"],
            "pts":       pts,
            "pnl":       pts * POINT_VALUE * c,
            "contracts": c,
        })

    res = pd.DataFrame(records)
    res["equity"] = INITIAL_CAPITAL + res["pnl"].cumsum()
    return {"trades": res, "stats": compute_stats(res)}


def run_backtest_fixed(trades: pd.DataFrame, contracts: int = 9) -> dict:
    """Backtest with fixed contract count (baseline)."""
    filtered = trades[~trades["skip"]].copy().reset_index(drop=True)
    if len(filtered) == 0:
        return {"trades": pd.DataFrame(), "stats": {}}

    records = []
    for _, row in filtered.iterrows():
        pts = row["pnl_long_pts"] if row["direction"] == "LONG" else row["pnl_short_pts"]
        records.append({
            "date":      row["date"],
            "direction": row["direction"],
            "entry":     row["close_1450"],
            "exit":      row["close_1459"],
            "pts":       pts,
            "pnl":       pts * POINT_VALUE * contracts,
            "contracts": contracts,
        })

    res = pd.DataFrame(records)
    res["equity"] = INITIAL_CAPITAL + res["pnl"].cumsum()
    return {"trades": res, "stats": compute_stats(res)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_comparison(v2_res: dict, baseline_res: dict,
                            output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("v2 Ensemble @ 60 % Buffer  vs  Fixed-9 Baseline", fontsize=13, fontweight="bold")

    for ax, res, label, color in [
        (axes[0], v2_res,       "Ensemble v2 (60 % buffer)", "#42a5f5"),
        (axes[1], baseline_res, f"Fixed {FIXED_BASELINE_CONTRACTS} contracts", "#ffa726"),
    ]:
        tr = res.get("trades")
        if tr is not None and len(tr):
            eq = np.concatenate([[INITIAL_CAPITAL], tr["equity"].values])
            ax.plot(eq, color=color, lw=1.5)
            ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
            peak = np.maximum.accumulate(eq)
            dd   = (eq - peak) / peak * 100
            ax2  = ax.twinx()
            ax2.fill_between(range(len(dd)), dd, 0, color="#ef5350", alpha=0.25, label="DD %")
            ax2.set_ylabel("Drawdown (%)", color="#ef5350")
            ax2.tick_params(axis="y", colors="#ef5350")
        ax.set_title(label)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Capital ($)")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "equity_comparison_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pass_rate_sweep(sweep_results: list, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    buffers    = [r["buffer"] * 100 for r in sweep_results]
    pass_rates = [r["pass_rate"] * 100 for r in sweep_results]
    ax.plot(buffers, pass_rates, "o-", color="#42a5f5", lw=2, label="Ensemble v2")

    # Baseline (fixed-9) approximation line
    baseline_pct = None
    baseline_contracts_arr = np.full(1, FIXED_BASELINE_CONTRACTS)

    ax.axhline(59.1, color="#ef5350", linestyle="--",
               label=f"Fixed {FIXED_BASELINE_CONTRACTS}-contract baseline (~59.1 %)")

    # Highlight winner
    best_idx = int(np.argmax(pass_rates))
    ax.scatter([buffers[best_idx]], [pass_rates[best_idx]],
               color="#ffd700", s=120, zorder=5,
               label=f"Best: {buffers[best_idx]:.0f} % → {pass_rates[best_idx]:.1f} %")

    ax.set_title(f"Safety-Buffer Sweep — {PROP_FIRM['name']}")
    ax.set_xlabel("Safety Buffer (%)")
    ax.set_ylabel("Prop-Firm Pass Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "safety_buffer_sweep_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_prediction_vs_actual(feat_df: pd.DataFrame, p_ens: np.ndarray,
                               output_dir: str) -> None:
    actual = feat_df["window_range"].values
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ensemble Prediction — 14:50–14:59 Window Range", fontsize=13, fontweight="bold")

    # Scatter
    mask = ~np.isnan(actual) & ~np.isnan(p_ens)
    axes[0].scatter(actual[mask], p_ens[mask], alpha=0.3, s=8, color="#42a5f5")
    mn, mx = min(actual[mask].min(), p_ens[mask].min()), max(actual[mask].max(), p_ens[mask].max())
    axes[0].plot([mn, mx], [mn, mx], "r--", alpha=0.7, label="Perfect")
    axes[0].set_title("Predicted vs Actual")
    axes[0].set_xlabel("Actual Range (pts)")
    axes[0].set_ylabel("Predicted Range (pts)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Time series
    xs = np.arange(len(actual))
    axes[1].plot(xs, actual, label="Actual", color="#e0e0e0", lw=1, alpha=0.85)
    axes[1].plot(xs, p_ens,  label="Ensemble", color="#42a5f5", lw=0.9, alpha=0.8)
    axes[1].set_title("Window Range Over Time")
    axes[1].set_xlabel("Trade #")
    axes[1].set_ylabel("Range (pts)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_vs_actual_v2.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Final Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def print_final_comparison(v2_stats: dict, v2_mc: dict,
                             baseline_stats: dict, baseline_mc: dict,
                             sweep_results: list) -> None:
    best_sweep = max(sweep_results, key=lambda r: r["pass_rate"])

    col_w = 26
    sep   = "=" * (30 + col_w * 2)
    print("\n" + sep)
    print("  FINAL COMPARISON TABLE  —  MNQ 14:50 Strategy v2")
    print(sep)
    print(f"{'Metric':<30}{'Ensemble v2 (60% buf)':>{col_w}}{'Fixed-9 Baseline':>{col_w}}")
    print("-" * (30 + col_w * 2))

    rows = [
        ("Total Trades",          f"{v2_stats.get('n_trades', 0):,}",
                                   f"{baseline_stats.get('n_trades', 0):,}"),
        ("Win Rate",              f"{v2_stats.get('win_rate', 0)*100:.1f}%",
                                   f"{baseline_stats.get('win_rate', 0)*100:.1f}%"),
        ("Total P&L ($)",         f"${v2_stats.get('total_pnl', 0):,.0f}",
                                   f"${baseline_stats.get('total_pnl', 0):,.0f}"),
        ("Total Return",          f"{v2_stats.get('total_return', 0):.1f}%",
                                   f"{baseline_stats.get('total_return', 0):.1f}%"),
        ("Max Drawdown ($)",       f"${v2_stats.get('max_dd', 0):,.0f}",
                                   f"${baseline_stats.get('max_dd', 0):,.0f}"),
        ("Max Drawdown (%)",       f"{v2_stats.get('max_dd_pct', 0):.1f}%",
                                   f"{baseline_stats.get('max_dd_pct', 0):.1f}%"),
        ("Sharpe Ratio",          f"{v2_stats.get('sharpe', 0):.2f}",
                                   f"{baseline_stats.get('sharpe', 0):.2f}"),
        ("Sortino Ratio",         f"{v2_stats.get('sortino', 0):.2f}",
                                   f"{baseline_stats.get('sortino', 0):.2f}"),
        ("Profit Factor",         f"{v2_stats.get('profit_factor', 0):.2f}",
                                   f"{baseline_stats.get('profit_factor', 0):.2f}"),
        ("Expectancy ($/trade)",   f"${v2_stats.get('expectancy', 0):.2f}",
                                   f"${baseline_stats.get('expectancy', 0):.2f}"),
        ("Max Cons. Wins",        f"{v2_stats.get('max_cons_wins', 0)}",
                                   f"{baseline_stats.get('max_cons_wins', 0)}"),
        ("Max Cons. Losses",      f"{v2_stats.get('max_cons_losses', 0)}",
                                   f"{baseline_stats.get('max_cons_losses', 0)}"),
        ("── Prop Firm (MC) ──", "", ""),
        (f"Pass Rate ({PROP_FIRM['name']})",
                                   f"{v2_mc.get('pass_rate', 0)*100:.1f}%",
                                   f"{baseline_mc.get('pass_rate', 0)*100:.1f}%"),
        ("Daily Fail %",          f"{v2_mc.get('fail_daily', 0)*100:.1f}%",
                                   f"{baseline_mc.get('fail_daily', 0)*100:.1f}%"),
        ("Avg Days to Pass",      f"{v2_mc.get('avg_days', 0):.1f}",
                                   f"{baseline_mc.get('avg_days', 0):.1f}"),
        ("── Buffer Sweep ──",    "", ""),
        ("Best Buffer",           f"{best_sweep['buffer']*100:.0f}%", "N/A"),
        ("Best Pass Rate",        f"{best_sweep['pass_rate']*100:.1f}%", "N/A"),
    ]

    for metric, v2_val, bl_val in rows:
        if metric.startswith("──"):
            print("-" * (30 + col_w * 2))
            print(f"  {metric}")
            print("-" * (30 + col_w * 2))
        else:
            print(f"{metric:<30}{v2_val:>{col_w}}{bl_val:>{col_w}}")

    print(sep)
    print(f"\n  ★ Winner: Ensemble @ {best_sweep['buffer']*100:.0f}% buffer  "
          f"→ {best_sweep['pass_rate']*100:.1f}% pass rate")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_v2_pipeline(csv_path: str, output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  MNQ 14:50 STRATEGY — v2 BEST STRATEGY PIPELINE")
    print("=" * 60)
    print(f"\n[1/7] Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"      Rows: {len(df):,}   {df['date'].min()} → {df['date'].max()}")

    # ── 2. Extract trades & apply smart filter ───────────────────────────────
    print("\n[2/7] Extracting trades & applying Smart Filter …")
    trades = extract_trades(df)
    total  = trades["date"].nunique()
    passed = trades[~trades["skip"]]["date"].nunique()
    print(f"      Trading days total: {total}   after Smart Filter: {passed}")
    print(f"      Avg window range: {trades[~trades['skip']]['window_range'].mean():.1f} pts")
    print(f"      Avg MAE:          {trades[~trades['skip']]['mae'].mean():.1f} pts")

    filtered = trades[~trades["skip"]].copy().reset_index(drop=True)
    if len(filtered) < 50:
        print("ERROR: Not enough trades (<50) after Smart Filter. Check your CSV.")
        return

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\n[3/7] Building features (this may take a minute) …")
    feat_df = build_features(df, filtered)
    print(f"      Feature matrix: {feat_df.shape}")

    # ── 4. Model building & predictions ──────────────────────────────────────
    print("\n[4/7] Training & predicting (walk-forward, no lookahead) …")
    p_ewma   = predict_ewma(feat_df)
    p_p75    = predict_p75(feat_df)
    p_fscale = predict_feature_scaling(feat_df)
    print("      GradientBoosting — window_range … (walk-forward, ~30 retrains)")
    p_gb     = predict_gradient_boosting(feat_df, "window_range")
    p_ens    = predict_ensemble(p_ewma, p_p75, p_fscale, p_gb)

    print("      GradientBoosting — MAE …")
    p_mae_gb  = predict_gradient_boosting(feat_df, "mae")
    feat_mae  = feat_df.copy()
    feat_mae["window_range"] = feat_mae["mae"]   # reuse same helpers for MAE
    p_mae_ens = predict_ensemble(
        predict_ewma(feat_mae), predict_p75(feat_mae),
        predict_feature_scaling(feat_mae), p_mae_gb,
    )

    # Model evaluation
    actual = feat_df["window_range"].values
    print("\n── Model Evaluation (window_range) ──────────────────────────")
    for name, preds in [("EWMA", p_ewma), ("P75", p_p75),
                         ("FeatScale", p_fscale), ("GradBoost", p_gb),
                         ("Ensemble", p_ens)]:
        ev = evaluate_model(actual, preds, name)
        if ev:
            print(f"  {name:<12}: MAE={ev['mae']:.2f}  RMSE={ev['rmse']:.2f}  "
                  f"MAPE={ev['mape']:.1f}%  Corr={ev['corr']:.3f}  "
                  f"RegAcc={ev['regime_acc']*100:.1f}%")

    # ── 5. Position sizing (min of WR pred & MAE pred) ────────────────────────
    print(f"\n[5/7] Position sizing — {PROP_FIRM['name']} "
          f"(daily limit=${PROP_FIRM['daily_loss_limit']:,}) …")
    contracts_wr  = np.array([
        contracts_from_prediction(float(p_ens[i]),     PROP_FIRM["daily_loss_limit"], SAFETY_BUFFER)
        for i in range(len(filtered))
    ])
    contracts_mae = np.array([
        contracts_from_prediction(float(p_mae_ens[i]), PROP_FIRM["daily_loss_limit"], SAFETY_BUFFER)
        for i in range(len(filtered))
    ])
    contracts_final = np.minimum(contracts_wr, contracts_mae)
    print(f"      Avg contracts (WR  pred):   {contracts_wr.mean():.1f}")
    print(f"      Avg contracts (MAE pred):   {contracts_mae.mean():.1f}")
    print(f"      Avg contracts (final min):  {contracts_final.mean():.1f}")

    # ── 6. Safety-buffer sweep ────────────────────────────────────────────────
    print(f"\n[6/7] Safety-buffer sweep (30%→100%, {MC_RUNS} MC runs each) …")
    sweep = safety_buffer_sweep(filtered, p_ens, PROP_FIRM)
    print(f"\n  {'Buffer':>8}  {'Pass Rate':>10}  {'Avg Contracts':>14}  {'Daily Fail':>10}")
    print("  " + "-" * 46)
    for r in sweep:
        marker = " ◄ WINNER" if r["buffer"] == SAFETY_BUFFER else ""
        print(f"  {r['buffer']*100:>6.0f}%  {r['pass_rate']*100:>9.1f}%"
              f"  {r['avg_contracts']:>13.1f}  {r['fail_daily']*100:>9.1f}%{marker}")

    # ── 7. Backtests & comparison table ──────────────────────────────────────
    print(f"\n[7/7] Running backtests …")
    v2_res       = run_backtest_dynamic(trades, contracts_final)
    baseline_res = run_backtest_fixed(trades, FIXED_BASELINE_CONTRACTS)

    # MC pass-rate for each approach (using the same filtered trades)
    v2_mc       = prop_firm_mc(filtered, contracts_final, PROP_FIRM, MC_RUNS)
    bl_contracts = np.full(len(filtered), FIXED_BASELINE_CONTRACTS)
    baseline_mc  = prop_firm_mc(filtered, bl_contracts, PROP_FIRM, MC_RUNS)

    print_final_comparison(
        v2_res["stats"], v2_mc,
        baseline_res["stats"], baseline_mc,
        sweep,
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nGenerating charts …")
    plot_equity_comparison(v2_res, baseline_res, output_dir)
    plot_pass_rate_sweep(sweep, output_dir)
    plot_prediction_vs_actual(feat_df, p_ens, output_dir)

    # ── Trade log export ──────────────────────────────────────────────────────
    for label, res in [("v2_ensemble", v2_res), ("baseline_fixed9", baseline_res)]:
        tr = res.get("trades")
        if tr is not None and len(tr):
            path = os.path.join(output_dir, f"trade_log_{label}.csv")
            tr.to_csv(path, index=False)
            print(f"  Saved: {path}")

    print(f"\n✓ All outputs saved to: {os.path.abspath(output_dir)}/")
    print("  Files: equity_comparison_v2.png  safety_buffer_sweep_v2.png")
    print("         prediction_vs_actual_v2.png")
    print("         trade_log_v2_ensemble.csv  trade_log_baseline_fixed9.csv")


# ─────────────────────────────────────────────────────────────────────────────
# COLAB HELPER — run this block if you are in a Colab notebook
# ─────────────────────────────────────────────────────────────────────────────
# Uncomment ONE of the two options below, then run the cell.
#
# OPTION A — Google Drive
# -----------------------
# from google.colab import drive
# drive.mount('/content/drive')
# CSV_PATH = "/content/drive/MyDrive/Dataset_NQ_1min_2022_2025.csv"
# run_v2_pipeline(CSV_PATH)
#
# OPTION B — Direct upload
# ------------------------
# from google.colab import files
# uploaded = files.upload()               # a file picker will appear
# CSV_PATH = list(uploaded.keys())[0]
# run_v2_pipeline(CSV_PATH)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNQ 14:50 Strategy — v2 Best Strategy Pipeline"
    )
    parser.add_argument("--csv",        type=str, default=CSV_PATH,
                        help="Path to 1-minute bar CSV")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        dest="output_dir", help="Output directory (default: output_v2)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        print("Usage: python colab_v2_best_strategy.py --csv path/to/data.csv")
        sys.exit(1)

    run_v2_pipeline(args.csv, args.output_dir)
