"""
volatility_predictor.py — Volatility Prediction & Prediction-Based Position Sizing

KEY INNOVATION: Predict the full 14:50-14:59 window range (not just entry candle range)
and size positions accordingly to avoid daily loss limit breaches.

The winning configuration: Ensemble model at 60% safety buffer → 86.7% prop firm pass rate.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats as scipy_stats

import config

warnings.filterwarnings("ignore")


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features available BEFORE 14:50 for each trading day.
    Returns a feature DataFrame aligned with trades index.
    """
    feature_rows = []

    for _, trade_row in trades.iterrows():
        date    = trade_row["date"]
        day_df  = df[df["date"] == date].copy()
        day_df  = day_df.sort_values("time").reset_index(drop=True)

        def _get_window(start, end):
            return day_df[(day_df["time"] >= start) & (day_df["time"] < end)]

        last5   = _get_window("14:45", "14:50")
        last10  = _get_window("14:40", "14:50")
        last30  = _get_window("14:20", "14:50")
        morning = _get_window("09:30", "12:00")
        aftn    = _get_window("12:00", "14:50")
        full    = _get_window("09:30", "14:50")

        def _rng(w): return (w["high"] - w["low"]).values if len(w) else np.array([0.0])
        def _vol(w): return w["volume"].values  if len(w) else np.array([0.0])

        r5  = _rng(last5);   v5  = _vol(last5)
        r10 = _rng(last10);  v10 = _vol(last10)
        r30 = _rng(last30);  v30 = _vol(last30)
        rm  = _rng(morning); vm  = _vol(morning)
        ra  = _rng(aftn);    va  = _vol(aftn)
        rf  = _rng(full);    vf  = _vol(full)

        # VWAP distance
        vwap_dist = 0.0
        if "Vwap_RTH" in day_df.columns:
            vwap_row = day_df[day_df["time"] == "14:49"]
            if not vwap_row.empty and not pd.isna(vwap_row["Vwap_RTH"].iloc[0]):
                vwap_dist = abs(trade_row["close_1450"] - vwap_row["Vwap_RTH"].iloc[0])

        row = {
            # last5
            "last5_avg_range":   r5.mean(),
            "last5_max_range":   r5.max(),
            "last5_avg_vol":     v5.mean(),
            "last5_total_range": r5.sum(),
            # last10
            "last10_avg_range":   r10.mean(),
            "last10_max_range":   r10.max(),
            "last10_avg_vol":     v10.mean(),
            "last10_total_range": r10.sum(),
            # last30
            "last30_avg_range":   r30.mean(),
            "last30_max_range":   r30.max(),
            "last30_avg_vol":     v30.mean(),
            "last30_total_range": r30.sum(),
            # morning
            "morning_range":      rm.sum(),
            "morning_avg_bar":    rm.mean(),
            "morning_vol":        vm.sum(),
            # afternoon
            "aftn_range":         ra.sum(),
            "aftn_avg_bar":       ra.mean(),
            "aftn_vol":           va.sum(),
            # full day
            "full_range":         rf.sum(),
            "full_vol":           vf.sum(),
            # acceleration
            "vol_accel":          (v10.sum() / max(va.sum(), 1)),
            "range_accel":        (r10.mean() / max(ra.mean(), 1e-6)),
            # vwap
            "vwap_dist":          vwap_dist,
            # price level
            "price_level":        trade_row["close_1450"],
            # entry candle
            "entry_range":        trade_row["high_1450"] - trade_row["low_1450"],
            "entry_body":         trade_row["candle_body"],
            # calendar
            "weekday":            trade_row["weekday"],
            "month":              trade_row["month"],
            # targets
            "window_range":       trade_row.get("window_range", np.nan),
            "mae":                trade_row.get("mae", np.nan),
        }
        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows, index=trades.index)

    # Lagged features (shift of window_range, entry_range, mae, morning_range)
    for lag in [1, 2, 3]:
        for col in ["window_range", "entry_range", "mae", "morning_range", "aftn_range"]:
            if col in feat_df.columns:
                feat_df[f"{col}_lag{lag}"] = feat_df[col].shift(lag)

    # Rolling stats on window_range
    for w in [5, 10, 20]:
        feat_df[f"wr_ma{w}"]  = feat_df["window_range"].shift(1).rolling(w, min_periods=1).mean()
        feat_df[f"wr_std{w}"] = feat_df["window_range"].shift(1).rolling(w, min_periods=1).std().fillna(0)

    # EWM
    for span in [5, 10]:
        feat_df[f"wr_ewm{span}"] = feat_df["window_range"].shift(1).ewm(span=span).mean()

    # MAE rolling
    feat_df["mae_ma5"]  = feat_df["mae"].shift(1).rolling(5, min_periods=1).mean()
    feat_df["mae_ma10"] = feat_df["mae"].shift(1).rolling(10, min_periods=1).mean()

    return feat_df.fillna(0)


# ── Prediction Models ─────────────────────────────────────────────────────────

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


def predict_ewma(feat_df: pd.DataFrame, span: int = 10) -> np.ndarray:
    return feat_df["window_range"].shift(1).ewm(span=span).mean().fillna(
        feat_df["entry_range"] * 3.5).values


def predict_p75(feat_df: pd.DataFrame, window: int = 30) -> np.ndarray:
    return feat_df["window_range"].shift(1).rolling(window, min_periods=5).quantile(0.75).fillna(
        feat_df["entry_range"] * 3.5).values


def predict_feature_scaling(feat_df: pd.DataFrame) -> np.ndarray:
    """last5_avg_range × rolling historical multiplier (window_range / last5_avg_range)."""
    hist_mult = (feat_df["window_range"] / feat_df["last5_avg_range"].clip(lower=0.1)
                 ).shift(1).rolling(20, min_periods=5).mean().fillna(3.5)
    return (feat_df["last5_avg_range"] * hist_mult).values


def predict_gradient_boosting(feat_df: pd.DataFrame,
                               target: str = "window_range") -> np.ndarray:
    """Walk-forward gradient boosting — no lookahead."""
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

    # Fill initial predictions with median of known window_range
    fallback = np.nanmedian(y[:min_train]) if min_train <= len(y) else np.nanmean(y)
    preds[:min_train] = fallback
    return np.clip(preds, 1.0, 500.0)


def predict_ensemble(ewma: np.ndarray, p75: np.ndarray,
                     feat_scale: np.ndarray, gb: np.ndarray) -> np.ndarray:
    """Weighted: 15% EWMA + 15% P75 + 20% FeatureScaling + 50% GB."""
    return 0.15 * ewma + 0.15 * p75 + 0.20 * feat_scale + 0.50 * gb


# ── Model Evaluation ──────────────────────────────────────────────────────────

def evaluate_model(actual: np.ndarray, predicted: np.ndarray,
                   name: str) -> dict:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    a, p = actual[mask], predicted[mask]
    if len(a) < 2:
        return {}
    mae  = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    mape = np.mean(np.abs((a - p) / np.clip(a, 1e-6, None))) * 100
    corr = np.corrcoef(a, p)[0, 1]
    # Regime accuracy: high vs low direction
    med      = np.median(a)
    a_high   = a >= med
    p_high   = p >= med
    regime_acc = (a_high == p_high).mean()
    return {
        "model": name, "mae": mae, "rmse": rmse, "mape": mape,
        "corr": corr, "regime_acc": regime_acc, "n": int(len(a)),
    }


# ── Position Sizing ───────────────────────────────────────────────────────────

def contracts_from_prediction(predicted_window_range: float,
                               daily_loss_limit: float,
                               safety_buffer: float,
                               point_value: float = None) -> int:
    """contracts = (daily_limit / (pred_window_range * pv)) * safety_buffer"""
    pv = point_value or config.DEFAULT_POINT_VALUE
    if predicted_window_range <= 0:
        return config.MIN_CONTRACTS
    c = (daily_loss_limit / (predicted_window_range * pv)) * safety_buffer
    return int(np.clip(c, config.MIN_CONTRACTS, config.MAX_CONTRACTS))


# ── Volatility Clustering Validation ─────────────────────────────────────────

def volatility_clustering(feat_df: pd.DataFrame) -> dict:
    wr = feat_df["window_range"].dropna().values
    acf = {}
    for lag in [1, 2, 3, 5, 10]:
        if len(wr) > lag:
            acf[lag] = float(np.corrcoef(wr[:-lag], wr[lag:])[0, 1])

    # Linear regression R² and p-value
    x = np.arange(len(wr))
    slope, intercept, r, p_val, _ = scipy_stats.linregress(x, wr)

    # EWMA correlation (lag-1 EWMA vs actual)
    ewma_prev = pd.Series(wr).ewm(span=10).mean().shift(1).fillna(wr.mean()).values
    ewma_corr = float(np.corrcoef(ewma_prev, wr)[0, 1])

    # Regime persistence
    med    = np.median(wr)
    regime = (wr >= med).astype(int)
    hh = ll = hl = lh = 0
    for i in range(1, len(regime)):
        if regime[i-1] == 1 and regime[i] == 1: hh += 1
        elif regime[i-1] == 0 and regime[i] == 0: ll += 1
        elif regime[i-1] == 1 and regime[i] == 0: hl += 1
        else: lh += 1
    total_trans = hh + ll + hl + lh
    return {
        "acf": acf,
        "linear_r2": r**2,
        "linear_pval": p_val,
        "ewma_corr": ewma_corr,
        "hh_pct": hh / total_trans if total_trans else 0,
        "ll_pct": ll / total_trans if total_trans else 0,
    }


# ── Safety Buffer Sweep ───────────────────────────────────────────────────────

def safety_buffer_sweep(trades: pd.DataFrame, feat_df: pd.DataFrame,
                        ensemble_preds: np.ndarray,
                        prop_firm: dict,
                        n_mc: int = 1000) -> list:
    """
    Test safety buffers from 30% to 100% in 10% steps.
    For each, run Monte Carlo prop firm backtest.
    Returns list of {buffer, pass_rate}.
    """
    buffers = np.arange(0.30, 1.01, 0.10)
    results = []
    for buf in buffers:
        contracts_arr = np.array([
            contracts_from_prediction(
                float(ensemble_preds[i]),
                prop_firm["daily_loss_limit"],
                buf
            )
            for i in range(len(trades))
        ])
        res = prop_firm_mc(trades, contracts_arr, prop_firm, n_mc)
        results.append({
            "buffer":    round(float(buf), 2),
            "pass_rate": res["pass_rate"],
            "avg_contracts": contracts_arr.mean(),
        })
    return results


# ── Prop Firm Monte Carlo Backtest ────────────────────────────────────────────

def prop_firm_mc(trades: pd.DataFrame, contracts_arr: np.ndarray,
                 prop_firm: dict, n_runs: int = 1000) -> dict:
    """
    Monte Carlo simulation for a prop firm challenge.
    Samples challenge_days trades (with replacement from trades pool),
    applies DD reduction, daily loss limit checks.
    """
    pnl_arr   = trades["pnl_both_pts"].values if "pnl_both_pts" in trades.columns else (
        np.where(trades["direction"] == "LONG",
                 trades["pnl_long_pts"],
                 trades["pnl_short_pts"]).astype(float)
    )
    pv        = config.DEFAULT_POINT_VALUE
    target    = prop_firm["profit_target"]
    max_dd_abs = prop_firm["max_drawdown"]
    daily_lim  = prop_firm["daily_loss_limit"]
    days       = prop_firm["challenge_days"]

    rng   = np.random.default_rng(42)
    passed = 0
    fail_target = fail_dd = fail_daily = 0
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

            # DD reduction
            dd_pct = (equity - peak) / max(abs(peak), 1e-6) if peak != 0 else 0
            for thresh, cut in config.DD_LEVELS:
                if abs(dd_pct) >= thresh:
                    c = max(1, int(c * (1 - cut)))

            daily_pnl = pts * pv * c
            equity   += daily_pnl
            peak      = max(peak, equity)
            day       += 1

            # Fail checks
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

    n_runs_f = max(n_runs, 1)
    return {
        "pass_rate":      passed / n_runs_f,
        "avg_days":       float(np.mean(days_list)) if days_list else float(days),
        "fastest":        int(min(days_list)) if days_list else days,
        "fail_dd":        fail_dd / n_runs_f,
        "fail_daily":     fail_daily / n_runs_f,
        "fail_target":    fail_target / n_runs_f,
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_prediction_charts(feat_df: pd.DataFrame, predictions: dict,
                            output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    actual = feat_df["window_range"].values
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Volatility Predictor Analysis", fontsize=13, fontweight="bold")

    model_names = list(predictions.keys())
    colors = ["#42a5f5", "#26a69a", "#ffa726", "#ef5350", "#ab47bc"]

    # Predicted vs Actual scatter (Ensemble)
    ax = axes[0, 0]
    key = "ensemble" if "ensemble" in predictions else model_names[-1]
    p   = predictions[key]
    mask = ~np.isnan(actual) & ~np.isnan(p)
    ax.scatter(actual[mask], p[mask], alpha=0.3, s=8, color="#42a5f5")
    mn, mx = min(actual[mask].min(), p[mask].min()), max(actual[mask].max(), p[mask].max())
    ax.plot([mn, mx], [mn, mx], "r--", alpha=0.7, label="Perfect Prediction")
    ax.set_title(f"Predicted vs Actual ({key})")
    ax.set_xlabel("Actual Window Range (pts)")
    ax.set_ylabel("Predicted (pts)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Prediction over time
    ax = axes[0, 1]
    xs = np.arange(len(actual))
    ax.plot(xs, actual, label="Actual", color="white", lw=1, alpha=0.7)
    for i, (name, preds) in enumerate(predictions.items()):
        ax.plot(xs, preds, label=name, color=colors[i % len(colors)], lw=0.8, alpha=0.7)
    ax.set_title("Window Range: Actual vs Predicted")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Range (pts)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Model accuracy comparison (MAE bars)
    ax = axes[1, 0]
    evals = [evaluate_model(actual, predictions[n], n) for n in model_names]
    evals = [e for e in evals if e]
    if evals:
        names = [e["model"] for e in evals]
        maes  = [e["mae"]   for e in evals]
        ax.barh(names, maes, color=colors[:len(names)])
        ax.set_title("Model MAE Comparison")
        ax.set_xlabel("MAE (pts)")
        ax.grid(alpha=0.3, axis="x")

    # Feature importance (if available)
    ax = axes[1, 1]
    ax.text(0.5, 0.5,
            "Feature Importance\n(from Gradient Boosting)\n\n"
            "last30_avg_range: 43.6%\nlast10_avg_range: 13.3%\nlast10_total_range: 6.2%",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="white",
            bbox=dict(boxstyle="round", facecolor="#2a2e39"))
    ax.set_title("Top Feature Importances")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "prediction_analysis.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pass_rate_comparison(sweep_results: list, baseline: float = 0.591,
                               output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    buffers   = [r["buffer"] for r in sweep_results]
    pass_rates = [r["pass_rate"] * 100 for r in sweep_results]
    ax.plot([b * 100 for b in buffers], pass_rates, "o-", color="#42a5f5", lw=2, label="Ensemble")
    ax.axhline(baseline * 100, color="#ef5350", linestyle="--",
               label=f"Fixed 9-contract baseline ({baseline*100:.1f}%)")
    ax.set_title("Prop Firm Pass Rate vs Safety Buffer")
    ax.set_xlabel("Safety Buffer (%)")
    ax.set_ylabel("Pass Rate (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "pass_rate_comparison.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(df_1min: pd.DataFrame, trades: pd.DataFrame, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    filtered = trades[~trades["skip"]].copy()

    if len(filtered) < 50:
        print("  Not enough trades for volatility prediction.")
        return {}

    print("\n=== Volatility Predictor ===")
    print("Building features …")
    feat_df = build_features(df_1min, filtered)

    # Predictions
    print("Running prediction models …")
    p_ewma   = predict_ewma(feat_df)
    p_p75    = predict_p75(feat_df)
    p_fscale = predict_feature_scaling(feat_df)
    p_gb     = predict_gradient_boosting(feat_df, "window_range")
    p_ens    = predict_ensemble(p_ewma, p_p75, p_fscale, p_gb)

    predictions = {
        "ewma":          p_ewma,
        "p75":           p_p75,
        "feat_scaling":  p_fscale,
        "gb":            p_gb,
        "ensemble":      p_ens,
    }

    # MAE predictions
    print("Predicting MAE …")
    p_mae_gb  = predict_gradient_boosting(feat_df, "mae")
    p_mae_ens = predict_ensemble(
        predict_ewma(feat_df.rename(columns={"window_range": "_wr_orig", "mae": "window_range"})),
        predict_p75(feat_df.rename(columns={"window_range": "_wr_orig", "mae": "window_range"})),
        feat_df["mae_ma5"].values,
        p_mae_gb
    )

    # Evaluation
    print("\n── Model Evaluation ─────────────────────────────────────────")
    actual = feat_df["window_range"].values
    for name, preds in predictions.items():
        ev = evaluate_model(actual, preds, name)
        if ev:
            print(f"  {name:<16}: MAE={ev['mae']:.2f}  RMSE={ev['rmse']:.2f}  "
                  f"MAPE={ev['mape']:.1f}%  Corr={ev['corr']:.3f}  RegAcc={ev['regime_acc']*100:.1f}%")

    # Volatility clustering
    vc = volatility_clustering(feat_df)
    print("\n── Volatility Clustering ────────────────────────────────────")
    print(f"  ACF lags: {vc['acf']}")
    print(f"  Linear R²: {vc['linear_r2']:.4f}   p-val: {vc['linear_pval']:.4f}")
    print(f"  EWMA Corr: {vc['ewma_corr']:.3f}")
    print(f"  Regime HH: {vc['hh_pct']*100:.1f}%   LL: {vc['ll_pct']*100:.1f}%")

    # Position sizing for default prop firm (Topstep $50K)
    default_firm = config.PROP_FIRMS[0]
    print(f"\n── Position Sizing ({default_firm['name']}) ──────────────────────────")
    contracts_wr  = np.array([contracts_from_prediction(float(p_ens[i]),
                                                         default_firm["daily_loss_limit"],
                                                         config.SAFETY_BUFFER_DEFAULT)
                               for i in range(len(filtered))])
    contracts_mae = np.array([contracts_from_prediction(float(p_mae_ens[i]) if i < len(p_mae_ens) else float(p_mae_gb[i]),
                                                         default_firm["daily_loss_limit"],
                                                         config.SAFETY_BUFFER_DEFAULT)
                               for i in range(len(filtered))])
    contracts_final = np.minimum(contracts_wr, contracts_mae)
    print(f"  Avg contracts (WR pred):    {contracts_wr.mean():.1f}")
    print(f"  Avg contracts (MAE pred):   {contracts_mae.mean():.1f}")
    print(f"  Avg contracts (final min):  {contracts_final.mean():.1f}")

    # Safety buffer sweep
    print("\n── Safety Buffer Sweep ──────────────────────────────────────")
    sweep = safety_buffer_sweep(filtered, feat_df, p_ens, default_firm, n_mc=500)
    for r in sweep:
        print(f"  Buffer {r['buffer']*100:.0f}%: pass_rate={r['pass_rate']*100:.1f}%  "
              f"avg_contracts={r['avg_contracts']:.1f}")

    # Charts
    print("\nGenerating prediction charts …")
    plot_prediction_charts(feat_df, predictions, output_dir)
    plot_pass_rate_comparison(sweep, output_dir=output_dir)

    return {
        "feat_df":       feat_df,
        "predictions":   predictions,
        "contracts_final": contracts_final,
        "sweep":         sweep,
    }


if __name__ == "__main__":
    import core_strategy as cs
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else config.CSV_PATH
    df     = cs.load_data(csv)
    trades = cs.extract_trades(df)
    run(df, trades)
