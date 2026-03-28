"""
selection_bias.py — Selection Bias Analysis for Multi-Window Backtest

⚠  WARNING: Picking the best-performing windows from 381 backtest
possibilities creates a risk of selection bias (multiple-comparisons
problem).  Some windows will show strong statistics purely by chance.
This module quantifies and mitigates that risk.

Three complementary methods:

  1. Walk-forward cross-validation (pseudo out-of-sample test)
     Divide trading days into N folds.  In each fold, scan *only prior
     data* to pick the top-5 windows by Sharpe, then measure those same
     windows' performance out-of-sample in the next fold.  Aggregate
     across all folds to compare IS vs OOS performance.

  2. Null-distribution simulation (false-positive risk)
     Flip 381 independent "fair coins" (zero-mean PnL, n≈225 trades
     each) and record the best Sharpe and win-rate achieved across all
     381 windows.  Repeat 10,000 times to build the distribution of the
     maximum metric under H0: no real edge.  This answers:
     "How good can a lucky window look purely by chance?"

  3. Shrinkage estimates (conservative forward projection)
     Apply regression-to-the-mean shrinkage that accounts for the
     number of candidate windows tested.  The larger the scan (more
     comparisons), the stronger the shrinkage toward the null.

References
----------
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
- Harvey, Liu & Zhu (2016) "… and the Cross-Section of Expected Returns"
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

import config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HOLD_MINUTES         = 9          # default hold period (minutes)
ENTRY_START          = (9, 30)    # earliest entry time (h, m)
ENTRY_END            = (15, 50)   # latest entry time (h, m) so exit ≤ 15:59
MIN_ENTRY_TRADES     = 20         # skip windows with fewer qualifying trades
# Minimum IS→OOS Sharpe retention considered acceptable (below = red flag)
MIN_SHARPE_RETENTION = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(arr: np.ndarray) -> float:
    """Annualised Sharpe ratio (sqrt-252 convention, matches existing code)."""
    if len(arr) < 2:
        return 0.0
    s = arr.std(ddof=1)
    return float(arr.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def _window_stats(arr: np.ndarray) -> dict:
    """Return a standard stats dict for a PnL array (dollars)."""
    if len(arr) == 0:
        return {"n": 0, "win_rate": 0.0, "avg_pnl": 0.0,
                "total_pnl": 0.0, "profit_factor": 0.0, "sharpe": 0.0}
    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    gp     = float(wins.sum())
    gl     = float(abs(losses.sum()))
    return {
        "n":             len(arr),
        "win_rate":      float((arr > 0).mean()),
        "avg_pnl":       float(arr.mean()),
        "total_pnl":     float(arr.sum()),
        "profit_factor": gp / gl if gl > 0 else 0.0,
        "sharpe":        _sharpe(arr),
    }


def _all_entry_times(hold_min: int = HOLD_MINUTES):
    """All (hour, minute) entry pairs from ENTRY_START to ENTRY_END."""
    sh, sm = ENTRY_START
    eh, em = ENTRY_END
    start  = sh * 60 + sm
    end    = eh * 60 + em
    pairs  = []
    for t in range(start, end + 1):
        xh = (t + hold_min) // 60
        xm = (t + hold_min) % 60
        if xh > 16 or (xh == 16 and xm > 0):
            continue   # exit after market close
        pairs.append((t // 60, t % 60, xh, xm))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Window Scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_windows(
    df: pd.DataFrame,
    dates=None,
    hold_minutes: int = HOLD_MINUTES,
    skip_thursdays: bool = None,
    skip_months: list = None,
    min_body: float = None,
    point_value: float = None,
    min_trades: int = MIN_ENTRY_TRADES,
) -> pd.DataFrame:
    """
    Scan every possible entry window (from ENTRY_START to ENTRY_END,
    exiting `hold_minutes` later) over the given trading *dates*.

    Parameters
    ----------
    df            : 1-minute bar DataFrame from ``core_strategy.load_data``
    dates         : list of date objects to include (default: all in df)
    hold_minutes  : hold period in minutes (default: 9)
    skip_thursdays: if True, skip Thursday entries (default: config value)
    skip_months   : list of months to skip (default: config value)
    min_body      : minimum candle body in points (default: config value)
    point_value   : $/point for the instrument (default: config value)
    min_trades    : windows with fewer trades are excluded from results

    Returns
    -------
    DataFrame sorted by Sharpe (descending), one row per window.
    """
    if skip_thursdays is None:
        skip_thursdays = config.SKIP_THURSDAYS
    skip_months_ = config.SKIP_MONTHS          if skip_months is None  else skip_months
    min_body_    = config.MIN_CANDLE_BODY      if min_body    is None  else min_body
    pv           = config.DEFAULT_POINT_VALUE  if point_value is None  else point_value

    all_dates = sorted(df["date"].unique()) if dates is None else sorted(dates)

    # ── Apply day-level filters ───────────────────────────────────────────────
    df_sub = df[df["date"].isin(set(all_dates))].copy()
    if "time" in df_sub.columns:
        df_sub["sb_t"] = df_sub["time"]
    else:
        df_sub["sb_t"] = df_sub["timestamp"].dt.strftime("%H:%M")

    # Identify which dates pass the skip filters
    day_meta = (
        df_sub.groupby("date")
        .first()[["weekday", "month"]]
        .reset_index()
    )
    valid_dates = set()
    for row in day_meta.itertuples(index=False):
        if skip_thursdays and row.weekday == 3:
            continue
        if row.month in skip_months_:
            continue
        valid_dates.add(row.date)

    # ── Build fast (date, time_str) → (open, close) lookup ───────────────────
    df_valid = df_sub[df_sub["date"].isin(valid_dates)][
        ["date", "sb_t", "open", "close"]
    ]
    lookup: dict = {}
    for r in df_valid.itertuples(index=False):
        lookup[(r.date, r.sb_t)] = (r.open, r.close)

    # ── Scan every window ─────────────────────────────────────────────────────
    valid_dates_list = sorted(valid_dates)
    entry_times      = _all_entry_times(hold_minutes)
    rows             = []

    for eh, em, xh, xm in entry_times:
        entry_str = f"{eh:02d}:{em:02d}"
        exit_str  = f"{xh:02d}:{xm:02d}"

        pnls: list = []
        for d in valid_dates_list:
            eb = lookup.get((d, entry_str))
            xb = lookup.get((d, exit_str))
            if eb is None or xb is None:
                continue
            o, c = eb
            body = abs(c - o)
            if body < min_body_ or c == o:
                continue
            direction = 1 if c > o else -1
            pnls.append((xb[1] - c) * direction * pv)

        if len(pnls) < min_trades:
            continue

        arr = np.array(pnls, dtype=float)
        st  = _window_stats(arr)
        rows.append({
            "entry_time": entry_str,
            "exit_time":  exit_str,
            **st,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Walk-Forward Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    n_folds: int = None,
    top_n: int = None,
    expanding: bool = True,
    hold_minutes: int = HOLD_MINUTES,
    point_value: float = None,
) -> dict:
    """
    Pseudo out-of-sample test via walk-forward cross-validation.

    For each test fold k (k = 1 … n_folds-1):
      - **Train**: all prior dates → scan all windows, pick ``top_n``
        by in-sample Sharpe.
      - **Test**: fold-k dates   → evaluate those *same* windows
        out-of-sample.

    Parameters
    ----------
    df          : 1-minute bar DataFrame
    n_folds     : number of equal-length folds (default: config.SB_N_FOLDS)
    top_n       : windows to select in-sample each fold (default: config.SB_TOP_N)
    expanding   : True → expanding train window; False → single-prior-fold train
    hold_minutes: hold period (default: 9 min)
    point_value : $/point (default: config value)

    Returns
    -------
    dict with keys:
      ``folds``     — list of per-fold result dicts
      ``aggregate`` — mean IS vs OOS stats across all folds
    """
    n_folds = n_folds or config.SB_N_FOLDS
    top_n   = top_n   or config.SB_TOP_N
    pv      = config.DEFAULT_POINT_VALUE if point_value is None else point_value

    all_dates = sorted(df["date"].unique())
    n_days    = len(all_dates)
    fold_size = n_days // n_folds

    # Build fold date boundaries
    boundaries = []
    for k in range(n_folds):
        s = k * fold_size
        e = s + fold_size if k < n_folds - 1 else n_days
        boundaries.append((s, e))

    fold_results = []

    for k in range(1, n_folds):
        ts, te      = boundaries[k]
        test_dates  = all_dates[ts:te]

        if expanding:
            train_dates = all_dates[:ts]
        else:
            rs, re      = boundaries[k - 1]
            train_dates = all_dates[rs:re]

        if len(train_dates) < 30 or len(test_dates) < 10:
            continue

        # Step 1 — scan train
        train_scan = scan_windows(
            df, dates=train_dates, hold_minutes=hold_minutes, point_value=pv
        )
        if train_scan.empty:
            continue

        # Step 2 — pick top_n windows by IS Sharpe
        top_windows = train_scan.head(top_n)["entry_time"].tolist()

        # Step 3 — scan test (allow fewer trades per window to avoid NaN folds)
        test_scan = scan_windows(
            df, dates=test_dates, hold_minutes=hold_minutes,
            point_value=pv, min_trades=5,
        )

        is_rows  = train_scan[train_scan["entry_time"].isin(top_windows)]
        oos_rows = (test_scan[test_scan["entry_time"].isin(top_windows)]
                    if not test_scan.empty else pd.DataFrame())

        def _agg(sub: pd.DataFrame) -> dict:
            if sub.empty:
                return {"avg_sharpe": 0.0, "avg_win_rate": 0.0, "avg_pnl": 0.0}
            return {
                "avg_sharpe":   float(sub["sharpe"].mean()),
                "avg_win_rate": float(sub["win_rate"].mean()),
                "avg_pnl":      float(sub["avg_pnl"].mean()),
            }

        fold_results.append({
            "fold":        k,
            "train_days":  len(train_dates),
            "test_days":   len(test_dates),
            "top_windows": top_windows,
            "is_stats":    _agg(is_rows),
            "oos_stats":   _agg(oos_rows),
        })

    # ── Aggregate across folds ────────────────────────────────────────────────
    if fold_results:
        is_sharpes  = [f["is_stats"]["avg_sharpe"]   for f in fold_results]
        oos_sharpes = [f["oos_stats"]["avg_sharpe"]  for f in fold_results]
        is_wrs      = [f["is_stats"]["avg_win_rate"] for f in fold_results]
        oos_wrs     = [f["oos_stats"]["avg_win_rate"] for f in fold_results]
        is_pnls     = [f["is_stats"]["avg_pnl"]      for f in fold_results]
        oos_pnls    = [f["oos_stats"]["avg_pnl"]     for f in fold_results]

        is_sh_mean  = float(np.mean(is_sharpes))
        oos_sh_mean = float(np.mean(oos_sharpes))
        retention   = float(oos_sh_mean / is_sh_mean) if is_sh_mean != 0 else 0.0

        aggregate = {
            "is_sharpe_mean":   is_sh_mean,
            "oos_sharpe_mean":  oos_sh_mean,
            "is_wr_mean":       float(np.mean(is_wrs)),
            "oos_wr_mean":      float(np.mean(oos_wrs)),
            "is_pnl_mean":      float(np.mean(is_pnls)),
            "oos_pnl_mean":     float(np.mean(oos_pnls)),
            "sharpe_retention": retention,
            "n_folds":          len(fold_results),
        }
    else:
        aggregate = {}

    return {"folds": fold_results, "aggregate": aggregate}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Null Distribution Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_null_distribution(
    n_windows: int = None,
    n_trades: int = None,
    n_sim: int = None,
    pnl_std: float = None,
) -> dict:
    """
    Simulate the null distribution of the best window under H0: no real edge.

    In each simulation, ``n_windows`` independent zero-mean PnL series are
    drawn (each with ``n_trades`` observations).  The maximum win-rate and
    maximum Sharpe ratio across all windows are recorded.  After ``n_sim``
    repetitions the full sampling distribution is available.

    Key insight: the Sharpe ratio of a single window with n trades under H0
    follows approximately N(0, sqrt(252/n)).  For n=225, that is ≈ N(0, 1.06).
    The expected *maximum* across 381 such windows is ≈ 3.5, which means
    observing the "best" Sharpe > 2.5 is highly likely even with no real
    edge when 381 windows are scanned — a direct measure of selection bias.

    Parameters
    ----------
    n_windows : number of candidate windows tested (default: config.SB_N_WINDOWS)
    n_trades  : pseudo-trades per window (default: config.SB_N_TRADES_PER_WINDOW)
    n_sim     : Monte Carlo iterations (default: config.SB_NULL_SIMS)
    pnl_std   : $/trade std for null draws (default: config.SB_NULL_PNL_STD)

    Returns
    -------
    dict with percentile distributions and false-positive probabilities.
    """
    n_windows = n_windows or config.SB_N_WINDOWS
    n_trades  = n_trades  or config.SB_N_TRADES_PER_WINDOW
    n_sim     = n_sim     or config.SB_NULL_SIMS
    pnl_std_  = pnl_std   if pnl_std is not None else config.SB_NULL_PNL_STD

    rng = np.random.default_rng(42)

    max_wrs     = np.empty(n_sim)
    max_sharpes = np.empty(n_sim)

    for i in range(n_sim):
        # Shape: (n_windows, n_trades)
        pnls = rng.normal(0.0, pnl_std_, size=(n_windows, n_trades))

        wrs  = (pnls > 0).mean(axis=1)                          # win-rate per window
        stds = pnls.std(axis=1, ddof=1)
        shs  = np.where(stds > 0,
                        pnls.mean(axis=1) / stds * np.sqrt(252),
                        0.0)

        max_wrs[i]     = wrs.max()
        max_sharpes[i] = shs.max()

    def _pct(arr: np.ndarray) -> dict:
        return {p: float(np.percentile(arr, p)) for p in [50, 75, 90, 95, 99]}

    # Theoretical expected max Sharpe under H0 (for sanity check)
    # Each window Sharpe ~ N(0, sqrt(252/n_trades))
    # E[max of N N(0,1)] ≈ sqrt(2*ln(N))
    null_sh_std  = np.sqrt(252 / n_trades)
    expected_max = null_sh_std * np.sqrt(2 * np.log(n_windows))

    return {
        "n_windows":            n_windows,
        "n_trades":             n_trades,
        "n_sim":                n_sim,
        "max_wr":               _pct(max_wrs),
        "max_sharpe":           _pct(max_sharpes),
        "max_wrs_raw":          max_wrs,
        "max_sharpes_raw":      max_sharpes,
        "expected_max_sharpe":  float(expected_max),
        # False-positive probabilities
        "p_wr_56":     float((max_wrs >= 0.56).mean()),
        "p_wr_59":     float((max_wrs >= 0.59).mean()),
        "p_sharpe_25": float((max_sharpes >= 2.5).mean()),
        "p_sharpe_30": float((max_sharpes >= 3.0).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Shrinkage Estimates
# ─────────────────────────────────────────────────────────────────────────────

def shrinkage_estimate(
    observed_sharpe: float,
    observed_wr: float,
    observed_avg_pnl: float,
    n_windows: int = None,
    n_trades: int = None,
) -> dict:
    """
    Apply regression-to-the-mean (Bayesian shrinkage) to backtested metrics.

    Rationale
    ---------
    When we pick the best result from a large set of candidates, the
    observed performance is *inflated* relative to the true expected
    performance.  A simple shrinkage formula accounts for this:

        shrunk = null_value + shrinkage_factor × (observed - null_value)

    where::

        shrinkage_factor = 1 / (1 + sqrt(n_windows) / sqrt(n_trades))

    A large ``n_windows`` (many comparisons) → strong shrinkage toward null.
    A large ``n_trades`` (more data per window) → weaker shrinkage.

    For n_windows=381, n_trades=225:  sf ≈ 0.43
    → only ~43 % of the apparent edge above null is expected to persist.

    Parameters
    ----------
    observed_sharpe  : best-window Sharpe from the full backtest scan
    observed_wr      : best-window win-rate (0–1)
    observed_avg_pnl : best-window average $/trade
    n_windows        : candidate windows tested (default: config.SB_N_WINDOWS)
    n_trades         : approx trades per window (default: config.SB_N_TRADES_PER_WINDOW)

    Returns
    -------
    dict with shrinkage factor and both raw / shrunk estimates.
    """
    n_windows = n_windows or config.SB_N_WINDOWS
    n_trades  = n_trades  or config.SB_N_TRADES_PER_WINDOW

    sf = 1.0 / (1.0 + np.sqrt(n_windows) / np.sqrt(n_trades))

    # Null baselines (no-edge scenario)
    null_sharpe = 0.0
    null_wr     = 0.50
    null_pnl    = 0.0

    return {
        "shrinkage_factor":  float(sf),
        "n_windows":         n_windows,
        "n_trades":          n_trades,
        "observed_sharpe":   observed_sharpe,
        "shrunk_sharpe":     float(null_sharpe + sf * (observed_sharpe - null_sharpe)),
        "observed_wr":       observed_wr,
        "shrunk_wr":         float(null_wr     + sf * (observed_wr     - null_wr)),
        "observed_avg_pnl":  observed_avg_pnl,
        "shrunk_avg_pnl":    float(null_pnl    + sf * (observed_avg_pnl - null_pnl)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print / Report Functions
# ─────────────────────────────────────────────────────────────────────────────

def print_wf_cv_results(wf_cv: dict):
    """Print walk-forward cross-validation results."""
    folds = wf_cv.get("folds", [])
    agg   = wf_cv.get("aggregate", {})

    print("\n── Walk-Forward Cross-Validation (Selection Bias Test) ──────")
    print("  ⚠  Top windows are chosen IN-SAMPLE, then evaluated OUT-OF-SAMPLE.")
    print()
    hdr = (
        f"  {'Fold':>4}  {'Train':>6}  {'Test':>6}  "
        f"{'IS Sharpe':>10}  {'OOS Sharpe':>10}  "
        f"{'IS WR%':>7}  {'OOS WR%':>7}  "
        f"{'IS$/tr':>7}  {'OOS$/tr':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in folds:
        iss  = r["is_stats"]
        ooss = r["oos_stats"]
        print(
            f"  {r['fold']:>4}  "
            f"{r['train_days']:>6}  "
            f"{r['test_days']:>6}  "
            f"{iss['avg_sharpe']:>10.2f}  "
            f"{ooss['avg_sharpe']:>10.2f}  "
            f"{iss['avg_win_rate']*100:>6.1f}%  "
            f"{ooss['avg_win_rate']*100:>6.1f}%  "
            f"${iss['avg_pnl']:>6.2f}  "
            f"${ooss['avg_pnl']:>6.2f}"
        )
    if agg:
        print(f"\n  Aggregate ({agg['n_folds']} folds):")
        print(
            f"    In-sample:    Sharpe {agg['is_sharpe_mean']:.2f}  "
            f"WR {agg['is_wr_mean']*100:.1f}%  "
            f"Avg P&L ${agg['is_pnl_mean']:.2f}"
        )
        print(
            f"    Out-of-sample: Sharpe {agg['oos_sharpe_mean']:.2f}  "
            f"WR {agg['oos_wr_mean']*100:.1f}%  "
            f"Avg P&L ${agg['oos_pnl_mean']:.2f}"
        )
        pct   = agg["sharpe_retention"] * 100
        label = (
            "✓ edge persists OOS"
            if agg["sharpe_retention"] > MIN_SHARPE_RETENTION
            else "⚠  significant IS→OOS decay"
        )
        print(f"    Sharpe retention: {pct:.0f}%   {label}")


def print_null_distribution(nd: dict):
    """Print null distribution simulation results with false-positive warnings."""
    print("\n── Null Distribution: Max Performance by Luck ───────────────")
    print(
        f"  Simulated {nd['n_sim']:,} × {nd['n_windows']} independent "
        f"fair-coin windows ({nd['n_trades']} pseudo-trades each, mean=$0)"
    )
    print(
        f"  Theoretical expected max Sharpe under H0: "
        f"{nd['expected_max_sharpe']:.2f}"
    )
    print()
    print("  Best win-rate achievable by luck (percentiles of max WR):")
    for p, v in nd["max_wr"].items():
        print(f"    P{p:2d}: {v*100:.1f}%")
    print()
    print("  Best Sharpe achievable by luck (percentiles of max Sharpe):")
    for p, v in nd["max_sharpe"].items():
        print(f"    P{p:2d}: {v:.3f}")
    print()
    print("  Probability that a random scan produces at least one window with:")
    print(f"    WR ≥ 56%:     {nd['p_wr_56']*100:.1f}%")
    print(f"    WR ≥ 59%:     {nd['p_wr_59']*100:.1f}%")
    print(f"    Sharpe ≥ 2.5: {nd['p_sharpe_25']*100:.1f}%")
    print(f"    Sharpe ≥ 3.0: {nd['p_sharpe_30']*100:.1f}%")
    print()
    print(
        "  ⚠  If these probabilities are HIGH (>50%), the observed metrics"
        "\n     are likely within reach of pure luck and must be validated OOS."
    )


def print_shrinkage(sh: dict):
    """Print shrinkage-adjusted performance estimates."""
    print("\n── Shrinkage Estimates (Conservative Forward Projection) ─────")
    sf = sh["shrinkage_factor"]
    print(
        f"  Shrinkage factor: {sf:.2f}  "
        f"(scanned {sh['n_windows']} windows × {sh['n_trades']} trades each)"
    )
    print(
        f"  Interpretation: only {sf*100:.0f}% of the apparent edge above"
        "\n  the null baseline is expected to survive in live trading."
    )
    print()
    print(f"  {'Metric':<20}  {'Backtest (IS)':>14}  {'Shrunk est.':>12}  {'Δ':>8}")
    print("  " + "-" * 60)
    rows = [
        ("Sharpe Ratio",   sh["observed_sharpe"],       sh["shrunk_sharpe"],    ""),
        ("Win Rate",        sh["observed_wr"] * 100,     sh["shrunk_wr"] * 100, "%"),
        ("Avg P&L/trade",   sh["observed_avg_pnl"],      sh["shrunk_avg_pnl"],  "$"),
    ]
    for label, obs, shrunk, unit in rows:
        delta = shrunk - obs
        if unit == "%":
            print(f"  {label:<20}  {obs:>13.1f}%  {shrunk:>11.1f}%  {delta:>+7.1f}%")
        elif unit == "$":
            print(f"  {label:<20}  ${obs:>13.2f}  ${shrunk:>11.2f}  {delta:>+8.2f}")
        else:
            print(f"  {label:<20}  {obs:>14.2f}  {shrunk:>12.2f}  {delta:>+8.2f}")
    print()
    print("  ⚠  Use the shrunk estimates as a conservative lower bound for")
    print("     real-world performance expectations.")


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_bias_charts(
    wf_cv: dict,
    null_dist: dict,
    shrinkage: dict,
    output_dir: str = ".",
):
    """
    Save a three-panel selection-bias summary chart:
      1. IS vs OOS Sharpe per walk-forward fold
      2. Null distribution of the maximum Sharpe across 381 windows
      3. Backtested vs shrunk-estimate bar chart
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1e1e2e")
    fig.suptitle("Selection Bias Analysis", fontsize=14, fontweight="bold",
                 color="white")

    # ── Panel 1: IS vs OOS Sharpe per fold ───────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#2a2a3e")
    folds = wf_cv.get("folds", [])
    if folds:
        nums       = [f["fold"] for f in folds]
        is_sharpes  = [f["is_stats"]["avg_sharpe"]  for f in folds]
        oos_sharpes = [f["oos_stats"]["avg_sharpe"] for f in folds]
        x = np.arange(len(nums))
        w = 0.35
        ax.bar(x - w / 2, is_sharpes,  w, label="In-sample",     color="#42a5f5", alpha=0.85)
        ax.bar(x + w / 2, oos_sharpes, w, label="Out-of-sample", color="#26a69a", alpha=0.85)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {n}" for n in nums], color="white")
        ax.tick_params(colors="white")
        ax.set_ylabel("Avg Sharpe (top windows)", color="white")
        ax.set_title("IS vs OOS Sharpe\n(Walk-Forward CV)", color="white")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    # ── Panel 2: Null distribution of max Sharpe ─────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#2a2a3e")
    raw = null_dist.get("max_sharpes_raw", np.array([]))
    if len(raw):
        ax.hist(raw, bins=60, color="#ef5350", alpha=0.8, edgecolor="none",
                label="Max Sharpe by luck")
        obs = shrinkage.get("observed_sharpe", 0.0)
        ax.axvline(obs, color="gold", linewidth=2,
                   label=f"Observed IS ({obs:.2f})")
        shrunk = shrinkage.get("shrunk_sharpe", 0.0)
        ax.axvline(shrunk, color="white", linewidth=2, linestyle="--",
                   label=f"Shrunk ({shrunk:.2f})")
        p95 = null_dist["max_sharpe"].get(95, 0.0)
        ax.axvline(p95, color="#ff9800", linewidth=1.5, linestyle=":",
                   label=f"Null P95 ({p95:.2f})")
        ax.set_xlabel("Max Sharpe across all windows", color="white")
        ax.set_title(
            f"Null Distribution of Max Sharpe\n"
            f"({null_dist['n_windows']} random windows, "
            f"{null_dist['n_trades']} trades each)",
            color="white",
        )
        ax.tick_params(colors="white")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    # ── Panel 3: Backtested vs shrunk metrics ─────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#2a2a3e")
    labels = ["Sharpe", "WR (%)", "Avg P&L ($)"]
    obs_v  = [
        shrinkage["observed_sharpe"],
        shrinkage["observed_wr"] * 100,
        shrinkage["observed_avg_pnl"],
    ]
    shr_v  = [
        shrinkage["shrunk_sharpe"],
        shrinkage["shrunk_wr"] * 100,
        shrinkage["shrunk_avg_pnl"],
    ]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, obs_v, w, label="Backtested (IS)", color="#42a5f5", alpha=0.85)
    ax.bar(x + w / 2, shr_v, w, label="Shrunk (realistic)", color="#ff9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white")
    ax.tick_params(colors="white")
    ax.set_title(
        f"Shrinkage Adjustment\n(factor = {shrinkage['shrinkage_factor']:.2f})",
        color="white",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    plt.tight_layout()
    path = os.path.join(output_dir, "selection_bias.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    n_folds: int = None,
    top_n: int = None,
    n_windows: int = None,
    output_dir: str = ".",
) -> dict:
    """
    Full selection bias analysis pipeline.

    Parameters
    ----------
    df          : 1-minute bar DataFrame (from ``core_strategy.load_data``)
    n_folds     : walk-forward folds (default: config.SB_N_FOLDS)
    top_n       : top windows selected in-sample (default: config.SB_TOP_N)
    n_windows   : total windows scanned historically (default: config.SB_N_WINDOWS)
    output_dir  : directory for output chart

    Returns
    -------
    dict with keys ``walk_forward_cv``, ``null_distribution``, ``shrinkage``
    """
    n_folds   = n_folds   or config.SB_N_FOLDS
    top_n     = top_n     or config.SB_TOP_N
    n_windows = n_windows or config.SB_N_WINDOWS
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Selection Bias Analysis ===")
    print(
        f"\n  ⚠  WARNING: Scanning {n_windows} windows introduces a multiple-"
        "\n     comparisons problem.  Some windows may appear profitable purely"
        "\n     by chance.  This module quantifies that risk."
    )

    # ── 1. Walk-Forward Cross-Validation ─────────────────────────────────────
    print(
        f"\n  [1/3] Walk-forward CV  "
        f"({n_folds} folds, top {top_n} windows selected IS)"
    )
    print("       Scanning all windows per fold — please wait …")
    wf_cv = walk_forward_cv(df, n_folds=n_folds, top_n=top_n)
    print_wf_cv_results(wf_cv)

    # ── 2. Null Distribution ──────────────────────────────────────────────────
    print(f"\n  [2/3] Null distribution simulation ({config.SB_NULL_SIMS:,} trials) …")
    nd = simulate_null_distribution(n_windows=n_windows)
    print_null_distribution(nd)

    # ── 3. Shrinkage Estimates ────────────────────────────────────────────────
    # Use in-sample aggregate as the "observed" metrics being shrunk.
    agg         = wf_cv.get("aggregate", {})
    obs_sharpe  = agg.get("is_sharpe_mean",  2.5)
    obs_wr      = agg.get("is_wr_mean",      0.569)
    obs_avg_pnl = agg.get("is_pnl_mean",     8.68)

    print("\n  [3/3] Shrinkage estimates …")
    sh = shrinkage_estimate(
        observed_sharpe  = obs_sharpe,
        observed_wr      = obs_wr,
        observed_avg_pnl = obs_avg_pnl,
        n_windows        = n_windows,
    )
    print_shrinkage(sh)

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\n  Generating selection bias chart …")
    plot_bias_charts(wf_cv, nd, sh, output_dir)

    return {
        "walk_forward_cv":   wf_cv,
        "null_distribution": nd,
        "shrinkage":         sh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import core_strategy as cs
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else config.CSV_PATH
    df_raw   = cs.load_data(csv_path)
    run(df_raw)
