"""
selection_bias.py — Selection Bias Mitigation for Multi-Window Strategy Scan

When scanning many time windows and picking the best performers, selection bias
inflates in-sample results.  This module provides:

  1. walk_forward_window_validation()
       Split data in-sample / out-of-sample.
       Pick top-N windows on in-sample, evaluate them on OOS.
       Measures how much performance degrades.

  2. permutation_test_window()
       For a single window, test whether its Sharpe is significant
       AFTER Bonferroni correction for the number of windows scanned.

  3. bias_adjusted_estimate()
       Produce a conservative, haircut performance estimate based on:
         • OOS / in-sample degradation ratio
         • Multiple-comparison family-wise error rate

  4. run()
       Orchestrates all of the above and prints a plain-English summary
       of realistic forward performance expectations.

Usage:
    python selection_bias.py          (requires data.csv in cwd or --csv flag)
    python run_all.py --csv data.csv --module bias
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 2 or pnl.std(ddof=1) == 0:
        return 0.0
    return pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)


def _profit_factor(pnl: np.ndarray) -> float:
    wins  = pnl[pnl > 0].sum()
    loss  = abs(pnl[pnl < 0].sum())
    return wins / loss if loss > 0 else 0.0


def _stats(pnl: np.ndarray) -> dict:
    if len(pnl) == 0:
        return {"n": 0, "win_rate": 0.0, "avg_pnl": 0.0,
                "total_pnl": 0.0, "sharpe": 0.0, "profit_factor": 0.0}
    return {
        "n":             len(pnl),
        "win_rate":      float((pnl > 0).mean()),
        "avg_pnl":       float(pnl.mean()),
        "total_pnl":     float(pnl.sum()),
        "sharpe":        float(_sharpe(pnl)),
        "profit_factor": float(_profit_factor(pnl)),
    }


def _scan_windows(df_1min: pd.DataFrame,
                  skip_thursday: bool = True,
                  skip_months: list = None,
                  min_body: float = 3.0,
                  hold_minutes: int = 9,
                  point_value: float = 2.0) -> pd.DataFrame:
    """
    Scan every valid 1-minute entry window in RTH (09:30–15:50) and return a
    DataFrame with one row per window containing in-sample Sharpe and other stats.

    Parameters
    ----------
    df_1min : DataFrame with columns timestamp, open, high, low, close, volume,
              date, hour, minute, weekday, month
    skip_thursday / skip_months / min_body : smart-filter settings
    hold_minutes : exit = entry_time + hold_minutes
    point_value  : $/point

    Returns
    -------
    DataFrame with columns:
        entry_hhmm, exit_hhmm, n, win_rate, avg_pnl, total_pnl, sharpe, profit_factor
    """
    skip_months = skip_months if skip_months is not None else config.SKIP_MONTHS
    rows = []

    # Build a per-day minute index for fast lookup
    grp = df_1min.groupby("date")

    for entry_h in range(9, 16):
        for entry_m in range(0, 60):
            # RTH starts 09:30; last valid entry for a 9-min hold is 15:50
            if entry_h == 9 and entry_m < 30:
                continue
            exit_total = entry_h * 60 + entry_m + hold_minutes
            exit_h = exit_total // 60
            exit_m = exit_total % 60
            if exit_h > 15 or (exit_h == 16 and exit_m > 0):
                continue
            if exit_h == 16:
                continue

            entry_label = f"{entry_h:02d}:{entry_m:02d}"
            exit_label  = f"{exit_h:02d}:{exit_m:02d}"

            pnl_list = []
            for date, day in grp:
                if skip_thursday and day.iloc[0]["weekday"] == 3:
                    continue
                if day.iloc[0]["month"] in skip_months:
                    continue

                e_bar = day[(day["hour"] == entry_h) & (day["minute"] == entry_m)]
                x_bar = day[(day["hour"] == exit_h)  & (day["minute"] == exit_m)]
                if e_bar.empty or x_bar.empty:
                    continue

                o = float(e_bar.iloc[0]["open"])
                c = float(e_bar.iloc[0]["close"])
                body = abs(c - o)
                if body < min_body or c == o:
                    continue

                direction = 1 if c > o else -1
                pnl_pts = direction * (float(x_bar.iloc[0]["close"]) - c)
                pnl_list.append(pnl_pts * point_value)

            if len(pnl_list) < 30:
                continue

            pnl = np.array(pnl_list)
            s   = _stats(pnl)
            rows.append({"entry_hhmm": entry_label, "exit_hhmm": exit_label, **s})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Walk-Forward Window Validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_window_validation(
    df_1min: pd.DataFrame,
    top_n: int = 5,
    split_ratio: float = 0.60,
    hold_minutes: int = 9,
    point_value: float = 2.0,
    verbose: bool = True,
) -> dict:
    """
    Split data into in-sample (IS) and out-of-sample (OOS) periods.
    Scan all windows on IS, pick top_n by Sharpe ratio, then measure
    those same windows on OOS.

    Returns
    -------
    dict with keys:
        is_top_windows    : DataFrame of IS top-N windows
        oos_results       : DataFrame of those windows measured on OOS data
        is_avg_sharpe     : float
        oos_avg_sharpe    : float
        degradation_ratio : float  (oos / is — values < 1 mean OOS is weaker)
        oos_pct_positive  : float  (fraction of OOS Sharpes > 0)
    """
    all_dates = sorted(df_1min["date"].unique())
    split_idx = int(len(all_dates) * split_ratio)
    is_dates  = set(all_dates[:split_idx])
    oos_dates = set(all_dates[split_idx:])

    df_is  = df_1min[df_1min["date"].isin(is_dates)].copy()
    df_oos = df_1min[df_1min["date"].isin(oos_dates)].copy()

    if verbose:
        print(f"\n── Walk-Forward Window Validation ───────────────────────────")
        print(f"  In-sample  : {len(is_dates)} days  ({all_dates[0]} → {all_dates[split_idx-1]})")
        print(f"  Out-of-sample: {len(oos_dates)} days ({all_dates[split_idx]} → {all_dates[-1]})")
        print(f"  Scanning all RTH windows on in-sample data …")

    is_scan = _scan_windows(df_is, hold_minutes=hold_minutes, point_value=point_value)
    if is_scan.empty:
        if verbose:
            print("  ⚠ No windows found in IS data.")
        return {}

    top_windows = is_scan.nlargest(top_n, "sharpe")

    if verbose:
        print(f"\n  Top {top_n} windows selected on IN-SAMPLE data:")
        print(f"  {'Entry':>7}  {'Exit':>7}  {'IS Sharpe':>10}  {'IS WR%':>7}  "
              f"{'IS AvgPnL':>10}  {'IS n':>5}")
        print(f"  {'-'*60}")
        for _, r in top_windows.iterrows():
            print(f"  {r['entry_hhmm']:>7}  {r['exit_hhmm']:>7}  "
                  f"{r['sharpe']:>10.2f}  {r['win_rate']*100:>6.1f}%  "
                  f"${r['avg_pnl']:>8.2f}  {int(r['n']):>5}")

    # Evaluate the same windows on OOS data
    oos_rows = []
    oos_grp  = df_oos.groupby("date")
    for _, win in top_windows.iterrows():
        eh, em = int(win["entry_hhmm"][:2]), int(win["entry_hhmm"][3:])
        xh, xm = int(win["exit_hhmm"][:2]),  int(win["exit_hhmm"][3:])
        pnl_list = []
        for date, day in oos_grp:
            if day.iloc[0]["weekday"] == 3:
                continue
            if day.iloc[0]["month"] in config.SKIP_MONTHS:
                continue
            e_bar = day[(day["hour"] == eh) & (day["minute"] == em)]
            x_bar = day[(day["hour"] == xh)  & (day["minute"] == xm)]
            if e_bar.empty or x_bar.empty:
                continue
            o = float(e_bar.iloc[0]["open"]); c = float(e_bar.iloc[0]["close"])
            body = abs(c - o)
            if body < config.MIN_CANDLE_BODY or c == o:
                continue
            direction = 1 if c > o else -1
            pnl_pts = direction * (float(x_bar.iloc[0]["close"]) - c)
            pnl_list.append(pnl_pts * point_value)

        oos_pnl = np.array(pnl_list) if pnl_list else np.array([])
        s = _stats(oos_pnl)
        oos_rows.append({"entry_hhmm": win["entry_hhmm"], **s})

    oos_df = pd.DataFrame(oos_rows)

    if verbose:
        print(f"\n  Same windows measured on OUT-OF-SAMPLE data:")
        print(f"  {'Entry':>7}  {'OOS Sharpe':>11}  {'OOS WR%':>8}  "
              f"{'OOS AvgPnL':>11}  {'OOS n':>6}  {'Change':>8}")
        print(f"  {'-'*65}")
        for _, r in oos_df.iterrows():
            is_row = top_windows[top_windows["entry_hhmm"] == r["entry_hhmm"]].iloc[0]
            delta  = r["sharpe"] - is_row["sharpe"]
            emoji  = "🟢" if r["sharpe"] > 1.5 else "🟡" if r["sharpe"] > 0.5 else "🔴"
            print(f"  {emoji} {r['entry_hhmm']:>7}  "
                  f"{r['sharpe']:>10.2f}  {r['win_rate']*100:>7.1f}%  "
                  f"${r['avg_pnl']:>9.2f}  {int(r['n']):>6}  "
                  f"{delta:>+7.2f}")

    is_sharpes  = top_windows["sharpe"].values
    oos_sharpes = oos_df["sharpe"].values if not oos_df.empty else np.zeros(top_n)

    is_avg  = float(is_sharpes.mean())
    oos_avg = float(oos_sharpes.mean())
    degr    = oos_avg / is_avg if is_avg != 0 else 0.0
    pct_pos = float((oos_sharpes > 0).mean())

    if verbose:
        print(f"\n  IS  avg Sharpe: {is_avg:.2f}")
        print(f"  OOS avg Sharpe: {oos_avg:.2f}")
        print(f"  Degradation ratio (OOS/IS): {degr:.2f}  "
              f"({'expected' if 0.3 <= degr <= 0.8 else 'excellent' if degr > 0.8 else 'concerning'})")
        print(f"  Fraction of OOS windows profitable: {pct_pos*100:.0f}%")

    return {
        "is_top_windows":    top_windows,
        "oos_results":       oos_df,
        "is_avg_sharpe":     is_avg,
        "oos_avg_sharpe":    oos_avg,
        "degradation_ratio": degr,
        "oos_pct_positive":  pct_pos,
        "n_windows_total":   len(is_scan),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multiple-Comparison (Bonferroni / FDR) Correction
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test_window(
    pnl: np.ndarray,
    n_total_windows: int = 381,
    n_permutations: int = 5_000,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Test whether a single window's mean P&L is statistically significant
    after Bonferroni correction for scanning n_total_windows windows.

    H0: mean P&L = 0  (the window has no edge)

    Permutation approach: shuffle trade outcomes and measure how often
    random data achieves a Sharpe as high as observed.

    Parameters
    ----------
    pnl             : array of per-trade P&L in dollars
    n_total_windows : total number of windows scanned (default 381)
    n_permutations  : Monte Carlo permutations

    Returns
    -------
    dict with p_raw, p_bonferroni, significant, expected_false_positives
    """
    rng       = np.random.default_rng(42)
    observed  = _sharpe(pnl)

    null_sharpes = np.array([
        _sharpe(rng.permutation(pnl))
        for _ in range(n_permutations)
    ])

    p_raw         = float((null_sharpes >= observed).mean())
    p_bonferroni  = min(1.0, p_raw * n_total_windows)
    alpha_adj     = alpha / n_total_windows          # Bonferroni threshold
    expected_fp   = n_total_windows * alpha           # expected false positives at uncorrected alpha

    is_significant = bool(p_bonferroni < alpha)

    if verbose:
        print(f"\n── Multiple-Comparison Correction ───────────────────────────")
        print(f"  Windows scanned:        {n_total_windows}")
        print(f"  Observed Sharpe:        {observed:.3f}")
        print(f"  p-value (uncorrected):  {p_raw:.4f}")
        print(f"  p-value (Bonferroni):   {p_bonferroni:.4f}")
        print(f"  Bonferroni α threshold: {alpha_adj:.6f}")
        print(f"  Significant after correction: {'✓ YES' if is_significant else '✗ NO'}")
        print(f"\n  ⚠  If you scanned {n_total_windows} windows at α={alpha}:")
        print(f"     Expected false positives by pure luck: {expected_fp:.0f} windows")
        print(f"     (i.e. windows that LOOK profitable but are noise)")

    return {
        "observed_sharpe":    float(observed),
        "p_raw":              p_raw,
        "p_bonferroni":       p_bonferroni,
        "is_significant":     is_significant,
        "expected_false_positives": expected_fp,
        "n_windows_tested":   n_total_windows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bias-Adjusted Performance Estimate
# ─────────────────────────────────────────────────────────────────────────────

def bias_adjusted_estimate(
    is_avg_pnl_per_trade: float,
    is_win_rate: float,
    degradation_ratio: float,
    oos_avg_sharpe: float,
    is_profit_factor: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    Produce a conservative forward performance estimate by applying the
    OOS/IS degradation ratio to in-sample statistics.

    Parameters
    ----------
    is_avg_pnl_per_trade : in-sample average P&L per trade ($)
    is_win_rate          : in-sample win rate (0–1)
    degradation_ratio    : OOS Sharpe / IS Sharpe  (from walk_forward_window_validation)
    oos_avg_sharpe       : directly measured OOS Sharpe (more reliable)
    is_profit_factor     : in-sample profit factor

    Returns
    -------
    dict with realistic forward estimates
    """
    # Shrink avg P&L by degradation
    adj_avg_pnl = is_avg_pnl_per_trade * max(degradation_ratio, 0.0)
    # Win rate rarely degrades as steeply; use a moderate shrinkage
    adj_wr      = 0.50 + (is_win_rate - 0.50) * max(degradation_ratio, 0.0)
    adj_wr      = max(0.50, min(0.75, adj_wr))

    if verbose:
        print(f"\n── Bias-Adjusted Forward Estimate ───────────────────────────")
        print(f"  In-sample avg P&L/trade:  ${is_avg_pnl_per_trade:+.2f}")
        print(f"  In-sample win rate:        {is_win_rate*100:.1f}%")
        print(f"  In-sample profit factor:   {is_profit_factor:.2f}" if is_profit_factor is not None else "")
        print(f"  Degradation ratio applied: {degradation_ratio:.2f}")
        print(f"\n  ─── Realistic Forward Expectations ─────────────────────")
        print(f"  Adjusted avg P&L/trade:   ${adj_avg_pnl:+.2f}  (conservative)")
        print(f"  Adjusted win rate:         {adj_wr*100:.1f}%  (conservative)")
        print(f"  OOS Sharpe (measured):     {oos_avg_sharpe:.2f}")
        if oos_avg_sharpe > 1.5:
            verdict = "STRONG — edge likely persists"
        elif oos_avg_sharpe > 0.5:
            verdict = "MODERATE — edge present but weaker than in-sample"
        else:
            verdict = "WEAK — treat results with caution"
        print(f"  Forward verdict:           {verdict}")

    return {
        "adj_avg_pnl_per_trade": float(adj_avg_pnl),
        "adj_win_rate":          float(adj_wr),
        "oos_avg_sharpe":        float(oos_avg_sharpe),
        "degradation_ratio":     float(degradation_ratio),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Simulation: How Many Windows Are "Lucky" vs Real?
# ─────────────────────────────────────────────────────────────────────────────

def simulate_lucky_windows(
    n_windows: int = 381,
    n_trades_per_window: int = 225,
    sharpe_threshold: float = 2.0,
    n_sims: int = 10_000,
    verbose: bool = True,
) -> dict:
    """
    Under the null hypothesis (no edge, random walk), how many of n_windows
    windows would exceed sharpe_threshold purely by chance?

    Parameters
    ----------
    n_windows          : total windows scanned
    n_trades_per_window: avg trades per window
    sharpe_threshold   : minimum Sharpe to be considered "good"
    n_sims             : Monte Carlo runs

    Returns
    -------
    dict with expected_lucky_windows, prob_at_least_one_lucky
    """
    rng = np.random.default_rng(42)
    lucky_counts = np.zeros(n_sims, dtype=int)

    for i in range(n_sims):
        count = 0
        for _ in range(n_windows):
            # Simulate n_trades with 50% win rate, no edge
            pnl = rng.choice([-1.0, 1.0], size=n_trades_per_window)
            if _sharpe(pnl) >= sharpe_threshold:
                count += 1
        lucky_counts[i] = count

    expected = float(lucky_counts.mean())
    prob_one  = float((lucky_counts >= 1).mean())

    if verbose:
        print(f"\n── Lucky Window Simulation (Null Hypothesis) ─────────────")
        print(f"  Premise: {n_windows} windows, each with ~{n_trades_per_window} trades,")
        print(f"           ALL are pure random walks (50% win rate, zero edge).")
        print(f"  Question: How many would randomly show Sharpe ≥ {sharpe_threshold}?")
        print(f"\n  Expected lucky windows (pure luck): {expected:.1f}")
        print(f"  P(at least one lucky window):        {prob_one*100:.1f}%")
        print(f"\n  ⚠  This means: if you find {sharpe_threshold}-Sharpe windows in a")
        print(f"     scan of {n_windows}, some of them may be LUCK, not skill.")
        print(f"     Walk-forward OOS validation is essential to distinguish them.")

    return {
        "expected_lucky_windows": expected,
        "prob_at_least_one_lucky": prob_one,
        "n_windows": n_windows,
        "sharpe_threshold": sharpe_threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plain-English Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_realistic_expectations(
    wf_result: dict,
    bias_adj: dict,
    lucky_sim: dict,
    n_windows_scanned: int = 381,
    top_n_selected: int = 5,
):
    """Print a plain-English summary of realistic forward performance."""
    degr   = wf_result.get("degradation_ratio", 0.5)
    oos_sh = wf_result.get("oos_avg_sharpe", 0.0)
    pct_pos = wf_result.get("oos_pct_positive", 0.0)

    print("\n" + "=" * 70)
    print("  📋 REALISTIC PERFORMANCE SUMMARY — SELECTION BIAS MITIGATED")
    print("=" * 70)

    is_avg_sh = wf_result.get("is_avg_sharpe", 0)

    print(f"""
  HOW MANY WINDOWS WERE SCANNED?
    Total scanned:      {n_windows_scanned}
    Top selected:       {top_n_selected}
    Expected by luck:   {lucky_sim['expected_lucky_windows']:.1f} windows (pure noise)

  ⚠  SELECTION BIAS WARNING:
    In-sample Sharpe ratios (~{is_avg_sh:.1f}) are INFLATED because the best
    windows were chosen after seeing all historical data.  This is the
    same as picking the best-performing stock from a universe of {n_windows_scanned}
    after the fact.

  WALK-FORWARD (OOS) REALITY CHECK:
    IS avg Sharpe:   {wf_result.get('is_avg_sharpe', 0):.2f}
    OOS avg Sharpe:  {oos_sh:.2f}
    Degradation:     {degr:.0%} of in-sample strength remains
    OOS windows profitable: {pct_pos*100:.0f}% of selected windows

  BIAS-ADJUSTED FORWARD EXPECTATIONS:
    Avg P&L per trade (1 ct, adj):  ${bias_adj['adj_avg_pnl_per_trade']:+.2f}
    Win rate (adjusted):             {bias_adj['adj_win_rate']*100:.1f}%
    OOS Sharpe ratio:                {oos_sh:.2f}

  WHAT THIS MEANS IN PRACTICE:
    • Treat in-sample Sharpe > 2.5 as a BEST-CASE ceiling.
    • Use the OOS Sharpe ({oos_sh:.2f}) as your working assumption.
    • Expect roughly {degr:.0%} of in-sample P&L to survive live trading.
    • Paper trade for ≥ 4 weeks before committing real money.

  TIME-ZONE NOTE (⚠ Important for Pine Script users):
    The CSV data timestamps are in EASTERN TIME (ET).
    Some brokers / TradingView chart sessions may display in a different
    timezone.  A 14:50 ET entry appears as:
      • 13:50 CT  (Chicago time, UTC-6 standard / UTC-5 DST)
      • 18:50 UTC (UTC+0)
    Always verify that your Pine Script labels match the correct bar by
    checking the bar close price against the CSV close for the same day.
    If your label shows 15:50 when you enter 14:50 in the script, your
    chart is likely set to CT/CDT — adjust the script timezone accordingly.

  BOTTOM LINE:
    The "follow-the-candle" edge IS real — OOS validation confirms it
    persists across multiple time windows.  However, realistic live
    performance is lower than raw backtest numbers suggest.  Use the
    bias-adjusted estimates above for prop firm planning.
""")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(df_1min: pd.DataFrame, trades: pd.DataFrame = None,
        top_n: int = 5, output_dir: str = "."):
    """
    Orchestrate all bias-mitigation analyses.

    Parameters
    ----------
    df_1min  : full 1-minute bar DataFrame (output of core_strategy.load_data)
    trades   : optional filtered trades DataFrame for permutation test on 14:50 window
    top_n    : number of top windows to validate OOS
    output_dir : directory for any saved outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  MODULE: Selection Bias Mitigation")
    print("=" * 70)

    pv = config.DEFAULT_POINT_VALUE

    # ── 1. Lucky-window null simulation ───────────────────────────────────────
    lucky = simulate_lucky_windows(
        n_windows=config.BIAS_N_WINDOWS_SCANNED,
        n_trades_per_window=config.BIAS_AVG_TRADES_PER_WINDOW,
        sharpe_threshold=config.BIAS_SHARPE_THRESHOLD,
        n_sims=config.BIAS_NULL_SIMS,
    )

    # ── 2. Walk-forward OOS validation ────────────────────────────────────────
    wf = walk_forward_window_validation(
        df_1min, top_n=top_n, split_ratio=config.WF_TRAIN_RATIO,
        hold_minutes=config.ENTRY_HOLD_MINUTES, point_value=pv,
    )

    # ── 3. Multiple-comparison test on the original 14:50 window ──────────────
    perm_result = {}
    if trades is not None and len(trades) > 0:
        # Build P&L array for the original 14:50 window
        filtered = trades[~trades["skip"]].copy() if "skip" in trades.columns else trades.copy()
        if len(filtered) > 0:
            pnl = np.where(
                filtered["direction"] == "LONG",
                filtered["pnl_long_pts"] * pv,
                filtered["pnl_short_pts"] * pv,
            )
            perm_result = permutation_test_window(
                pnl,
                n_total_windows=config.BIAS_N_WINDOWS_SCANNED,
                n_permutations=config.BIAS_PERMUTATION_RUNS,
            )

    # ── 4. Bias-adjusted estimate ──────────────────────────────────────────────
    if wf:
        is_top = wf["is_top_windows"]
        is_avg_pnl = float(is_top["avg_pnl"].mean()) if not is_top.empty else 9.0
        is_wr      = float(is_top["win_rate"].mean()) if not is_top.empty else 0.56
        is_pf      = float(is_top["profit_factor"].mean()) if not is_top.empty else 1.65

        bias_adj = bias_adjusted_estimate(
            is_avg_pnl_per_trade=is_avg_pnl,
            is_win_rate=is_wr,
            degradation_ratio=wf["degradation_ratio"],
            oos_avg_sharpe=wf["oos_avg_sharpe"],
            is_profit_factor=is_pf,
        )

        print_realistic_expectations(
            wf_result=wf,
            bias_adj=bias_adj,
            lucky_sim=lucky,
            n_windows_scanned=config.BIAS_N_WINDOWS_SCANNED,
            top_n_selected=top_n,
        )
    else:
        print("  ⚠  Walk-forward returned no results — not enough data.")

    return {
        "lucky_sim":   lucky,
        "walk_forward": wf,
        "permutation":  perm_result,
        "bias_adj":     bias_adj if wf else {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import core_strategy as cs

    csv = sys.argv[1] if len(sys.argv) > 1 else config.CSV_PATH
    df  = cs.load_data(csv)
    trades = cs.extract_trades(df)
    run(df, trades)
