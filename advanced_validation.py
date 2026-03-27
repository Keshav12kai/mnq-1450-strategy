"""
advanced_validation.py — Statistical Validation Suite for MNQ 14:50 Strategy

Includes:
  - Monte Carlo Simulation (10,000 paths)
  - Walk-Forward Analysis (5 windows, 60/40 split)
  - Pattern Detection (DoW, monthly, volatility regime, candle-size filter)
  - Statistical Edge Validation (t-test, bootstrap CI, runs test)
  - Charts
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _equity_curve(pnl_series: np.ndarray, initial: float) -> np.ndarray:
    return np.concatenate([[initial], initial + np.cumsum(pnl_series)])


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = equity - peak
    return dd.min()


def _sharpe(pnl: np.ndarray, initial: float) -> float:
    r = pnl / initial
    if r.std(ddof=1) == 0:
        return 0.0
    return r.mean() / r.std(ddof=1) * np.sqrt(252)


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def monte_carlo(pnl_series: np.ndarray, initial_capital: float,
                n_runs: int = None, daily_loss_limit: float = None) -> dict:
    """
    Shuffle trade order n_runs times.
    Returns: final_cap distribution, max_dd distribution, sharpe distribution,
             probability of profit, probability DD stays under daily_loss_limit.
    """
    n_runs = n_runs or config.MC_RUNS
    pct    = config.MC_PERCENTILES
    n      = len(pnl_series)

    rng = np.random.default_rng(42)
    final_caps = np.empty(n_runs)
    max_dds    = np.empty(n_runs)
    sharpes    = np.empty(n_runs)

    for i in range(n_runs):
        shuffled = rng.permutation(pnl_series)
        eq       = _equity_curve(shuffled, initial_capital)
        final_caps[i] = eq[-1]
        max_dds[i]    = _max_drawdown(eq)
        sharpes[i]    = _sharpe(shuffled, initial_capital)

    result = {
        "final_cap_pct":  {p: float(np.percentile(final_caps, p)) for p in pct},
        "max_dd_pct":     {p: float(np.percentile(max_dds,    p)) for p in pct},
        "sharpe_pct":     {p: float(np.percentile(sharpes,    p)) for p in pct},
        "prob_profit":    float((final_caps > initial_capital).mean()),
        "final_caps":     final_caps,
        "max_dds":        max_dds,
        "sharpes":        sharpes,
    }
    if daily_loss_limit is not None:
        result["prob_dd_under_limit"] = float((np.abs(max_dds) < daily_loss_limit).mean())

    return result


def print_mc_results(mc: dict, initial_capital: float):
    print("\n── Monte Carlo Simulation ──────────────────────────────────")
    pct = config.MC_PERCENTILES
    print(f"  {'Metric':<30}", "  ".join(f"P{p:2d}" for p in pct))
    rows = [
        ("Final Capital ($)", "final_cap_pct"),
        ("Max Drawdown ($)",  "max_dd_pct"),
        ("Sharpe Ratio",      "sharpe_pct"),
    ]
    for label, key in rows:
        vals = mc[key]
        print(f"  {label:<30}", "  ".join(f"{vals[p]:>8,.1f}" for p in pct))
    print(f"  Probability of Profit:    {mc['prob_profit']*100:.1f}%")
    if "prob_dd_under_limit" in mc:
        print(f"  Prob DD Under Limit:      {mc['prob_dd_under_limit']*100:.1f}%")


# ── Walk-Forward ──────────────────────────────────────────────────────────────

def walk_forward(pnl_series: np.ndarray, initial_capital: float,
                 n_windows: int = None, train_ratio: float = None) -> list:
    """
    Returns list of dicts with train/test stats per window.
    """
    n_windows   = n_windows  or config.WF_WINDOWS
    train_ratio = train_ratio or config.WF_TRAIN_RATIO
    n = len(pnl_series)
    window_size = n // n_windows
    results = []

    for w in range(n_windows):
        start = w * window_size
        end   = start + window_size if w < n_windows - 1 else n
        chunk = pnl_series[start:end]
        split = int(len(chunk) * train_ratio)
        train = chunk[:split]
        test  = chunk[split:]
        if len(train) < 5 or len(test) < 2:
            continue

        def _stats(arr):
            wins = arr[arr > 0]
            return {
                "n": len(arr),
                "win_rate": (arr > 0).mean(),
                "avg_pnl":  arr.mean(),
                "total":    arr.sum(),
                "sharpe":   _sharpe(arr, initial_capital),
            }

        results.append({
            "window":     w + 1,
            "train_n":    len(train),
            "test_n":     len(test),
            "train":      _stats(train),
            "test":       _stats(test),
        })

    return results


def print_wf_results(wf: list):
    print("\n── Walk-Forward Analysis ───────────────────────────────────")
    print(f"  {'Win':>4}  {'Train WR':>10}  {'Train Sharpe':>13}  {'Test WR':>9}  {'Test Sharpe':>12}")
    for r in wf:
        print(
            f"  {r['window']:>4}  "
            f"{r['train']['win_rate']*100:>9.1f}%  "
            f"{r['train']['sharpe']:>13.2f}  "
            f"{r['test']['win_rate']*100:>8.1f}%  "
            f"{r['test']['sharpe']:>12.2f}"
        )
    # Out-of-sample edge persistence
    profitable_oos = sum(1 for r in wf if r["test"]["avg_pnl"] > 0)
    print(f"  OOS profitable windows: {profitable_oos}/{len(wf)}")


# ── Pattern Detection ─────────────────────────────────────────────────────────

def day_of_week_analysis(trades_df: pd.DataFrame, pnl_col: str = "pnl") -> pd.DataFrame:
    days = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
    g = trades_df.groupby("weekday")[pnl_col]
    df = pd.DataFrame({
        "day":       [days.get(d, d) for d in g.groups],
        "avg_pnl":   g.mean().values,
        "win_rate":  trades_df.groupby("weekday").apply(
                        lambda x: (x[pnl_col] > 0).mean()).values,
        "total_pnl": g.sum().values,
        "count":     g.count().values,
    })
    return df.reset_index(drop=True)


def monthly_analysis(trades_df: pd.DataFrame, pnl_col: str = "pnl") -> pd.DataFrame:
    months = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    g = trades_df.groupby("month")[pnl_col]
    df = pd.DataFrame({
        "month":     [months.get(m, m) for m in g.groups],
        "avg_pnl":   g.mean().values,
        "win_rate":  trades_df.groupby("month").apply(
                        lambda x: (x[pnl_col] > 0).mean()).values,
        "total_pnl": g.sum().values,
        "count":     g.count().values,
    })
    return df.reset_index(drop=True)


def volatility_regime_analysis(trades_df: pd.DataFrame, pnl_col: str = "pnl") -> dict:
    """Split into low / high vol by median candle_range (high - low of 14:50 candle)."""
    if "candle_body" not in trades_df.columns:
        return {}
    med = trades_df["candle_body"].median()
    low  = trades_df[trades_df["candle_body"] <  med]
    high = trades_df[trades_df["candle_body"] >= med]

    def _s(subset):
        return {
            "n": len(subset),
            "avg_pnl": subset[pnl_col].mean() if len(subset) else 0.0,
            "win_rate": (subset[pnl_col] > 0).mean() if len(subset) else 0.0,
            "total":    subset[pnl_col].sum() if len(subset) else 0.0,
        }

    # Quartile breakdown
    q = np.percentile(trades_df["candle_body"], [25, 50, 75])
    quartiles = []
    edges = [0] + list(q) + [1e9]
    labels = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    for i in range(4):
        sub = trades_df[(trades_df["candle_body"] >= edges[i]) &
                        (trades_df["candle_body"] <  edges[i+1])]
        quartiles.append({"label": labels[i], **_s(sub)})

    return {"low": _s(low), "high": _s(high), "quartiles": quartiles}


def candle_size_filter_analysis(trades_df: pd.DataFrame, pnl_col: str = "pnl") -> list:
    """Test different candle-body thresholds (p10 … p90)."""
    results = []
    for p in [10, 25, 50, 75, 90]:
        thresh = np.percentile(trades_df["candle_body"], p)
        sub    = trades_df[trades_df["candle_body"] >= thresh]
        results.append({
            "percentile": p,
            "threshold":  thresh,
            "n":          len(sub),
            "avg_pnl":    sub[pnl_col].mean() if len(sub) else 0.0,
            "win_rate":   (sub[pnl_col] > 0).mean() if len(sub) else 0.0,
            "total":      sub[pnl_col].sum() if len(sub) else 0.0,
        })
    return results


# ── Statistical Edge Validation ───────────────────────────────────────────────

def statistical_edge(pnl_series: np.ndarray) -> dict:
    n   = len(pnl_series)
    mu  = pnl_series.mean()
    se  = pnl_series.std(ddof=1) / np.sqrt(n)

    # One-sample t-test H0: mean = 0
    t_stat, t_pval = scipy_stats.ttest_1samp(pnl_series, 0.0)

    # Bootstrap 95% CI (10,000 samples)
    rng = np.random.default_rng(42)
    boot_means = np.array([rng.choice(pnl_series, n, replace=True).mean()
                           for _ in range(10_000)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Skewness & kurtosis
    skew = scipy_stats.skew(pnl_series)
    kurt = scipy_stats.kurtosis(pnl_series)

    # Runs test for serial independence
    # Convert to binary (win=1 / loss=0)
    binary = (pnl_series > 0).astype(int)
    n1 = binary.sum()
    n0 = n - n1
    runs = 1 + np.sum(binary[1:] != binary[:-1])
    if n1 > 0 and n0 > 0:
        expected_runs = (2 * n1 * n0) / n + 1
        var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
        z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0.0
        p_runs = 2 * (1 - scipy_stats.norm.cdf(abs(z_runs)))
    else:
        z_runs = 0.0
        p_runs = 1.0

    return {
        "n":          n,
        "mean_pnl":   mu,
        "std_pnl":    pnl_series.std(ddof=1),
        "t_stat":     t_stat,
        "t_pval":     t_pval,
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "skewness":   skew,
        "kurtosis":   kurt,
        "runs":       int(runs),
        "z_runs":     z_runs,
        "p_runs":     p_runs,
        "is_significant": bool(t_pval < 0.05),
        "is_independent": bool(p_runs > 0.05),
    }


def print_edge_results(edge: dict):
    print("\n── Statistical Edge Validation ─────────────────────────────")
    print(f"  Trades:            {edge['n']}")
    print(f"  Mean P&L:          ${edge['mean_pnl']:.2f}")
    print(f"  Std P&L:           ${edge['std_pnl']:.2f}")
    print(f"  T-stat:            {edge['t_stat']:.3f}   p={edge['t_pval']:.4f}  {'✓ SIGNIFICANT' if edge['is_significant'] else '✗ not significant'}")
    print(f"  Bootstrap 95% CI:  [${edge['ci_lo']:.2f}, ${edge['ci_hi']:.2f}]")
    print(f"  Skewness:          {edge['skewness']:.3f}")
    print(f"  Kurtosis:          {edge['kurtosis']:.3f}")
    print(f"  Runs test:         z={edge['z_runs']:.3f}   p={edge['p_runs']:.4f}  {'✓ independent' if edge['is_independent'] else '✗ autocorrelated'}")


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_mc_charts(mc: dict, initial_capital: float, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Monte Carlo Simulation", fontsize=13, fontweight="bold")

    # Final capital distribution
    ax = axes[0]
    ax.hist(mc["final_caps"], bins=60, color="#42a5f5", alpha=0.8, edgecolor="none")
    ax.axvline(initial_capital, color="red", linestyle="--", label="Initial Capital")
    for p, color in [(5, "#ef5350"), (50, "white"), (95, "#26a69a")]:
        ax.axvline(mc["final_cap_pct"][p], color=color, linestyle=":", alpha=0.9,
                   label=f"P{p}: ${mc['final_cap_pct'][p]:,.0f}")
    ax.set_title("Final Capital Distribution")
    ax.set_xlabel("Final Capital ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Max drawdown distribution
    ax = axes[1]
    ax.hist(mc["max_dds"], bins=60, color="#ef5350", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="white", linestyle="--")
    for p, color in [(5, "#26a69a"), (50, "white"), (95, "#ef5350")]:
        ax.axvline(mc["max_dd_pct"][p], color=color, linestyle=":", alpha=0.9,
                   label=f"P{p}: ${mc['max_dd_pct'][p]:,.0f}")
    ax.set_title("Max Drawdown Distribution")
    ax.set_xlabel("Max Drawdown ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "mc_charts.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pattern_charts(trades_df: pd.DataFrame, pnl_col: str = "pnl",
                        output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Pattern Analysis", fontsize=13, fontweight="bold")

    # Day of week
    ax = axes[0, 0]
    dow = day_of_week_analysis(trades_df, pnl_col)
    clrs = ["#26a69a" if v >= 0 else "#ef5350" for v in dow["avg_pnl"]]
    ax.bar(dow["day"], dow["avg_pnl"], color=clrs)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Avg P&L by Day of Week")
    ax.set_ylabel("Avg P&L ($)")
    ax.grid(alpha=0.3, axis="y")

    # Monthly
    ax = axes[0, 1]
    mon = monthly_analysis(trades_df, pnl_col)
    clrs = ["#26a69a" if v >= 0 else "#ef5350" for v in mon["total_pnl"]]
    ax.bar(mon["month"], mon["total_pnl"], color=clrs)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Total P&L by Month")
    ax.set_ylabel("Total P&L ($)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")

    # Rolling win rate (30-trade window)
    ax = axes[1, 0]
    if len(trades_df) >= 30:
        roll_wr = (trades_df[pnl_col] > 0).rolling(30).mean() * 100
        ax.plot(roll_wr.values, color="#42a5f5", lw=1.2)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Rolling Win Rate (30-trade window)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Win Rate (%)")
        ax.grid(alpha=0.3)

    # P&L vs candle body size scatter
    ax = axes[1, 1]
    if "candle_body" in trades_df.columns:
        ax.scatter(trades_df["candle_body"], trades_df[pnl_col],
                   alpha=0.3, s=8, color="#42a5f5")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("P&L vs Candle Body Size")
        ax.set_xlabel("Candle Body (pts)")
        ax.set_ylabel("P&L ($)")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "pattern_analysis.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(trades_df: pd.DataFrame, pnl_col: str = "pnl",
        initial_capital: float = None, output_dir: str = "."):
    cap = initial_capital or config.INITIAL_CAPITAL
    os.makedirs(output_dir, exist_ok=True)

    if trades_df.empty:
        print("  No trades to validate.")
        return {}

    pnl = trades_df[pnl_col].values

    print("\n=== Advanced Validation ===")

    # Monte Carlo
    mc = monte_carlo(pnl, cap)
    print_mc_results(mc, cap)

    # Walk-Forward
    wf = walk_forward(pnl, cap)
    print_wf_results(wf)

    # Pattern Detection
    print("\n── Day-of-Week Analysis ─────────────────────────────────────")
    if "weekday" in trades_df.columns:
        dow = day_of_week_analysis(trades_df, pnl_col)
        print(dow.to_string(index=False))

    print("\n── Monthly Analysis ─────────────────────────────────────────")
    if "month" in trades_df.columns:
        mon = monthly_analysis(trades_df, pnl_col)
        print(mon.to_string(index=False))

    print("\n── Volatility Regime Analysis ───────────────────────────────")
    if "candle_body" in trades_df.columns:
        vr = volatility_regime_analysis(trades_df, pnl_col)
        print(f"  Low vol  (n={vr['low']['n']}):  avg P&L ${vr['low']['avg_pnl']:.2f}  WR {vr['low']['win_rate']*100:.1f}%")
        print(f"  High vol (n={vr['high']['n']}): avg P&L ${vr['high']['avg_pnl']:.2f}  WR {vr['high']['win_rate']*100:.1f}%")
        print("  Quartile breakdown:")
        for q in vr.get("quartiles", []):
            print(f"    {q['label']:<12}: n={q['n']}  avg ${q['avg_pnl']:.2f}  WR {q['win_rate']*100:.1f}%")

    print("\n── Candle-Size Filter Analysis ──────────────────────────────")
    if "candle_body" in trades_df.columns:
        csf = candle_size_filter_analysis(trades_df, pnl_col)
        for r in csf:
            print(f"  Body >= P{r['percentile']:2d} ({r['threshold']:.1f} pts): "
                  f"n={r['n']}  avg ${r['avg_pnl']:.2f}  WR {r['win_rate']*100:.1f}%  total ${r['total']:,.0f}")

    # Statistical Edge
    edge = statistical_edge(pnl)
    print_edge_results(edge)

    # Charts
    print("\nGenerating validation charts …")
    plot_mc_charts(mc, cap, output_dir)
    plot_pattern_charts(trades_df, pnl_col, output_dir)

    return {
        "monte_carlo": mc,
        "walk_forward": wf,
        "edge": edge,
    }


if __name__ == "__main__":
    import core_strategy as cs
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else config.CSV_PATH
    df  = cs.load_data(csv)
    trades_all = cs.extract_trades(df)
    filtered   = trades_all[~trades_all["skip"]].copy()
    filtered["pnl"] = np.where(
        filtered["direction"] == "LONG",
        filtered["pnl_long_pts"] * config.DEFAULT_POINT_VALUE,
        filtered["pnl_short_pts"] * config.DEFAULT_POINT_VALUE,
    )
    run(filtered)
