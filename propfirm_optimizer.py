"""
propfirm_optimizer.py — Prop Firm Challenge Optimizer

Sweeps contract sizes (1-50) for each prop firm, finds optimal contract count,
calculates pass rates, expected cost, expected value, and rankings.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

warnings.filterwarnings("ignore")


# ── Core Simulation ───────────────────────────────────────────────────────────

def simulate_challenge(pnl_pts: np.ndarray, contracts: int,
                        prop_firm: dict, n_runs: int = 1000,
                        seed: int = 42) -> dict:
    """
    Monte Carlo simulation of a prop firm challenge with fixed contract size.
    Applies smart-filtered trade P&L (already filtered).
    """
    pv        = config.DEFAULT_POINT_VALUE
    target    = prop_firm["profit_target"]
    max_dd    = prop_firm["max_drawdown"]
    daily_lim = prop_firm["daily_loss_limit"]
    days      = prop_firm["challenge_days"]

    rng = np.random.default_rng(seed)
    passed = 0
    fail_dd = fail_daily = fail_target = 0
    days_to_pass = []
    daily_pnls_all = []

    for _ in range(n_runs):
        equity     = 0.0
        peak       = 0.0
        failed     = False
        day        = 0
        daily_pnls = []
        idxs = rng.integers(0, len(pnl_pts), size=days)

        for d in range(days):
            pts        = pnl_pts[idxs[d]]
            c          = contracts

            # DD reduction
            if peak > 0:
                dd_pct = (equity - peak) / peak
                for thresh, cut in config.DD_LEVELS:
                    if abs(dd_pct) >= thresh:
                        c = max(1, int(c * (1 - cut)))

            daily_pnl   = pts * pv * c
            equity     += daily_pnl
            peak        = max(peak, equity)
            day        += 1
            daily_pnls.append(daily_pnl)

            if equity < -max_dd:
                failed = True; fail_dd += 1; break
            if daily_pnl < -daily_lim:
                failed = True; fail_daily += 1; break

        daily_pnls_all.extend(daily_pnls)
        if not failed:
            if equity >= target:
                passed += 1
                days_to_pass.append(day)
            else:
                fail_target += 1

    n = max(n_runs, 1)
    return {
        "contracts":      contracts,
        "pass_rate":      passed / n,
        "fail_dd":        fail_dd / n,
        "fail_daily":     fail_daily / n,
        "fail_target":    fail_target / n,
        "avg_days":       float(np.mean(days_to_pass)) if days_to_pass else float(days),
        "fastest":        int(min(days_to_pass)) if days_to_pass else days,
        "daily_pnls":     np.array(daily_pnls_all),
    }


# ── Contract Sweep ────────────────────────────────────────────────────────────

def contract_sweep(pnl_pts: np.ndarray, prop_firm: dict,
                   max_c: int = 50, n_runs: int = 1000) -> pd.DataFrame:
    """Sweep contracts 1-max_c, return DataFrame of results."""
    rows = []
    for c in range(1, max_c + 1):
        r = simulate_challenge(pnl_pts, c, prop_firm, n_runs)
        rows.append({
            "contracts":     c,
            "pass_rate":     r["pass_rate"],
            "fail_dd":       r["fail_dd"],
            "fail_daily":    r["fail_daily"],
            "fail_target":   r["fail_target"],
            "avg_days":      r["avg_days"],
            "fastest":       r["fastest"],
        })
    return pd.DataFrame(rows)


def find_optimal_contracts(sweep_df: pd.DataFrame) -> dict:
    """Return the contract row with the highest pass rate."""
    best = sweep_df.loc[sweep_df["pass_rate"].idxmax()]
    return best.to_dict()


# ── Multi-Firm Sweep ──────────────────────────────────────────────────────────

def multi_firm_sweep(pnl_pts: np.ndarray,
                     prop_firms: list = None,
                     n_runs: int = 500) -> pd.DataFrame:
    """
    For each prop firm, find optimal contracts and compute metrics.
    Returns combined summary DataFrame.
    """
    firms = prop_firms or config.PROP_FIRMS
    rows  = []

    for firm in firms:
        sweep = contract_sweep(pnl_pts, firm, max_c=30, n_runs=n_runs)
        opt   = find_optimal_contracts(sweep)
        fee   = firm.get("fee", 200)

        # Expected cost to get funded (geometric series: fee / pass_rate)
        pr = opt["pass_rate"]
        expected_attempts = 1 / pr if pr > 0 else float("inf")
        expected_cost     = fee * expected_attempts

        # Simplified 6-month EV:
        # assume monthly profit = account_size * 0.05 (5% payout),
        # funded probability in 6 months = 1 - (1-pass_rate)^6_attempts
        monthly_profit   = firm["account_size"] * 0.05 * 0.80  # 80% payout
        funded_prob_6mo  = 1 - (1 - pr) ** 6 if pr > 0 else 0
        ev_6mo           = funded_prob_6mo * monthly_profit * 6 - expected_cost

        rows.append({
            "firm":            firm["name"],
            "account_size":    firm["account_size"],
            "profit_target":   firm["profit_target"],
            "max_drawdown":    firm["max_drawdown"],
            "daily_limit":     firm["daily_loss_limit"],
            "fee":             fee,
            "opt_contracts":   int(opt["contracts"]),
            "pass_rate":       opt["pass_rate"],
            "fail_dd":         opt["fail_dd"],
            "fail_daily":      opt["fail_daily"],
            "avg_days":        opt["avg_days"],
            "expected_cost":   expected_cost,
            "ev_6mo":          ev_6mo,
        })

    return pd.DataFrame(rows)


# ── Rankings ──────────────────────────────────────────────────────────────────

def print_rankings(summary_df: pd.DataFrame):
    print("\n=== Multi-Firm Rankings ===")

    print("\n── By Pass Rate ─────────────────────────────────────────────")
    top = summary_df.sort_values("pass_rate", ascending=False).head(10)
    print(top[["firm", "opt_contracts", "pass_rate", "fail_dd", "fail_daily", "avg_days"]].to_string(index=False))

    print("\n── By Expected Cost (cheapest to pass) ──────────────────────")
    bot = summary_df.sort_values("expected_cost").head(10)
    print(bot[["firm", "fee", "opt_contracts", "pass_rate", "expected_cost"]].to_string(index=False))

    print("\n── By 6-Month EV ────────────────────────────────────────────")
    ev = summary_df.sort_values("ev_6mo", ascending=False).head(10)
    print(ev[["firm", "fee", "opt_contracts", "pass_rate", "ev_6mo"]].to_string(index=False))


def print_top3_detail(summary_df: pd.DataFrame, pnl_pts: np.ndarray):
    top3 = summary_df.sort_values("pass_rate", ascending=False).head(3)
    print("\n=== Top-3 Detailed Analysis ===")
    for _, row in top3.iterrows():
        firm = next((f for f in config.PROP_FIRMS if f["name"] == row["firm"]), None)
        if firm is None:
            continue
        pr = row["pass_rate"]
        print(f"\n  {row['firm']}")
        print(f"    Optimal contracts: {int(row['opt_contracts'])}")
        print(f"    Pass rate:         {pr*100:.1f}%")
        print(f"    Avg days to pass:  {row['avg_days']:.1f}")
        print(f"    Fail DD:           {row['fail_dd']*100:.1f}%")
        print(f"    Fail daily limit:  {row['fail_daily']*100:.1f}%")
        print(f"    Challenge fee:     ${firm['fee']:,.0f}")
        print(f"    Expected cost:     ${row['expected_cost']:,.0f}")
        # Multi-attempt probabilities
        print("    Pass by attempt:")
        cumul = 0.0
        for att in [1, 2, 3, 5, 10]:
            p_by  = 1 - (1 - pr) ** att
            print(f"      Attempt {att:2d}: {p_by*100:.1f}%")


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_sweep_chart(sweep_df: pd.DataFrame, firm_name: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Contract Sweep — {firm_name}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(sweep_df["contracts"], sweep_df["pass_rate"] * 100, "o-", color="#42a5f5", lw=2)
    best_c = sweep_df.loc[sweep_df["pass_rate"].idxmax(), "contracts"]
    best_r = sweep_df["pass_rate"].max() * 100
    ax.axvline(best_c, color="#ffa726", linestyle="--", label=f"Optimal: {best_c} contracts")
    ax.set_title("Pass Rate vs Contracts")
    ax.set_xlabel("Contracts")
    ax.set_ylabel("Pass Rate (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.stackplot(
        sweep_df["contracts"],
        sweep_df["fail_dd"] * 100,
        sweep_df["fail_daily"] * 100,
        sweep_df["fail_target"] * 100,
        labels=["Fail DD", "Fail Daily Limit", "Fail Target"],
        colors=["#ef5350", "#ffa726", "#78909c"],
        alpha=0.8
    )
    ax.set_title("Failure Breakdown vs Contracts")
    ax.set_xlabel("Contracts")
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = firm_name.replace(" ", "_").replace("$", "").replace("/", "_")
    path = os.path.join(output_dir, f"sweep_{safe_name}.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_multi_firm_chart(summary_df: pd.DataFrame, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    top15 = summary_df.sort_values("pass_rate", ascending=False).head(15)
    colors = ["#26a69a" if v >= 0.5 else "#ef5350" for v in top15["pass_rate"]]
    bars = ax.barh(top15["firm"], top15["pass_rate"] * 100, color=colors)
    for bar, v in zip(bars, top15["pass_rate"] * 100):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8)
    ax.axvline(59.1, color="#ffa726", linestyle="--", alpha=0.7, label="Fixed 9-contract baseline (59.1%)")
    ax.set_title("Prop Firm Pass Rates (Optimal Contracts)")
    ax.set_xlabel("Pass Rate (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(output_dir, "multi_firm_pass_rates.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(trades: pd.DataFrame, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    filtered = trades[~trades["skip"]].copy() if "skip" in trades.columns else trades.copy()

    pnl_pts = np.where(
        filtered["direction"] == "LONG",
        filtered["pnl_long_pts"],
        filtered["pnl_short_pts"],
    ).astype(float)

    if len(pnl_pts) < 10:
        print("  Not enough trades for prop firm optimization.")
        return {}

    print("\n=== Prop Firm Optimizer ===")

    # Detailed sweep for first prop firm
    firm0 = config.PROP_FIRMS[0]
    print(f"\nRunning contract sweep for {firm0['name']} …")
    sweep_df = contract_sweep(pnl_pts, firm0, max_c=30, n_runs=500)
    opt = find_optimal_contracts(sweep_df)
    print(f"  Optimal contracts: {int(opt['contracts'])}  pass_rate: {opt['pass_rate']*100:.1f}%")

    plot_sweep_chart(sweep_df, firm0["name"], output_dir)

    # Multi-firm sweep
    print("\nRunning multi-firm sweep …")
    summary = multi_firm_sweep(pnl_pts, n_runs=500)

    print_rankings(summary)
    print_top3_detail(summary, pnl_pts)

    plot_multi_firm_chart(summary, output_dir)

    # Save summary CSV
    path = os.path.join(output_dir, "propfirm_summary.csv")
    summary.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    return {"sweep": sweep_df, "summary": summary}


if __name__ == "__main__":
    import core_strategy as cs
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else config.CSV_PATH
    df     = cs.load_data(csv)
    trades = cs.extract_trades(df)
    run(trades)
