"""
core_strategy.py — Main Backtester for MNQ Multi-Window Strategy

Entry:  close of entry candle, direction = candle direction
Exit:   close of exit candle (per window definition)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

import config

warnings.filterwarnings("ignore")


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and parse the 1-minute bar CSV."""
    df = pd.read_csv(csv_path)
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    # timestamp ET
    ts_col = next(c for c in df.columns if "timestamp" in c.lower())
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"]    = df["timestamp"].dt.date
    df["time"]    = df["timestamp"].dt.strftime("%H:%M")
    df["weekday"] = df["timestamp"].dt.dayofweek   # 0=Mon … 4=Fri; 3=Thu
    df["month"]   = df["timestamp"].dt.month
    return df


# ── Smart Filter ─────────────────────────────────────────────────────────────

def smart_filter(row: pd.Series) -> bool:
    """Return True if the trade should be SKIPPED."""
    if config.SKIP_THURSDAYS and row["weekday"] == 3:
        return True
    if row["month"] in config.SKIP_MONTHS:
        return True
    body = abs(row["entry_close"] - row["entry_open"])
    if body < config.MIN_CANDLE_BODY:
        return True
    return False


# ── Trade Extraction ──────────────────────────────────────────────────────────

def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading day and each window in config.TRADE_WINDOWS, extract the
    entry candle and exit candle close.  Returns a DataFrame with one row per
    potential trade (day × window).  Missing windows for a given day are skipped
    gracefully.
    """
    records = []
    for date, grp in df.groupby("date"):
        for window in config.TRADE_WINDOWS:
            entry_time = window["entry_time"]
            exit_time  = window["exit_time"]
            entry_row  = grp[grp["time"] == entry_time]
            exit_row   = grp[grp["time"] == exit_time]
            if entry_row.empty or exit_row.empty:
                continue
            e = entry_row.iloc[0]
            x = exit_row.iloc[0]
            direction = "LONG" if e["close"] >= e["open"] else "SHORT"
            body = abs(e["close"] - e["open"])
            pnl_long  = x["close"] - e["close"]   # points if long
            pnl_short = e["close"] - x["close"]   # points if short
            records.append({
                "date":          date,
                "window_name":   window["name"],
                "entry_time":    entry_time,
                "exit_time":     exit_time,
                "weekday":       e["weekday"],
                "month":         e["month"],
                "entry_open":    e["open"],
                "entry_close":   e["close"],
                "entry_high":    e["high"],
                "entry_low":     e["low"],
                "entry_volume":  e["volume"],
                "exit_close":    x["close"],
                "direction":     direction,
                "candle_body":   body,
                "pnl_long_pts":  pnl_long,
                "pnl_short_pts": pnl_short,
                # full window high/low
                "window_high":   grp[(grp["time"] >= entry_time) & (grp["time"] <= exit_time)]["high"].max(),
                "window_low":    grp[(grp["time"] >= entry_time) & (grp["time"] <= exit_time)]["low"].min(),
            })
    trades = pd.DataFrame(records)
    if not trades.empty:
        trades["window_range"] = trades["window_high"] - trades["window_low"]
        # MAE (max adverse excursion)
        def _mae(r):
            if r["direction"] == "LONG":
                return max(0.0, r["entry_close"] - r["window_low"])
            else:
                return max(0.0, r["window_high"] - r["entry_close"])
        trades["mae"] = trades.apply(_mae, axis=1)
        trades["skip"] = trades.apply(smart_filter, axis=1)
    return trades


# ── Backtest Engine ───────────────────────────────────────────────────────────

def run_backtest(
    trades: pd.DataFrame,
    direction: str = "both",   # "long", "short", "both"
    contracts: int = 1,
    point_value: float = None,
    initial_capital: float = None,
) -> dict:
    """
    Run a single backtest pass.

    direction: 'long'  — only take LONG signals
               'short' — only take SHORT signals
               'both'  — take all signals (follow candle direction)
    """
    pv   = point_value    or config.DEFAULT_POINT_VALUE
    cap  = initial_capital or config.INITIAL_CAPITAL

    filtered = trades[~trades["skip"]].copy()

    pnl_col = {
        "long":  "pnl_long_pts",
        "short": "pnl_short_pts",
        "both":  None,
    }[direction]

    results = []
    for _, row in filtered.iterrows():
        if direction == "long" and row["direction"] != "LONG":
            continue
        if direction == "short" and row["direction"] != "SHORT":
            continue
        if direction == "both":
            pts = row["pnl_long_pts"] if row["direction"] == "LONG" else row["pnl_short_pts"]
        else:
            pts = row[pnl_col]

        results.append({
            "date":       row["date"],
            "window_name": row.get("window_name", ""),
            "direction":  row["direction"] if direction == "both" else direction.upper(),
            "entry":      row["entry_close"],
            "exit":       row["exit_close"],
            "pts":        pts,
            "pnl":        pts * pv * contracts,
            "contracts":  contracts,
            "entry_open":  row["entry_open"],
            "entry_high":  row["entry_high"],
            "entry_low":   row["entry_low"],
            "entry_close": row["entry_close"],
            "window_high": row["window_high"],
            "window_low":  row["window_low"],
        })

    if not results:
        return {"trades": pd.DataFrame(), "equity": pd.Series([cap]), "stats": {}}

    res = pd.DataFrame(results)
    res["equity"] = cap + res["pnl"].cumsum()

    stats = compute_stats(res, cap, pv)
    return {"trades": res, "equity": res["equity"], "stats": stats}


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(res: pd.DataFrame, initial_capital: float, point_value: float) -> dict:
    pnl   = res["pnl"].values
    pts   = res["pts"].values
    eq    = np.concatenate([[initial_capital], initial_capital + np.cumsum(pnl)])
    n     = len(pnl)

    wins      = pnl[pnl > 0]
    losses    = pnl[pnl < 0]
    win_rate  = len(wins) / n if n else 0.0

    # Returns
    final_cap   = eq[-1]
    total_return = (final_cap - initial_capital) / initial_capital * 100
    # CAGR — approximate number of trading years
    trading_days = n / 252 if n else 1
    cagr = ((final_cap / initial_capital) ** (1 / max(trading_days, 1e-9)) - 1) * 100

    # Drawdown
    peak    = np.maximum.accumulate(eq)
    dd      = eq - peak                          # negative dollars
    dd_pct  = np.where(peak > 0, dd / peak * 100, 0.0)
    max_dd  = dd.min()
    max_dd_pct = dd_pct.min()

    # Max DD duration
    in_dd     = dd < 0
    dd_dur    = 0
    cur_dur   = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            dd_dur = max(dd_dur, cur_dur)
        else:
            cur_dur = 0

    # Sharpe / Sortino / Calmar
    daily_ret = pnl / initial_capital
    rf = 0.0
    excess = daily_ret - rf
    sharpe   = (excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) > 0 else 0.0
    down     = daily_ret[daily_ret < 0]
    sortino  = (excess.mean() / down.std(ddof=1) * np.sqrt(252)) if len(down) > 1 and down.std(ddof=1) > 0 else 0.0
    calmar   = (total_return / 100) / abs(max_dd_pct / 100) if max_dd_pct != 0 else 0.0

    # Profit factor
    gross_profit = wins.sum()  if len(wins)   else 0.0
    gross_loss   = losses.sum() if len(losses) else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")

    # Expectancy
    avg_win  = wins.mean()   if len(wins)   else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Max win / loss
    max_win  = wins.max()   if len(wins)   else 0.0
    max_loss = losses.min() if len(losses) else 0.0

    # Streaks
    streaks = _streaks(pnl)

    # Risk of ruin (simplified)
    if win_rate > 0 and win_rate < 1:
        avg_r = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        ror = ((1 - win_rate) / win_rate) ** (initial_capital / max(abs(avg_loss), 1))
        ror = min(ror, 1.0)
    else:
        ror = 0.0

    return {
        "n_trades":      n,
        "win_rate":      win_rate,
        "n_wins":        len(wins),
        "n_losses":      len(losses),
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "max_win":       max_win,
        "max_loss":      max_loss,
        "expectancy":    expectancy,
        "gross_profit":  gross_profit,
        "gross_loss":    gross_loss,
        "profit_factor": profit_factor,
        "total_pnl":     pnl.sum(),
        "total_pts":     pts.sum(),
        "final_capital": final_cap,
        "total_return":  total_return,
        "cagr":          cagr,
        "max_dd":        max_dd,
        "max_dd_pct":    max_dd_pct,
        "max_dd_dur":    dd_dur,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "calmar":        calmar,
        "risk_of_ruin":  ror,
        "max_cons_wins":   streaks[0],
        "max_cons_losses": streaks[1],
    }


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


# ── Buy & Hold Benchmark ──────────────────────────────────────────────────────

def buy_and_hold(df: pd.DataFrame, initial_capital: float = None) -> dict:
    """Simple buy-and-hold on MNQ from first to last available close."""
    cap = initial_capital or config.INITIAL_CAPITAL
    pv  = config.DEFAULT_POINT_VALUE
    first = df["close"].iloc[0]
    last  = df["close"].iloc[-1]
    pts   = last - first
    pnl   = pts * pv * 1   # 1 contract
    final = cap + pnl
    ret   = (final - cap) / cap * 100
    return {
        "initial": cap,
        "final":   final,
        "pts":     pts,
        "pnl":     pnl,
        "return":  ret,
    }


# ── Console Comparison Table ──────────────────────────────────────────────────

def print_comparison(results: dict):
    """Print side-by-side stats for all four approaches."""
    keys = ["long", "short", "both", "bh"]
    labels = {
        "long":  "Bullish Only",
        "short": "Bearish Only",
        "both":  "Both (Follow)",
        "bh":    "Buy & Hold",
    }
    stat_keys = [
        ("n_trades",      "Total Trades"),
        ("win_rate",      "Win Rate %"),
        ("total_pnl",     "Total P&L $"),
        ("total_pts",     "Total P&L pts"),
        ("final_capital", "Final Capital $"),
        ("total_return",  "Total Return %"),
        ("cagr",          "CAGR %"),
        ("max_dd",        "Max DD $"),
        ("max_dd_pct",    "Max DD %"),
        ("sharpe",        "Sharpe"),
        ("sortino",       "Sortino"),
        ("calmar",        "Calmar"),
        ("profit_factor", "Profit Factor"),
        ("expectancy",    "Expectancy $"),
        ("max_cons_wins", "Max Cons. Wins"),
        ("max_cons_losses","Max Cons. Losses"),
    ]
    col_w = 16
    header = f"{'Metric':<24}" + "".join(f"{labels[k]:>{col_w}}" for k in keys)
    print("\n" + "=" * (24 + col_w * len(keys)))
    print("  STRATEGY COMPARISON")
    print("=" * (24 + col_w * len(keys)))
    print(header)
    print("-" * (24 + col_w * len(keys)))
    for sk, label in stat_keys:
        row = f"{label:<24}"
        for k in keys:
            st = results.get(k, {}).get("stats", {})
            v  = st.get(sk, float("nan"))
            if sk == "win_rate":
                row += f"{v*100:>{col_w}.1f}"
            elif isinstance(v, float):
                row += f"{v:>{col_w},.1f}"
            else:
                row += f"{v:>{col_w}}"
        print(row)
    print("=" * (24 + col_w * len(keys)))


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_comparison_charts(results: dict, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MNQ Multi-Window Strategy — Comparison", fontsize=14, fontweight="bold")

    colors = {"long": "#26a69a", "short": "#ef5350", "both": "#42a5f5", "bh": "#ffa726"}
    labels = {"long": "Bullish Only", "short": "Bearish Only", "both": "Both", "bh": "Buy & Hold"}

    # 1 — Equity curves
    ax = axes[0, 0]
    for k, res in results.items():
        if k == "bh":
            continue
        eq = res.get("equity")
        if eq is not None and len(eq):
            ax.plot(eq.values, label=labels[k], color=colors[k])
    ax.axhline(config.INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Equity Curves")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Capital ($)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2 — Win rate bar
    ax = axes[0, 1]
    dirs = [k for k in ["long", "short", "both"] if k in results]
    wr   = [results[k]["stats"].get("win_rate", 0) * 100 for k in dirs]
    bars = ax.bar(dirs, wr, color=[colors[k] for k in dirs])
    ax.set_title("Win Rate by Direction")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, wr):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    # 3 — P&L distribution
    ax = axes[1, 0]
    for k in ["long", "short", "both"]:
        if k not in results:
            continue
        tr = results[k].get("trades")
        if tr is not None and len(tr):
            ax.hist(tr["pnl"].values, bins=30, alpha=0.5, label=labels[k], color=colors[k])
    ax.axvline(0, color="white", linestyle="--", alpha=0.7)
    ax.set_title("P&L Distribution")
    ax.set_xlabel("P&L ($)")
    ax.legend()

    # 4 — Monthly P&L (both direction)
    ax = axes[1, 1]
    if "both" in results:
        tr = results["both"].get("trades")
        if tr is not None and len(tr):
            tr = tr.copy()
            tr["ym"] = pd.to_datetime(tr["date"].astype(str)).dt.to_period("M")
            monthly = tr.groupby("ym")["pnl"].sum()
            clrs = ["#26a69a" if v >= 0 else "#ef5350" for v in monthly.values]
            ax.bar(range(len(monthly)), monthly.values, color=clrs)
            ax.set_xticks(range(0, len(monthly), max(1, len(monthly) // 8)))
            ax.set_xticklabels(
                [str(monthly.index[i]) for i in range(0, len(monthly), max(1, len(monthly) // 8))],
                rotation=45, fontsize=7
            )
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Monthly P&L (Both Direction)")
            ax.set_ylabel("P&L ($)")

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_charts.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_drawdown_comparison(results: dict, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Drawdown Comparison", fontsize=13, fontweight="bold")
    colors = {"long": "#26a69a", "short": "#ef5350", "both": "#42a5f5"}

    for ax, k in zip(axes, ["long", "short", "both"]):
        if k not in results:
            ax.set_visible(False)
            continue
        eq = results[k].get("equity")
        if eq is None or not len(eq):
            ax.set_visible(False)
            continue
        eq_arr = np.concatenate([[config.INITIAL_CAPITAL], eq.values])
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak * 100
        ax.fill_between(range(len(dd)), dd, 0, color=colors[k], alpha=0.6)
        ax.set_title(k.capitalize())
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "drawdown_comparison.png")
    fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── TradingView-Style Last 5 Trades Charts ────────────────────────────────────

def plot_last5_trades(trades_df: pd.DataFrame, df_1min: pd.DataFrame,
                      direction_label: str, output_dir: str = "."):
    """Plot the last 5 trades with dark-theme TradingView-style candlestick charts."""
    os.makedirs(output_dir, exist_ok=True)
    BG   = "#131722"
    BULL = "#26a69a"
    BEAR = "#ef5350"

    recent = trades_df.tail(5).reset_index(drop=True)

    for i, row in recent.iterrows():
        date_str = str(row["date"])
        day_data = df_1min[df_1min["date"].astype(str) == date_str].copy()
        # Dynamic focus window: ±10 minutes around the trade's entry/exit times
        entry_time = row.get("entry_time", config.ENTRY_TIME)
        exit_time  = row.get("exit_time",  config.EXIT_TIME)
        # Compute window start/end as HH:MM strings with 10-min padding
        def _add_minutes(hhmm: str, delta: int) -> str:
            h, m = int(hhmm[:2]), int(hhmm[3:])
            total = h * 60 + m + delta
            total = max(0, min(total, 23 * 60 + 59))
            return f"{total // 60:02d}:{total % 60:02d}"
        win_start = _add_minutes(entry_time, -10)
        win_end   = _add_minutes(exit_time,  +10)
        window = day_data[(day_data["time"] >= win_start) & (day_data["time"] <= win_end)].copy()
        if window.empty:
            continue

        fig = plt.figure(figsize=(10, 6), facecolor=BG)
        gs  = GridSpec(3, 1, figure=fig, hspace=0.05, height_ratios=[3, 1, 0.5])
        ax_c = fig.add_subplot(gs[0])  # candles
        ax_v = fig.add_subplot(gs[1], sharex=ax_c)  # volume
        ax_b = fig.add_subplot(gs[2])  # badge

        for ax in [ax_c, ax_v]:
            ax.set_facecolor(BG)
            ax.tick_params(colors="#d1d4dc", labelsize=8)
            ax.spines["bottom"].set_color("#2a2e39")
            ax.spines["top"].set_color("#2a2e39")
            ax.spines["left"].set_color("#2a2e39")
            ax.spines["right"].set_color("#2a2e39")
            ax.yaxis.label.set_color("#d1d4dc")
            ax.xaxis.label.set_color("#d1d4dc")
            ax.grid(True, color="#2a2e39", alpha=0.5)

        # Draw candles
        xs = range(len(window))
        for j, (_, c) in enumerate(window.iterrows()):
            color = BULL if c["close"] >= c["open"] else BEAR
            ax_c.plot([j, j], [c["low"], c["high"]], color=color, linewidth=1)
            body_bot = min(c["open"], c["close"])
            body_top = max(c["open"], c["close"])
            ax_c.add_patch(plt.Rectangle(
                (j - 0.3, body_bot), 0.6, max(body_top - body_bot, 0.1),
                color=color, zorder=3
            ))

        # Volume bars
        for j, (_, c) in enumerate(window.iterrows()):
            color = BULL if c["close"] >= c["open"] else BEAR
            ax_v.bar(j, c["volume"], color=color, alpha=0.7, width=0.8)

        # Entry / exit markers
        times_list = window["time"].tolist()
        try:
            ei = times_list.index(entry_time)
            xi = times_list.index(exit_time)
        except ValueError:
            plt.close(fig)
            continue

        entry_price = row["entry"]
        exit_price  = row["exit"]
        pnl         = row["pnl"]
        is_long     = row["direction"] == "LONG"

        arrow_kw = dict(arrowstyle="->", color="white", lw=1.5)
        ax_c.annotate(
            f"{'▲' if is_long else '▼'} Entry\n{entry_price:.2f}",
            xy=(ei, entry_price),
            xytext=(ei - 1, entry_price + (10 if is_long else -10)),
            color=BULL if is_long else BEAR, fontsize=7, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BULL if is_long else BEAR, lw=1.2),
        )
        ax_c.annotate(
            f"{'▼' if is_long else '▲'} Exit\n{exit_price:.2f}",
            xy=(xi, exit_price),
            xytext=(xi + 0.5, exit_price + (10 if not is_long else -10)),
            color=BULL if pnl >= 0 else BEAR, fontsize=7,
            arrowprops=dict(arrowstyle="->", color=BULL if pnl >= 0 else BEAR, lw=1.2),
        )
        ax_c.annotate(
            "", xy=(xi, exit_price), xytext=(ei, entry_price),
            arrowprops=dict(arrowstyle="-", linestyle="dashed", color="yellow", lw=0.8, alpha=0.7)
        )

        window_label = row.get("window_name", f"{entry_time}→{exit_time}")
        ax_c.set_title(
            f"MNQ Multi-Window Trade — {date_str}   [{direction_label}]   {window_label}",
            color="#d1d4dc", fontsize=10, pad=6
        )
        ax_c.set_ylabel("Price", color="#d1d4dc")

        # Badge
        ax_b.set_facecolor(BG)
        ax_b.axis("off")
        badge_color = "#26a69a" if pnl >= 0 else "#ef5350"
        ax_b.text(
            0.5, 0.5,
            f"P&L: {'+'if pnl>=0 else ''}{pnl:,.0f}$   pts: {'+'if row['pts']>=0 else ''}{row['pts']:.1f}",
            transform=ax_b.transAxes,
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color, edgecolor="none")
        )

        # X-axis labels
        plt.setp(ax_c.get_xticklabels(), visible=False)
        ax_v.set_xticks(xs)
        ax_v.set_xticklabels(
            [t if j % 2 == 0 else "" for j, t in enumerate(times_list)],
            rotation=45, fontsize=7, color="#d1d4dc"
        )
        ax_v.set_ylabel("Vol", color="#d1d4dc")

        path = os.path.join(output_dir, f"last5_trade_{i+1}_{date_str}_{direction_label}.png")
        fig.savefig(path, dpi=config.CHART_DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved: {path}")


# ── Trade Log Export ──────────────────────────────────────────────────────────

def export_trade_log(trades_df: pd.DataFrame, direction: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"trade_log_{direction}.csv")
    trades_df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(csv_path: str, initial_capital: float = None, point_value: float = None,
        output_dir: str = "."):
    cap = initial_capital or config.INITIAL_CAPITAL
    pv  = point_value or config.DEFAULT_POINT_VALUE

    print("Loading data …")
    df = load_data(csv_path)
    print(f"  Rows: {len(df):,}   from {df['date'].min()} to {df['date'].max()}")

    print("Extracting trades …")
    trades = extract_trades(df)
    total_days  = trades["date"].nunique()
    passed_days = trades[~trades["skip"]]["date"].nunique()
    print(f"  Trading days: {total_days}   after smart filter: {passed_days}")

    results = {}
    for direction in ["long", "short", "both"]:
        res = run_backtest(trades, direction=direction, contracts=1,
                           point_value=pv, initial_capital=cap)
        results[direction] = res

    # Buy & Hold
    bh = buy_and_hold(df, cap)
    results["bh"] = {
        "trades": pd.DataFrame(),
        "equity": pd.Series([cap, bh["final"]]),
        "stats":  {
            "n_trades": 1, "win_rate": 1.0 if bh["pnl"] > 0 else 0.0,
            "n_wins": 1 if bh["pnl"] > 0 else 0, "n_losses": 0 if bh["pnl"] > 0 else 1,
            "total_pnl": bh["pnl"], "total_pts": bh["pts"],
            "final_capital": bh["final"], "total_return": bh["return"],
            "cagr": bh["return"], "max_dd": 0.0, "max_dd_pct": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "profit_factor": float("inf") if bh["pnl"] > 0 else 0.0,
            "expectancy": bh["pnl"], "max_cons_wins": 1, "max_cons_losses": 0,
        },
    }

    print_comparison(results)

    print("\nGenerating charts …")
    plot_comparison_charts(results, output_dir)
    plot_drawdown_comparison(results, output_dir)

    for direction in ["long", "short", "both"]:
        tr = results[direction].get("trades")
        if tr is not None and len(tr):
            lbl = {"long": "LONG", "short": "SHORT", "both": "BOTH"}[direction]
            plot_last5_trades(tr, df, lbl, output_dir)
            export_trade_log(tr, direction, output_dir)

    return results, trades, df


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else config.CSV_PATH
    run(csv)
