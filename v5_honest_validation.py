# =============================================================================
# v5_honest_validation.py — MNQ "Follow The Candle" Honest Validation Suite
# Designed for Google Colab — upload your 1-min MNQ CSV to get REAL answers.
# Also runnable as a local script: python v5_honest_validation.py --csv data.csv
# =============================================================================
# Every single number in this script is computed from YOUR data.
# No hardcoded results. No fake estimates. Just honest statistics.
# =============================================================================

import subprocess
import sys

# Install dependencies (works in both Colab and local environments)
subprocess.run(
    [sys.executable, "-m", "pip", "-q", "install",
     "pandas", "numpy", "matplotlib", "scikit-learn", "scipy", "seaborn"],
    check=True, capture_output=True
)

import os
import io
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, ttest_ind, ttest_1samp

warnings.filterwarnings("ignore")

# ── Data loading: Colab upload or local --csv argument ───────────────────────
try:
    from google.colab import files as _colab_files
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

if _IN_COLAB:
    print("📂 Please upload your MNQ 1-minute CSV file...")
    uploaded = _colab_files.upload()
    fname     = list(uploaded.keys())[0]
    csv_bytes = list(uploaded.values())[0]
    print(f"✅ Uploaded: {fname}")
else:
    import argparse
    _parser = argparse.ArgumentParser(
        description="MNQ Honest Validation — run locally with a CSV path"
    )
    _parser.add_argument("--csv", required=True, help="Path to 1-minute MNQ CSV file")
    _args  = _parser.parse_args()
    fname  = _args.csv
    with open(fname, "rb") as _f:
        csv_bytes = _f.read()
    print(f"✅ Loaded: {fname}")

# ── Output directory ─────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================
POINT_VALUE    = 2.0       # $/point per MNQ contract
MIN_BODY       = 3.0       # minimum candle body to take trade (points)
ACCOUNT_SIZE   = 50_000    # prop firm account size ($)
PROFIT_TARGET  = 3_000     # challenge profit target ($)
MAX_DD         = 2_000     # max drawdown allowed ($)
CHALLENGE_DAYS = 30        # challenge duration (days)
N_SIMS         = 10_000    # Monte Carlo / permutation iterations
DAILY_LIMIT    = 1_000     # Topstep-style daily loss limit ($)
DEFAULT_CONTRACTS = 7      # "aggressive" sizing from previous research

# The 5 user-specified windows
USER_WINDOWS = [
    (12, 16, 12, 27),   # entry_h, entry_m, exit_h, exit_m
    (12, 51, 13,  0),
    (13, 55, 14,  5),
    (14, 50, 15,  1),
    (15, 48, 15, 58),
]

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(csv_bytes):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    ts_col = [c for c in df.columns if "timestamp" in c or "time" in c][0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"]    = df["timestamp"].dt.date
    df["hour"]    = df["timestamp"].dt.hour
    df["minute"]  = df["timestamp"].dt.minute
    df["weekday"] = df["timestamp"].dt.weekday   # Mon=0, Thu=3
    df["month"]   = df["timestamp"].dt.month
    df["body"]    = (df["close"] - df["open"]).abs()
    return df

df_all = load_data(csv_bytes)
print(f"✅ Loaded {len(df_all):,} rows | "
      f"{df_all['date'].nunique()} trading days | "
      f"{df_all['timestamp'].min().date()} → {df_all['timestamp'].max().date()}")

t_total_start = time.time()
print("\n⏱️ ESTIMATED RUNTIME (with vectorized engine)")
print("  Section 1 (Walk-Forward Scan)   ~30-90 sec")
print("  Section 2 (Worst-Day Analysis)  < 5 sec")
print("  Section 3 (Correlation)         < 5 sec")
print("  Section 4 (Filter Validation)   < 5 sec")
print("  Section 5 (Position Sizing)     ~10-20 sec")
print("  Section 6 (Permutation Test)    ~30-90 sec")
print("  Section 7 (Summary)             < 1 sec")
print("  ─────────────────────────────────────────")
print("  TOTAL ESTIMATE: ~2-4 minutes")

# =============================================================================
# CORE TRADE GENERATION
# =============================================================================
def get_trades_for_window(df, entry_h, entry_m, exit_h, exit_m,
                           skip_thu=True, skip_jun=True, skip_oct=True,
                           min_body=MIN_BODY):
    """Return trades for a single (entry, exit) window — vectorized for speed."""
    empty = pd.DataFrame(columns=["date","entry_h","entry_m","exit_h","exit_m",
                                  "direction","pnl_pts","body"])
    entry_bars = df[(df["hour"] == entry_h) & (df["minute"] == entry_m)]
    exit_bars  = df[(df["hour"] == exit_h)  & (df["minute"] == exit_m)]
    if entry_bars.empty or exit_bars.empty:
        return empty
    # One bar per date (safety for duplicate timestamps)
    eb = entry_bars.drop_duplicates(subset="date", keep="first").set_index("date")
    xb = exit_bars.drop_duplicates(subset="date", keep="first").set_index("date")
    merged = eb[["open", "close", "weekday", "month"]].join(
        xb[["close"]].rename(columns={"close": "exit_close"}), how="inner"
    )
    if merged.empty:
        return empty
    # Calendar filters
    if skip_thu:
        merged = merged[merged["weekday"] != 3]
    if skip_jun:
        merged = merged[merged["month"] != 6]
    if skip_oct:
        merged = merged[merged["month"] != 10]
    # Body filter
    merged["body"] = (merged["close"] - merged["open"]).abs()
    if min_body > 0:
        merged = merged[merged["body"] >= min_body]
    if merged.empty:
        return empty
    # Direction & P&L
    merged["direction"] = np.where(merged["close"] > merged["open"], 1, -1)
    merged["pnl_pts"]   = merged["direction"] * (merged["exit_close"] - merged["close"])
    return pd.DataFrame({
        "date":      merged.index,
        "entry_h":   entry_h,
        "entry_m":   entry_m,
        "exit_h":    exit_h,
        "exit_m":    exit_m,
        "direction": merged["direction"].values,
        "pnl_pts":   merged["pnl_pts"].values,
        "body":      merged["body"].values,
    })


def window_stats(trades_df, contracts=1):
    """Compute key performance metrics for a trades DataFrame."""
    if trades_df.empty or len(trades_df) < 5:
        return None
    pnl = trades_df["pnl_pts"].values * contracts * POINT_VALUE
    n   = len(pnl)
    wins = pnl[pnl > 0]
    loss = pnl[pnl < 0]
    win_rate = len(wins) / n if n else 0
    avg_pnl  = pnl.mean()
    std_pnl  = pnl.std(ddof=1) if n > 1 else 1e-9
    sharpe   = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0
    pf       = (wins.sum() / abs(loss.sum())) if len(loss) > 0 and loss.sum() != 0 else np.inf
    cum      = np.cumsum(pnl)
    peak     = np.maximum.accumulate(cum)
    max_dd   = (cum - peak).min() if len(cum) else 0
    return dict(n=n, win_rate=win_rate, avg_pnl=avg_pnl, std_pnl=std_pnl,
                sharpe=sharpe, profit_factor=pf, total_pnl=pnl.sum(), max_dd=max_dd)


# Precompute all USER_WINDOWS trades on full dataset
all_user_trades = {}
for w in USER_WINDOWS:
    trades = get_trades_for_window(df_all, *w)
    label  = f"{w[0]:02d}:{w[1]:02d}→{w[2]:02d}:{w[3]:02d}"
    all_user_trades[label] = trades

print(f"✅ User windows loaded: "
      + ", ".join(f"{k}({len(v)}tr)" for k, v in all_user_trades.items()))

# =============================================================================
# SECTION 1 — WALK-FORWARD / OUT-OF-SAMPLE TEST
# =============================================================================
print("\n" + "═"*70)
print("  SECTION 1 — WALK-FORWARD / OUT-OF-SAMPLE TEST")
print("═"*70)
t_sec1 = time.time()

all_days = sorted(df_all["date"].unique())
split_idx = int(len(all_days) * 0.60)
train_days = set(all_days[:split_idx])
test_days  = set(all_days[split_idx:])
df_train   = df_all[df_all["date"].isin(train_days)].copy()
df_test    = df_all[df_all["date"].isin(test_days)].copy()

print(f"  Training set : {len(train_days)} days  "
      f"({min(train_days)} → {max(train_days)})")
print(f"  Test set     : {len(test_days)} days  "
      f"({min(test_days)} → {max(test_days)})")

# --- Scan ALL windows on TRAINING data ---
print("\n  🔍 Scanning all minute windows on TRAINING data...")

candidate_results = []
scan_count = 0
for entry_h in range(9, 16):
    for entry_m in range(0, 60):
        for hold in range(5, 16):
            # compute exit time
            total_m  = entry_h * 60 + entry_m + hold
            exit_h   = total_m // 60
            exit_m   = total_m %  60
            if exit_h > 15 or (exit_h == 15 and exit_m > 59):
                continue
            scan_count += 1
            trades = get_trades_for_window(df_train, entry_h, entry_m, exit_h, exit_m)
            if len(trades) < 10:
                continue
            stats = window_stats(trades)
            if stats is None:
                continue
            candidate_results.append({
                "entry_h": entry_h, "entry_m": entry_m,
                "exit_h": exit_h, "exit_m": exit_m,
                "hold": hold,
                "label": f"{entry_h:02d}:{entry_m:02d}→{exit_h:02d}:{exit_m:02d}",
                **stats
            })
    print(f"    ... scanned hour {entry_h}:xx ({scan_count} windows so far, "
          f"{time.time()-t_sec1:.0f}s elapsed)")

print(f"  ✅ Scanned {scan_count} windows in {time.time()-t_sec1:.1f}s, "
      f"found {len(candidate_results)} with ≥10 trades")

cand_df = pd.DataFrame(candidate_results).sort_values("sharpe", ascending=False)
top5_train = cand_df.head(5)

print("\n  📊 TOP-5 WINDOWS SELECTED FROM TRAINING DATA:")
print(f"  {'Window':<22} {'N':>5} {'WinRate':>8} {'AvgPnL':>8} {'Sharpe':>8}")
print("  " + "-"*54)
for _, row in top5_train.iterrows():
    print(f"  {row['label']:<22} {int(row['n']):>5} {row['win_rate']:>7.1%} "
          f"  ${row['avg_pnl']:>6.2f}  {row['sharpe']:>7.3f}")

# --- Test SELECTED windows on TEST set ---
print("\n  🎯 OUT-OF-SAMPLE RESULTS (selected windows on TEST set):")
print(f"  {'Window':<22} {'N':>5} {'WinRate':>8} {'AvgPnL':>8} {'Sharpe':>8}  {'IS→OOS Δ'}")
print("  " + "-"*65)

selected_oos_sharpes = []
for _, row in top5_train.iterrows():
    trades_oos = get_trades_for_window(
        df_test, row["entry_h"], row["entry_m"], row["exit_h"], row["exit_m"]
    )
    stats_oos = window_stats(trades_oos)
    if stats_oos:
        delta = stats_oos["sharpe"] - row["sharpe"]
        marker = "🟢" if stats_oos["sharpe"] > 0.5 else ("🟡" if stats_oos["sharpe"] > 0 else "🔴")
        print(f"  {row['label']:<22} {int(stats_oos['n']):>5} "
              f"{stats_oos['win_rate']:>7.1%}   ${stats_oos['avg_pnl']:>6.2f}  "
              f"{stats_oos['sharpe']:>7.3f}  {delta:+.3f} {marker}")
        selected_oos_sharpes.append(stats_oos["sharpe"])
    else:
        print(f"  {row['label']:<22}  (insufficient OOS trades)")

# --- Test USER's specific 5 windows on TEST set ---
print("\n  📌 USER'S SPECIFIC 5 WINDOWS ON TEST SET:")
print(f"  {'Window':<22} {'N':>5} {'WinRate':>8} {'AvgPnL':>8} {'Sharpe':>8}")
print("  " + "-"*54)
for label, trades_all in all_user_trades.items():
    trades_oos = trades_all[trades_all["date"].isin(test_days)]
    stats_oos  = window_stats(trades_oos)
    if stats_oos:
        marker = "🟢" if stats_oos["sharpe"] > 0.5 else ("🟡" if stats_oos["sharpe"] > 0 else "🔴")
        print(f"  {label:<22} {int(stats_oos['n']):>5} "
              f"{stats_oos['win_rate']:>7.1%}   ${stats_oos['avg_pnl']:>6.2f}  "
              f"{stats_oos['sharpe']:>7.3f}  {marker}")
    else:
        print(f"  {label:<22}  (insufficient OOS trades)")

# IS vs OOS summary table
print("\n  📊 IN-SAMPLE vs OUT-OF-SAMPLE (User's 5 Windows Combined):")
for label, trades_all in all_user_trades.items():
    trades_is  = trades_all[trades_all["date"].isin(train_days)]
    trades_oos = trades_all[trades_all["date"].isin(test_days)]
    si = window_stats(trades_is)
    so = window_stats(trades_oos)
    if si and so:
        print(f"  {label}: IS Sharpe={si['sharpe']:.3f} WR={si['win_rate']:.1%} | "
              f"OOS Sharpe={so['sharpe']:.3f} WR={so['win_rate']:.1%} | "
              f"Δ={so['sharpe']-si['sharpe']:+.3f}")

print(f"\n  ⏱️ Section 1 completed in {time.time()-t_sec1:.1f}s")

# =============================================================================
# SECTION 2 — WORST-DAY ANALYSIS
# =============================================================================
print("\n" + "═"*70)
print("  SECTION 2 — WORST-DAY ANALYSIS  (7 contracts)")
print("═"*70)
t_sec2 = time.time()

# Build daily P&L per window and combined
daily_pnl_by_window = {}
for label, trades in all_user_trades.items():
    dp = (trades.groupby("date")["pnl_pts"].sum() * DEFAULT_CONTRACTS * POINT_VALUE)
    daily_pnl_by_window[label] = dp

# Align to union of all dates where at least 1 window traded
all_trade_dates = sorted(
    set().union(*[set(v.index) for v in daily_pnl_by_window.values()])
)
daily_matrix = pd.DataFrame(
    {k: v.reindex(all_trade_dates, fill_value=0) for k, v in daily_pnl_by_window.items()},
    index=all_trade_dates
)
daily_combined = daily_matrix.sum(axis=1)

print(f"\n  Total trading days with ≥1 window trigger: {len(daily_combined)}")

worst10 = daily_combined.nsmallest(10)
best10  = daily_combined.nlargest(10)

print("\n  🔴 WORST 10 DAYS:")
for d, pnl in worst10.items():
    print(f"     {d}   ${pnl:>8,.2f}")

print("\n  🟢 BEST 10 DAYS:")
for d, pnl in best10.items():
    print(f"     {d}   ${pnl:>8,.2f}")

# Days where ALL 5 windows triggered AND all 5 lost
all_5_triggered = (daily_matrix != 0).all(axis=1)
all_5_lost      = (daily_matrix < 0).all(axis=1)
days_all5_trigger = all_5_triggered.sum()
days_all5_lose    = (all_5_triggered & all_5_lost).sum()
print(f"\n  Days all 5 windows triggered: {days_all5_trigger}")
print(f"  Days all 5 windows triggered AND all 5 lost: {days_all5_lose} "
      f"({'🔴 HIGH RISK' if days_all5_lose > 5 else '🟡 moderate' if days_all5_lose > 0 else '🟢 none observed'})")

# Consecutive losing days
losing_streak = 0
max_streak    = 0
cur_streak    = 0
for pnl in daily_combined.values:
    if pnl < 0:
        cur_streak += 1
        max_streak = max(max_streak, cur_streak)
    else:
        cur_streak = 0
print(f"  Max consecutive losing days: {max_streak} "
      f"({'🔴' if max_streak >= 5 else '🟡' if max_streak >= 3 else '🟢'})")

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(daily_combined.values, bins=40, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="black", linestyle="--")
axes[0].axvline(worst10.iloc[0], color="red", linestyle="--", label=f"Worst: ${worst10.iloc[0]:,.0f}")
axes[0].set_title("Daily P&L Distribution (7 contracts, all 5 windows combined)")
axes[0].set_xlabel("Daily P&L ($)")
axes[0].set_ylabel("Days")
axes[0].legend()

# Equity curve
eq_curve = daily_combined.cumsum()
axes[1].plot(range(len(eq_curve)), eq_curve.values, color="navy")
axes[1].fill_between(range(len(eq_curve)), eq_curve.values, 0,
                     where=(eq_curve.values >= 0), alpha=0.2, color="green")
axes[1].fill_between(range(len(eq_curve)), eq_curve.values, 0,
                     where=(eq_curve.values < 0), alpha=0.2, color="red")
axes[1].set_title("Cumulative P&L — All 5 Windows (7 contracts)")
axes[1].set_xlabel("Trading Days")
axes[1].set_ylabel("Cumulative P&L ($)")
plt.tight_layout()
plt.savefig("outputs/worst_day_analysis.png", dpi=150)
plt.show()
print("  📊 Chart saved → outputs/worst_day_analysis.png")
print(f"  ⏱️ Section 2 completed in {time.time()-t_sec2:.1f}s")

# =============================================================================
# SECTION 3 — WINDOW CORRELATION ANALYSIS
# =============================================================================
print("\n" + "═"*70)
print("  SECTION 3 — WINDOW CORRELATION ANALYSIS")
print("═"*70)
t_sec3 = time.time()

# Only look at days where BOTH windows traded (non-zero)
window_labels = list(daily_matrix.columns)
corr_matrix   = pd.DataFrame(index=window_labels, columns=window_labels, dtype=float)

for w1 in window_labels:
    for w2 in window_labels:
        if w1 == w2:
            corr_matrix.loc[w1, w2] = 1.0
            continue
        mask = (daily_matrix[w1] != 0) & (daily_matrix[w2] != 0)
        if mask.sum() < 10:
            corr_matrix.loc[w1, w2] = np.nan
            continue
        r, _ = pearsonr(daily_matrix.loc[mask, w1], daily_matrix.loc[mask, w2])
        corr_matrix.loc[w1, w2] = r

print("\n  Pearson Correlation of Daily P&L Between Windows:")
print("  (positive = tend to win/lose together; negative = good diversification)")
print()
print(corr_matrix.to_string())

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
mask_nan = corr_matrix.isnull()
sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, ax=ax, mask=mask_nan.values)
ax.set_title("Window Pair Correlation (Daily P&L)")
plt.tight_layout()
plt.savefig("outputs/window_correlation.png", dpi=150)
plt.show()
print("  📊 Chart saved → outputs/window_correlation.png")

# Count days with 3+, 4+, 5 windows all losing
n_losing_per_day = (daily_matrix < 0).sum(axis=1)
triggered_per_day = (daily_matrix != 0).sum(axis=1)
print(f"\n  Days where 3+ of the 5 windows all LOST: "
      f"{(n_losing_per_day >= 3).sum()}")
print(f"  Days where 4+ of the 5 windows all LOST: "
      f"{(n_losing_per_day >= 4).sum()}")
print(f"  Days where ALL 5 windows all LOST:        "
      f"{(n_losing_per_day == 5).sum()}")

avg_corr = corr_matrix.values.astype(float)
np.fill_diagonal(avg_corr, np.nan)
mean_corr = np.nanmean(avg_corr)
print(f"\n  Mean inter-window correlation: {mean_corr:.3f} "
      f"({'🔴 high — losses cluster!' if mean_corr > 0.5 else '🟡 moderate' if mean_corr > 0.2 else '🟢 low — good diversification'})")
print(f"  ⏱️ Section 3 completed in {time.time()-t_sec3:.1f}s")

# =============================================================================
# SECTION 4 — FILTER VALIDATION
# =============================================================================
print("\n" + "═"*70)
print("  SECTION 4 — FILTER VALIDATION")
print("═"*70)
print("  Testing whether each filter is justified by data, or might be overfit.\n")
t_sec4 = time.time()

filter_tests = [
    ("Skip Thursday",    {"skip_thu": True},  {"skip_thu": False}),
    ("Skip June",        {"skip_jun": True},  {"skip_jun": False}),
    ("Skip October",     {"skip_oct": True},  {"skip_oct": False}),
    ("Body ≥ 3.0 pts",  {"min_body": 3.0},   {"min_body": 0.0}),
]

for filter_name, with_kw, without_kw in filter_tests:
    # Collect P&L from all 5 user windows under each condition
    all_with    = []
    all_without = []
    for w in USER_WINDOWS:
        kw_with    = dict(skip_thu=True, skip_jun=True, skip_oct=True, min_body=MIN_BODY)
        kw_without = dict(skip_thu=True, skip_jun=True, skip_oct=True, min_body=MIN_BODY)
        kw_with.update(with_kw)
        kw_without.update(without_kw)
        t_with    = get_trades_for_window(df_all, *w, **kw_with)
        t_without = get_trades_for_window(df_all, *w, **kw_without)
        all_with.extend(t_with["pnl_pts"].tolist())
        all_without.extend(t_without["pnl_pts"].tolist())

    arr_with    = np.array(all_with)
    arr_without = np.array(all_without)

    wr_with    = (arr_with > 0).mean()
    wr_without = (arr_without > 0).mean()
    avg_with   = arr_with.mean()
    avg_without = arr_without.mean()

    std_with = arr_with.std(ddof=1) if len(arr_with) > 1 else 1e-9
    sr_with  = (avg_with / std_with) * np.sqrt(252) if std_with > 0 else 0

    std_without = arr_without.std(ddof=1) if len(arr_without) > 1 else 1e-9
    sr_without  = (avg_without / std_without) * np.sqrt(252) if std_without > 0 else 0

    # t-test: are the filtered-in and all-trades significantly different?
    if len(arr_with) > 5 and len(arr_without) > 5:
        t_stat, p_val = ttest_ind(arr_with, arr_without, equal_var=False)
    else:
        t_stat, p_val = 0.0, 1.0

    justified = (avg_with > avg_without) and (wr_with > wr_without)
    if justified and p_val < 0.10:
        verdict = "🟢 Justified (statistically significant benefit)"
    elif justified and p_val < 0.20:
        verdict = "🟡 Mildly justified (p < 0.20, weak significance)"
    elif justified:
        verdict = "🟡 Direction correct but NOT statistically significant — possible overfit"
    else:
        verdict = "🔴 Filter may HURT or make no difference — possible overfit"

    print(f"  ── {filter_name}")
    print(f"     With filter:    n={len(arr_with):4d}  WR={wr_with:.1%}  Avg={avg_with:.3f}pts  Sharpe={sr_with:.3f}")
    print(f"     Without filter: n={len(arr_without):4d}  WR={wr_without:.1%}  Avg={avg_without:.3f}pts  Sharpe={sr_without:.3f}")
    print(f"     t-test p-value: {p_val:.4f}")
    print(f"     Verdict: {verdict}\n")

print(f"  ⏱️ Section 4 completed in {time.time()-t_sec4:.1f}s")

# =============================================================================
# SECTION 5 — SAFE POSITION SIZING
# =============================================================================
print("\n" + "═"*70)
print("  SECTION 5 — SAFE POSITION SIZING")
print("═"*70)
t_sec5 = time.time()

# 1-contract worst day
worst_day_1ct = daily_combined.min() / DEFAULT_CONTRACTS   # rescale to 1 contract
print(f"\n  Worst historical day at 1 contract: ${worst_day_1ct:,.2f}")

# Max contracts to stay under daily drawdown limits
max_ct_2k = int(abs(MAX_DD / worst_day_1ct)) if worst_day_1ct < 0 else 99
max_ct_1k = int(abs(DAILY_LIMIT / worst_day_1ct)) if worst_day_1ct < 0 else 99
max_ct_2k = max(1, max_ct_2k)
max_ct_1k = max(1, max_ct_1k)

print(f"  Max contracts keeping worst day under $2,000 DD:  {max_ct_2k} contracts")
print(f"  Max contracts keeping worst day under $1,000 DD:  {max_ct_1k} contracts "
      f"(Topstep daily limit)")

# Prop firm simulation helper
def simulate_prop_firm(daily_pnl_1ct, contracts, n_sims=N_SIMS,
                       target=PROFIT_TARGET, max_dd=MAX_DD,
                       daily_limit=DAILY_LIMIT, days=CHALLENGE_DAYS):
    """Monte Carlo prop firm challenge simulation."""
    pnl_arr = daily_pnl_1ct * contracts
    rng = np.random.default_rng(42)
    passed = 0
    for _ in range(n_sims):
        sample = rng.choice(pnl_arr, size=days, replace=True)
        cum_pnl = np.cumsum(sample)
        # track drawdown from peak
        peak = np.maximum.accumulate(cum_pnl)
        dd   = (cum_pnl - peak).min()
        # check daily limit
        daily_ok = (sample >= -daily_limit).all()
        if cum_pnl[-1] >= target and dd >= -max_dd and daily_ok:
            passed += 1
    return passed / n_sims

# Use 1-contract P&L aligned to all trade days (average over windows per day)
pnl_1ct = daily_combined.values / DEFAULT_CONTRACTS

print("\n  🎲 Prop Firm Simulation ($50K, $3K target, $2K max DD, $1K daily limit)")
print(f"  {'Sizing':<25} {'Contracts':>10} {'Pass Rate':>12}")
print("  " + "-"*48)
for label, cts in [
    ("Aggressive (7ct)",         DEFAULT_CONTRACTS),
    (f"Safe ($2K limit, {max_ct_2k}ct)", max_ct_2k),
    (f"Safe ($1K limit, {max_ct_1k}ct)", max_ct_1k),
    ("Conservative (3ct)",       3),
]:
    rate = simulate_prop_firm(pnl_1ct, cts)
    marker = "🟢" if rate >= 0.70 else ("🟡" if rate >= 0.50 else "🔴")
    print(f"  {label:<25} {cts:>10}  {rate:>10.1%} {marker}")

print(f"\n  ⏱️ Section 5 completed in {time.time()-t_sec5:.1f}s")

# =============================================================================
# SECTION 6 — PERMUTATION / RANDOMNESS TEST
# =============================================================================
print("\n" + "═"*70)
print("  SECTION 6 — PERMUTATION / RANDOMNESS TEST")
print("═"*70)
t_sec6 = time.time()

def combined_sharpe_for_windows(df, windows):
    """Combined Sharpe of a list of (entry_h, entry_m, exit_h, exit_m) windows."""
    all_pnl = []
    for w in windows:
        t = get_trades_for_window(df, *w)
        all_pnl.extend(t["pnl_pts"].tolist())
    if len(all_pnl) < 20:
        return np.nan
    arr = np.array(all_pnl) * POINT_VALUE
    return (arr.mean() / arr.std(ddof=1)) * np.sqrt(252)

# Compute actual Sharpe of user's 5 windows
actual_sharpe = combined_sharpe_for_windows(df_all, USER_WINDOWS)
print(f"\n  User's 5-window combined Sharpe: {actual_sharpe:.4f}")

# Pre-compute trades for ALL possible windows (for fast permutation lookup)
print("  ⚙️ Pre-computing trades for all possible windows...")
t_cache = time.time()
window_pnl_cache = {}
all_possible_windows = []
for eh in range(9, 16):
    for em in range(0, 60):
        for hold in range(5, 16):
            total_m = eh * 60 + em + hold
            xh, xm  = total_m // 60, total_m % 60
            if xh > 15 or (xh == 15 and xm > 59):
                continue
            key = (eh, em, xh, xm)
            all_possible_windows.append(key)
            t = get_trades_for_window(df_all, *key)
            if len(t) >= 20:
                window_pnl_cache[key] = t["pnl_pts"].values * POINT_VALUE

cached_keys = list(window_pnl_cache.keys())
n_cached = len(cached_keys)
print(f"  ✅ Pre-computed {n_cached} valid windows (of {len(all_possible_windows)} total) "
      f"in {time.time()-t_cache:.1f}s")

# Random sampling using pre-computed cache
print(f"  Running {N_SIMS:,} random selections of 5 windows each...")
rng_perm = np.random.default_rng(0)
random_sharpes = []
for i in range(N_SIMS):
    idx = rng_perm.choice(n_cached, size=5, replace=False)
    all_pnl = np.concatenate([window_pnl_cache[cached_keys[j]] for j in idx])
    if len(all_pnl) >= 20:
        sh = (all_pnl.mean() / all_pnl.std(ddof=1)) * np.sqrt(252)
        random_sharpes.append(sh)
    if (i + 1) % 2500 == 0:
        print(f"    ... {i+1:,}/{N_SIMS:,} ({(i+1)/N_SIMS*100:.0f}%)")

random_sharpes = np.array(random_sharpes)
p_value = (random_sharpes >= actual_sharpe).mean()

print(f"\n  Random distribution stats:")
print(f"    Mean   : {random_sharpes.mean():.4f}")
print(f"    Std    : {random_sharpes.std():.4f}")
print(f"    P95    : {np.percentile(random_sharpes, 95):.4f}")
print(f"    P99    : {np.percentile(random_sharpes, 99):.4f}")
print(f"\n  User's Sharpe {actual_sharpe:.4f} vs random mean {random_sharpes.mean():.4f}")
print(f"  p-value (fraction of random selections that beat yours): {p_value:.4f}")

if p_value < 0.01:
    perm_verdict = "🟢 p < 0.01 — Your edge is VERY UNLIKELY to be random luck (top 1%)"
elif p_value < 0.05:
    perm_verdict = "🟢 p < 0.05 — Your edge is statistically unlikely to be random luck"
elif p_value < 0.10:
    perm_verdict = "🟡 p < 0.10 — Marginal significance; edge could partially be luck"
else:
    perm_verdict = "🔴 p ≥ 0.10 — Warning: random window selection could produce similar results"
print(f"\n  {perm_verdict}")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(random_sharpes, bins=60, color="lightblue", edgecolor="none",
        label="Random 5-window combos")
ax.axvline(actual_sharpe, color="red", linewidth=2.5,
           label=f"Your strategy Sharpe = {actual_sharpe:.3f}")
ax.axvline(np.percentile(random_sharpes, 95), color="orange", linestyle="--",
           linewidth=1.5, label="95th percentile")
ax.set_title("Permutation Test: Your Strategy vs. 10,000 Random Window Selections")
ax.set_xlabel("Combined Sharpe Ratio")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/permutation_test.png", dpi=150)
plt.show()
print("  📊 Chart saved → outputs/permutation_test.png")
print(f"  ⏱️ Section 6 completed in {time.time()-t_sec6:.1f}s")

# =============================================================================
# SECTION 7 — FINAL SUMMARY & RECOMMENDATION
# =============================================================================
print("\n" + "═"*70)
print("  SECTION 7 — FINAL SUMMARY & RECOMMENDATION")
print("═"*70)

# Gather OOS sharpe for user windows
oos_sharpes_user = []
for label, trades_all in all_user_trades.items():
    trades_oos = trades_all[trades_all["date"].isin(test_days)]
    stats_oos  = window_stats(trades_oos)
    if stats_oos:
        oos_sharpes_user.append(stats_oos["sharpe"])

avg_oos_sharpe    = np.mean(oos_sharpes_user) if oos_sharpes_user else 0.0
avg_train_sharpe  = cand_df.head(5)["sharpe"].mean()

# Decision logic
score = 0
checks = []

# 1. OOS edge persists?
if avg_oos_sharpe > 0.5:
    score += 2
    checks.append("🟢 OOS Sharpe positive and > 0.5 (edge persists)")
elif avg_oos_sharpe > 0:
    score += 1
    checks.append("🟡 OOS Sharpe positive but weak (< 0.5)")
else:
    checks.append("🔴 OOS Sharpe negative — edge did NOT survive OOS test")

# 2. Permutation test
if p_value < 0.05:
    score += 2
    checks.append(f"🟢 Permutation test p={p_value:.3f} < 0.05 — edge unlikely to be luck")
elif p_value < 0.10:
    score += 1
    checks.append(f"🟡 Permutation test p={p_value:.3f} — marginal significance")
else:
    checks.append(f"🔴 Permutation test p={p_value:.3f} — cannot rule out luck")

# 3. Correlation risk
if mean_corr < 0.3:
    score += 1
    checks.append(f"🟢 Low inter-window correlation ({mean_corr:.2f}) — good diversification")
elif mean_corr < 0.6:
    score += 0
    checks.append(f"🟡 Moderate inter-window correlation ({mean_corr:.2f}) — be aware of clustering")
else:
    score -= 1
    checks.append(f"🔴 High inter-window correlation ({mean_corr:.2f}) — losses cluster!")

# 4. Safe sizing
if max_ct_2k >= 5:
    score += 1
    checks.append(f"🟢 Safe contract count ({max_ct_2k}ct) adequate for $2K DD limit")
elif max_ct_2k >= 3:
    checks.append(f"🟡 Safe contract count ({max_ct_2k}ct) is conservative")
else:
    score -= 1
    checks.append(f"🔴 Safe contract count ({max_ct_2k}ct) very low — worst day is severe")

# 5. Worst consecutive days
if max_streak <= 3:
    score += 1
    checks.append(f"🟢 Max consecutive losing days: {max_streak} — manageable")
elif max_streak <= 5:
    checks.append(f"🟡 Max consecutive losing days: {max_streak} — watch drawdown")
else:
    score -= 1
    checks.append(f"🔴 Max consecutive losing days: {max_streak} — significant drawdown risk")

print("\n  DETAILED ASSESSMENT:")
for c in checks:
    print(f"    {c}")

print(f"\n  IS Sharpe (train): {avg_train_sharpe:.3f}  →  OOS Sharpe (test): {avg_oos_sharpe:.3f}")
print(f"  Permutation p-value: {p_value:.4f}")
print(f"  Max safe contracts ($2K DD): {max_ct_2k}")
print(f"  Max safe contracts ($1K daily): {max_ct_1k}")
print(f"  Max consecutive losing days: {max_streak}")
print(f"  Mean window correlation: {mean_corr:.3f}")

print(f"\n  OVERALL SCORE: {score}/7")
print()
if score >= 5:
    recommendation = "🟢 GO — Evidence supports paper trading, then live"
    detail = (f"  Start with {max_ct_1k} contracts (Topstep $1K daily limit) or "
              f"{max_ct_2k} contracts (generic $2K max DD).\n"
              f"  Use ALL 5 windows: " + ", ".join(all_user_trades.keys()) + "\n"
              f"  Paper trade for at least 1 month before going live.")
elif score >= 3:
    recommendation = "🟡 CAUTION — Mixed signals, paper trade only"
    detail = (f"  Strategy shows some edge but has weaknesses.\n"
              f"  Paper trade ONLY with {max_ct_1k} contracts.\n"
              f"  Re-evaluate after 30 paper trading days.\n"
              f"  Do NOT risk real money until OOS metrics improve.")
else:
    recommendation = "🔴 NO-GO — Do not trade this strategy yet"
    detail = (f"  Strategy did not survive out-of-sample validation.\n"
              f"  The in-sample results may be overfitted.\n"
              f"  Do NOT trade, even on paper, until significant re-validation.")

print("  " + "─"*60)
print(f"  RECOMMENDATION: {recommendation}")
print("  " + "─"*60)
print(detail)
print()

print("═"*70)
print("  v5_honest_validation.py complete — all results from YOUR data.")
print("  Charts saved in outputs/")
total_elapsed = time.time() - t_total_start
minutes = int(total_elapsed // 60)
seconds = int(total_elapsed % 60)
print(f"  ⏱️ Total runtime: {minutes}m {seconds}s")
print("═"*70)
