"""
config.py — Central Configuration for MNQ 14:50 Strategy
"""

# ── CSV Data ────────────────────────────────────────────────────────────────
CSV_PATH = "data.csv"          # override with --csv flag

# ── Capital & Instrument ─────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000       # dollars
POINT_VALUE = {
    "MNQ": 2.0,                # $2 per point
    "NQ":  20.0,               # $20 per point
}
DEFAULT_INSTRUMENT = "MNQ"
DEFAULT_POINT_VALUE = POINT_VALUE[DEFAULT_INSTRUMENT]

# ── Strategy Timing ──────────────────────────────────────────────────────────
ENTRY_TIME        = "14:50"    # single-window mode: enter at close of this candle
EXIT_TIME         = "14:59"    # single-window mode: exit at close of this candle
ENTRY_HOLD_MINUTES = 9         # hold duration used across all windows

# ── Multi-Window Configuration ───────────────────────────────────────────────
# Each tuple: (entry_hour, entry_minute, exit_hour, exit_minute, label)
#
# ⚠ IMPORTANT: These 5 windows are ILLUSTRATIVE EXAMPLES only.
#   They have NOT been validated against real market data — no data.csv file
#   is included in this repository.  The 14:50 window is the original strategy
#   hypothesis; the other 4 are plausible structural candidates listed for
#   reference.
#
#   You MUST run `python run_all.py --csv your_data.csv --module bias` with
#   your own 1-minute MNQ CSV to discover which windows are actually robust
#   in your data.
#
# ⚠ Timezone note: timestamps are EASTERN TIME (ET / New York).
#   In TradingView/Pine Script, verify your chart is set to "America/New_York"
#   so that 14:50 ET appears as 14:50 on the chart, NOT 13:50 (CT) or 15:50.
MULTI_WINDOWS = [
    (12, 16, 12, 25, "12:16→12:25"),   # illustrative — validate with your data
    (12, 51, 13,  0, "12:51→13:00"),   # illustrative — validate with your data
    (13, 55, 14,  4, "13:55→14:04"),   # illustrative — validate with your data
    (14, 50, 14, 59, "14:50→14:59"),   # original hypothesis (pre-close positioning)
    (15, 48, 15, 57, "15:48→15:57"),   # illustrative — closing auction candidate
]

# ── Smart Filters ────────────────────────────────────────────────────────────
SKIP_THURSDAYS     = True
SKIP_MONTHS        = [6, 10]   # June and October
MIN_CANDLE_BODY    = 3.0       # minimum candle body in points

# ── Position Sizing ──────────────────────────────────────────────────────────
DEFAULT_CONTRACTS  = 1
MAX_CONTRACTS      = 50
MIN_CONTRACTS      = 1

# ── DD Reduction Rules ───────────────────────────────────────────────────────
DD_LEVELS = [
    (0.03, 0.50),   # at  3% drawdown: cut contracts by 50%
    (0.06, 0.75),   # at  6% drawdown: cut contracts by 75%
    (0.08, 1.00),   # at  8% drawdown: exit entirely
]

# ── Volatility Predictor ─────────────────────────────────────────────────────
SAFETY_BUFFER_DEFAULT = 0.60   # 60% = winning configuration

# ── Prop Firm Configurations ─────────────────────────────────────────────────
# Each entry: name, account_size, profit_target, max_drawdown, daily_loss_limit,
#             challenge_days, challenge_fee
PROP_FIRMS = [
    # Topstep
    {"name": "Topstep $50K",   "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_000,  "daily_loss_limit": 1_000, "challenge_days": 10, "fee": 165},
    {"name": "Topstep $100K",  "account_size": 100_000, "profit_target": 6_000,
     "max_drawdown": 3_000,  "daily_loss_limit": 2_000, "challenge_days": 10, "fee": 325},
    {"name": "Topstep $150K",  "account_size": 150_000, "profit_target": 9_000,
     "max_drawdown": 4_500,  "daily_loss_limit": 3_000, "challenge_days": 10, "fee": 375},

    # Apex
    {"name": "Apex $25K",      "account_size": 25_000,  "profit_target": 1_500,
     "max_drawdown": 1_500,  "daily_loss_limit": 500,   "challenge_days": 7,  "fee": 167},
    {"name": "Apex $50K",      "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_500,  "daily_loss_limit": 1_000, "challenge_days": 7,  "fee": 187},
    {"name": "Apex $75K",      "account_size": 75_000,  "profit_target": 4_250,
     "max_drawdown": 2_750,  "daily_loss_limit": 1_250, "challenge_days": 7,  "fee": 207},
    {"name": "Apex $100K",     "account_size": 100_000, "profit_target": 6_000,
     "max_drawdown": 3_000,  "daily_loss_limit": 2_000, "challenge_days": 7,  "fee": 247},
    {"name": "Apex $150K",     "account_size": 150_000, "profit_target": 8_000,
     "max_drawdown": 5_000,  "daily_loss_limit": 3_000, "challenge_days": 7,  "fee": 297},
    {"name": "Apex $250K",     "account_size": 250_000, "profit_target": 12_500,
     "max_drawdown": 6_500,  "daily_loss_limit": 4_500, "challenge_days": 7,  "fee": 517},
    {"name": "Apex $300K",     "account_size": 300_000, "profit_target": 20_000,
     "max_drawdown": 7_500,  "daily_loss_limit": 5_000, "challenge_days": 7,  "fee": 657},

    # MyFundedFutures
    {"name": "MFF $50K",       "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_000,  "daily_loss_limit": 1_000, "challenge_days": 10, "fee": 165},
    {"name": "MFF $100K",      "account_size": 100_000, "profit_target": 6_000,
     "max_drawdown": 3_000,  "daily_loss_limit": 2_000, "challenge_days": 10, "fee": 250},

    # TradeDay
    {"name": "TradeDay $25K",  "account_size": 25_000,  "profit_target": 1_500,
     "max_drawdown": 1_500,  "daily_loss_limit": 500,   "challenge_days": 5,  "fee": 150},
    {"name": "TradeDay $50K",  "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_000,  "daily_loss_limit": 1_000, "challenge_days": 5,  "fee": 200},
    {"name": "TradeDay $100K", "account_size": 100_000, "profit_target": 6_000,
     "max_drawdown": 3_000,  "daily_loss_limit": 2_000, "challenge_days": 5,  "fee": 300},

    # Bulenox
    {"name": "Bulenox $50K",   "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_500,  "daily_loss_limit": 1_000, "challenge_days": 10, "fee": 175},
    {"name": "Bulenox $100K",  "account_size": 100_000, "profit_target": 6_000,
     "max_drawdown": 3_500,  "daily_loss_limit": 2_000, "challenge_days": 10, "fee": 295},

    # Take Profit Trader
    {"name": "TPT $50K",       "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_000,  "daily_loss_limit": 1_000, "challenge_days": 10, "fee": 170},
    {"name": "TPT $100K",      "account_size": 100_000, "profit_target": 6_000,
     "max_drawdown": 3_000,  "daily_loss_limit": 2_000, "challenge_days": 10, "fee": 325},

    # Leeloo
    {"name": "Leeloo $50K",    "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_500,  "daily_loss_limit": 1_100, "challenge_days": 10, "fee": 150},
    {"name": "Leeloo $100K",   "account_size": 100_000, "profit_target": 5_000,
     "max_drawdown": 3_500,  "daily_loss_limit": 2_000, "challenge_days": 10, "fee": 250},

    # Elite Trader Funding
    {"name": "ETF $25K",       "account_size": 25_000,  "profit_target": 1_500,
     "max_drawdown": 1_500,  "daily_loss_limit": 500,   "challenge_days": 5,  "fee": 140},
    {"name": "ETF $50K",       "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_000,  "daily_loss_limit": 1_000, "challenge_days": 5,  "fee": 220},

    # Earn2Trade
    {"name": "E2T $25K",       "account_size": 25_000,  "profit_target": 1_500,
     "max_drawdown": 1_500,  "daily_loss_limit": 500,   "challenge_days": 10, "fee": 150},
    {"name": "E2T $50K",       "account_size": 50_000,  "profit_target": 3_000,
     "max_drawdown": 2_000,  "daily_loss_limit": 1_000, "challenge_days": 10, "fee": 245},
]

# ── Monte Carlo ──────────────────────────────────────────────────────────────
MC_RUNS        = 10_000
MC_PERCENTILES = [5, 25, 50, 75, 95]

# ── Walk-Forward ─────────────────────────────────────────────────────────────
WF_WINDOWS     = 5
WF_TRAIN_RATIO = 0.60

# ── Selection Bias Mitigation ────────────────────────────────────────────────
# Total number of 1-minute RTH windows scanned during strategy discovery.
# At 9:30–15:50 RTH with 9-min hold, there are ~381 valid entry minutes.
BIAS_N_WINDOWS_SCANNED      = 381
# Average number of trades observed per window after smart filters
BIAS_AVG_TRADES_PER_WINDOW  = 225
# Minimum Sharpe ratio used to identify "good" windows
BIAS_SHARPE_THRESHOLD       = 2.0
# Monte Carlo runs for the null-hypothesis lucky-window simulation
BIAS_NULL_SIMS              = 5_000
# Permutation runs for the multiple-comparison significance test
BIAS_PERMUTATION_RUNS       = 5_000

# ── Chart Output ─────────────────────────────────────────────────────────────
CHART_DPI = 150
