"""
config.py — Central Configuration for MNQ Multi-Window Strategy
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

# ── Trade Windows ────────────────────────────────────────────────────────────
# All times Eastern Time (ET).  Each window: entry at candle close, exit at candle close.
TRADE_WINDOWS = [
    {
        "name":               "Window 1 — Mid-session rebalancing",
        "entry_time":         "12:16",
        "exit_time":          "12:27",
        "hold_minutes":       11,
        "conviction":         "medium",
        "structural_reason":  "Mid-session rebalancing",
    },
    {
        "name":               "Window 2 — Lunch-hour order flow",
        "entry_time":         "12:51",
        "exit_time":          "13:00",
        "hold_minutes":       9,
        "conviction":         "medium",
        "structural_reason":  "Lunch-hour order flow",
    },
    {
        "name":               "Window 3 — Early pre-close positioning",
        "entry_time":         "13:55",
        "exit_time":          "14:05",
        "hold_minutes":       10,
        "conviction":         "medium",
        "structural_reason":  "Early pre-close positioning",
    },
    {
        "name":               "Window 4 — Pre-close institutional re-balance",
        "entry_time":         "14:50",
        "exit_time":          "15:01",
        "hold_minutes":       11,
        "conviction":         "high",
        "structural_reason":  "Pre-close institutional re-balance",
    },
    {
        "name":               "Window 5 — Closing auction burst",
        "entry_time":         "15:48",
        "exit_time":          "15:58",
        "hold_minutes":       10,
        "conviction":         "high",
        "structural_reason":  "Closing auction burst",
    },
]

# ── Legacy aliases (point to Window 4 for backward compatibility) ─────────────
ENTRY_TIME  = "14:50"          # enter at close of 14:50 candle
EXIT_TIME   = "15:01"          # exit at close of 15:01 candle

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

# ── Chart Output ─────────────────────────────────────────────────────────────
CHART_DPI = 150
