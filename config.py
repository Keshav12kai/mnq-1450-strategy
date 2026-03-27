"""
config.py — Central Configuration for MNQ 14:50 Strategy

RULE SET EXPLANATION
--------------------
Pass rate changes significantly with different prop firm rule sets:

  v2-reference (original ~86.7% result):
    - challenge_days = 10   (short challenge window → fewer chances to fail)
    - max_drawdown   = $2,000
    - daily_loss_limit = $1,000
    - Topstep $50K style

  30-day challenge (stricter):
    - challenge_days = 30   (3× more days → ~3× more chances to hit DD/daily limits)
    - max_drawdown   = $2,000
    - daily_loss_limit = $1,000
    - Pass rate drops to ~60-65% even with the same position-sizing model

  Apex $50K style ($2.5k DD):
    - max_drawdown   = $2,500  (+$500 headroom → easier to survive)
    - challenge_days = 7
    - daily_loss_limit = $1,000

Key levers (in order of impact):
  1. challenge_days  — largest effect; doubling days roughly halves pass rate
  2. max_drawdown    — $500 extra headroom can add 5-10 pp to pass rate
  3. daily_loss_limit — tight limits cause sudden fails on volatile days
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
ENTRY_TIME  = "14:50"          # enter at close of 14:50 candle
EXIT_TIME   = "14:59"          # exit at close of 14:59 candle

# ── Smart Filters ────────────────────────────────────────────────────────────
SKIP_THURSDAYS     = True
SKIP_MONTHS        = [6, 10]   # June and October
MIN_CANDLE_BODY    = 3.0       # minimum candle body in points

# ── Position Sizing ──────────────────────────────────────────────────────────
DEFAULT_CONTRACTS  = 1
MAX_CONTRACTS      = 50
MIN_CONTRACTS      = 1

# ── Explicit Daily Loss Limit ────────────────────────────────────────────────
# Default daily loss limit used when running outside a specific prop firm context
# (e.g. when computing position sizes independently of the PROP_FIRMS list).
# This is NOT automatically applied to entries in PROP_FIRMS; each firm entry
# carries its own daily_loss_limit.  Set this to match the firm you are targeting.
DAILY_LOSS_LIMIT   = 1_000     # dollars

# ── DD Reduction Rules ───────────────────────────────────────────────────────
DD_LEVELS = [
    (0.03, 0.50),   # at  3% drawdown: cut contracts by 50%
    (0.06, 0.75),   # at  6% drawdown: cut contracts by 75%
    (0.08, 1.00),   # at  8% drawdown: exit entirely
]

# ── Volatility Predictor ─────────────────────────────────────────────────────
SAFETY_BUFFER_DEFAULT = 0.60   # 60% = winning configuration (v2-reference result)

# ── Prop Firm Configurations ─────────────────────────────────────────────────
# Each entry: name, account_size, profit_target, max_drawdown, daily_loss_limit,
#             challenge_days, challenge_fee
#
# NOTE: challenge_days is the single biggest driver of pass-rate variance.
#   10-day challenge  → ~86-87% pass rate with Ensemble 60% buffer
#   30-day challenge  → ~60-65% pass rate with the same model (3× more days)
PROP_FIRMS = [
    # ── User-specified target (3k profit, 2k max DD, 30-day, MNQ) ────────────
    {"name": "Custom 30-day $50K", "account_size": 50_000, "profit_target": 3_000,
     "max_drawdown": 2_000, "daily_loss_limit": 1_000, "challenge_days": 30, "fee": 165},

    # ── v2-reference firm (10-day → original ~86.7% result) ──────────────────
    {"name": "v2-ref 10-day $50K", "account_size": 50_000, "profit_target": 3_000,
     "max_drawdown": 2_000, "daily_loss_limit": 1_000, "challenge_days": 10, "fee": 165},

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
