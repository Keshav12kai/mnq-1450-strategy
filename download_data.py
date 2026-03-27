"""
download_data.py — Download 1-minute MNQ/NQ futures data via yfinance

This script fetches historical 1-minute bar data for MNQ (Micro Nasdaq) or
NQ (Nasdaq) futures from Yahoo Finance and saves it in the exact CSV format
required by this strategy system.

IMPORTANT — Yahoo Finance limitation:
  1-minute bars are only available for the most recent 7 calendar days.
  For a full multi-year backtest you need to export data from your own
  brokerage/platform (NinjaTrader, Sierra Chart, Tradovate, etc.) and
  place the file in data/data.csv  (or data/data.csv.gz if compressed).

Usage:
    # Download recent data for MNQ (default)
    python download_data.py

    # Download NQ instead of MNQ
    python download_data.py --ticker NQ=F

    # Save to a custom path
    python download_data.py --output data/mnq_recent.csv

    # Extend an existing CSV with new data
    python download_data.py --append

    # Compress the output to save disk space (~10× smaller)
    python download_data.py --compress
"""

import argparse
import os
import sys

import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_TICKER     = "MNQ=F"
DEFAULT_OUTPUT     = "data/data.csv"
EXPECTED_COLUMNS   = ["timestamp ET", "open", "high", "low", "close", "volume"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _require_yfinance():
    try:
        import yfinance as yf   # noqa: F401
    except ImportError:
        print("ERROR: yfinance is not installed.")
        print("Run:  pip install yfinance")
        sys.exit(1)


def download(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    """
    Download up to 7 days of 1-minute bars from Yahoo Finance and return
    a DataFrame formatted for this system.

    Columns returned: timestamp ET, open, high, low, close, volume
    """
    import yfinance as yf

    print(f"Downloading 1-minute bars for {ticker} (last 7 days) …")
    raw = yf.download(
        ticker,
        period="7d",
        interval="1m",
        prepost=False,        # regular trading hours only
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        print(f"ERROR: No data returned for {ticker}.")
        print("Possible causes:")
        print("  • No internet connection")
        print("  • The ticker symbol is incorrect")
        print("  • Yahoo Finance is temporarily unavailable")
        sys.exit(1)

    # Flatten MultiIndex columns if present (yfinance ≥ 0.2 may produce them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Ensure the index has an Eastern-Time timezone
    idx = raw.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = idx

    # Format timestamp to MM/DD/YYYY HH:MM (Eastern)
    df.insert(0, "timestamp ET", df.index.strftime("%m/%d/%Y %H:%M"))

    df = df.rename(columns={
        "Open":   "open",
        "High":   "high",
        "Low":    "low",
        "Close":  "close",
        "Volume": "volume",
    })
    df = df.reset_index(drop=True)

    # Keep only the columns the system expects
    df = df[EXPECTED_COLUMNS]

    if not df.empty:
        print(f"  Downloaded {len(df):,} rows  "
              f"({df['timestamp ET'].iloc[0]}  →  {df['timestamp ET'].iloc[-1]})")
    else:
        print("  WARNING: Download returned 0 rows after processing.")
    return df


def save(df: pd.DataFrame, path: str, compress: bool = False,
         append: bool = False) -> str:
    """Save df to *path*.  Returns the final path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if compress and not path.endswith(".gz"):
        path += ".gz"

    if append and os.path.isfile(path):
        existing = pd.read_csv(path)
        existing.columns = [c.strip() for c in existing.columns]
        ts_col = next((c for c in existing.columns if "timestamp" in c.lower()), None)
        if ts_col is None:
            print("  WARNING: Existing file has no 'timestamp' column; overwriting instead of appending.")
        else:
            existing = existing.rename(columns={ts_col: "timestamp ET"})
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp ET"])
            combined = combined.sort_values("timestamp ET").reset_index(drop=True)
            df = combined
            print(f"  Merged with existing file → {len(df):,} rows total")

    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  Saved to {os.path.abspath(path)}  ({size_mb:.1f} MB)")
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download 1-min MNQ/NQ futures data via yfinance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ticker",   default=DEFAULT_TICKER,
                   help=f"Yahoo Finance ticker (default: {DEFAULT_TICKER})")
    p.add_argument("--output",   default=DEFAULT_OUTPUT,
                   help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--compress", action="store_true",
                   help="Save as gzip-compressed .csv.gz (~10× smaller)")
    p.add_argument("--append",   action="store_true",
                   help="Append to existing file instead of overwriting")
    return p.parse_args()


def main():
    args = parse_args()
    _require_yfinance()
    df = download(ticker=args.ticker)
    final_path = save(df, args.output, compress=args.compress, append=args.append)
    print()
    print("Done!  You can now run:")
    print(f"  python run_all.py --csv {final_path}")


if __name__ == "__main__":
    main()
