"""
run_all.py — Single Entry Point for MNQ Multi-Window Strategy System

Usage:
    python run_all.py --csv path/to/data.csv [--module all] [--capital 10000] [--point-value 2]

Modules:
    core        — run core backtester
    validation  — run advanced validation
    prediction  — run volatility predictor
    propfirm    — run prop firm optimizer
    all         — run all modules (default)
"""

import argparse
import os
import sys
import time

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="MNQ Multi-Window Strategy Backtesting & Prop Firm Optimizer"
    )
    parser.add_argument(
        "--csv", type=str, default=config.CSV_PATH,
        help="Path to 1-minute bar CSV file"
    )
    parser.add_argument(
        "--module", type=str, default="all",
        choices=["core", "validation", "prediction", "propfirm", "all"],
        help="Which module to run (default: all)"
    )
    parser.add_argument(
        "--capital", type=float, default=config.INITIAL_CAPITAL,
        help="Initial capital in dollars (default: 10000)"
    )
    parser.add_argument(
        "--point-value", type=float, default=config.DEFAULT_POINT_VALUE,
        dest="point_value",
        help="Dollar value per point (MNQ=2, NQ=20; default: 2)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        dest="output_dir",
        help="Directory for output charts and CSVs (default: output)"
    )
    return parser.parse_args()


def run_core(csv_path: str, capital: float, point_value: float, output_dir: str):
    import core_strategy as cs
    print("\n" + "=" * 60)
    print("  MODULE: Core Strategy Backtester")
    print("=" * 60)
    t0 = time.time()
    results, trades, df = cs.run(
        csv_path,
        initial_capital=capital,
        point_value=point_value,
        output_dir=output_dir,
    )
    print(f"\n  ✓ Core module completed in {time.time()-t0:.1f}s")
    return results, trades, df


def run_validation(trades_df, capital: float, output_dir: str):
    import numpy as np
    import advanced_validation as av

    print("\n" + "=" * 60)
    print("  MODULE: Advanced Validation")
    print("=" * 60)
    t0 = time.time()

    # Use "both" direction P&L
    filtered = trades_df[~trades_df["skip"]].copy()
    filtered["pnl"] = np.where(
        filtered["direction"] == "LONG",
        filtered["pnl_long_pts"] * config.DEFAULT_POINT_VALUE,
        filtered["pnl_short_pts"] * config.DEFAULT_POINT_VALUE,
    )
    av.run(filtered, pnl_col="pnl", initial_capital=capital, output_dir=output_dir)
    print(f"\n  ✓ Validation module completed in {time.time()-t0:.1f}s")


def run_prediction(df_1min, trades_df, output_dir: str):
    import volatility_predictor as vp

    print("\n" + "=" * 60)
    print("  MODULE: Volatility Predictor")
    print("=" * 60)
    t0 = time.time()
    vp.run(df_1min, trades_df, output_dir=output_dir)
    print(f"\n  ✓ Prediction module completed in {time.time()-t0:.1f}s")


def run_propfirm(trades_df, output_dir: str):
    import propfirm_optimizer as pfo

    print("\n" + "=" * 60)
    print("  MODULE: Prop Firm Optimizer")
    print("=" * 60)
    t0 = time.time()
    pfo.run(trades_df, output_dir=output_dir)
    print(f"\n  ✓ Prop firm module completed in {time.time()-t0:.1f}s")


def main():
    args = parse_args()

    # Validate CSV path
    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        print("Usage: python run_all.py --csv path/to/data.csv")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  MNQ MULTI-WINDOW STRATEGY SYSTEM")
    print("=" * 60)
    print(f"  CSV:         {args.csv}")
    print(f"  Capital:     ${args.capital:,.0f}")
    print(f"  Point Value: ${args.point_value}/pt")
    print(f"  Module:      {args.module}")
    print(f"  Output dir:  {args.output_dir}")

    t_start = time.time()

    # Always need core data
    results = trades = df = None
    if args.module in ("core", "all"):
        results, trades, df = run_core(args.csv, args.capital, args.point_value, args.output_dir)
    else:
        # Load data only
        import core_strategy as cs
        print("\nLoading data for non-core module …")
        df     = cs.load_data(args.csv)
        trades = cs.extract_trades(df)

    if args.module in ("validation", "all"):
        run_validation(trades, args.capital, args.output_dir)

    if args.module in ("prediction", "all"):
        run_prediction(df, trades, args.output_dir)

    if args.module in ("propfirm", "all"):
        run_propfirm(trades, args.output_dir)

    print("\n" + "=" * 60)
    print(f"  ALL DONE — total time: {time.time()-t_start:.1f}s")
    print(f"  Output saved to: {os.path.abspath(args.output_dir)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
