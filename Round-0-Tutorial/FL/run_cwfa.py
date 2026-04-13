"""
run_cwfa.py — entry point for running CWFA on tutorial or Round 1 data.

Usage:
    python run_cwfa.py

To adapt for Round 1:
    1. Add Round 1 price CSVs to PRICE_FILES list
    2. Adjust PARAM_GRID to include parameters for new products
    3. Run
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from simulator import load_price_data, run_backtest
from cwfa import run_cwfa, OBJECTIVES
import numpy as np

# ============================================================
# CONFIGURATION — edit this section
# ============================================================

# Price CSV files to include (add Round 1 files here when available)
PRICE_FILES = [
    "/mnt/user-data/uploads/prices_round_0_day_-2.csv",
    "/mnt/user-data/uploads/prices_round_0_day_-1.csv",
]

# Trader module path (must be importable)
TRADER_MODULE = "Trader_v6_4"
TRADER_PATH   = "/home/claude/cwfa"

# Products to optimise (None = all products in data)
PRODUCTS = ["TOMATOES"]   # EMERALDS uses fixed FV — no free parameters worth sweeping

# CWFA settings
K         = 4       # blocks (with 2 days: 4 = ~half-day blocks of ~1000 ticks each)
P         = 2       # OOS blocks per combo → C(4,2) = 6 independent OOS paths
OBJECTIVE = "sharpe"
PURGE     = True

# Parameter grid — inventory management parameters only.
#
# EXCLUDED (with rationale):
#   TOM_EMA_SPAN     — requires updating TOM_ALPHA too; already validated by FV notebook (EMA-7 optimal)
#   TOM_UNWIND_TIME  — too fine-grained; position resets within block cycle before it activates
#
# INCLUDED:
#   TOM_SKEW_TRIGGER  — strong monotonic discrimination (15→2.34 Sharpe, 40→1.62)
#   TOM_VELOCITY_THR  — threshold=2.0 is notably bad; 3.5-5.0 near-optimal
#   TOM_VOL_CALM_THRESH — modest effect on regime detection; worth checking across periods
#
# Total: 4 × 3 × 3 = 36 combos per IS window × 6 CWFA splits = 216 backtests
PARAM_GRID = {
    "TOM_SKEW_TRIGGER":     [15, 20, 25, 30],
    "TOM_VELOCITY_THR":     [2.5, 3.5, 5.0],
    "TOM_VOL_CALM_THRESH":  [1.0, 1.5, 2.0],
}

# ============================================================
# VALIDATION — run backtester on full data and compare to log
# ============================================================

def validate_backtester():
    """
    Quick sanity check: run the full backtester on both days and
    compare final PnL to the known submission result (~1473 for Tomatoes).
    """
    print("=" * 60)
    print("VALIDATION: full backtest vs submission log")
    print("=" * 60)

    sys.path.insert(0, TRADER_PATH)
    import importlib
    trader_mod = importlib.import_module(TRADER_MODULE)
    trader = trader_mod.Trader()

    data = load_price_data(PRICE_FILES)
    result = run_backtest(trader, data, products=["TOMATOES", "EMERALDS"])

    print(f"  Backtester final PnL (all products): {result['final_pnl']:.2f}")
    print(f"  Known submission PnL:                ~2447")
    print(f"  Max drawdown:                        {result['max_drawdown']:.2f}")
    print(f"  Trades executed:                     {len(result['trade_log'])}")
    print()
    for prod, info in result["per_product"].items():
        print(f"  {prod}: PnL={info['final_pnl']:.2f}, "
              f"final_pos={info['final_position']}")
    print()
    ratio = result["final_pnl"] / 2447.0
    if 0.7 <= ratio <= 1.3:
        print("  ✓ Backtester within 30% of submission — fill model is reasonable")
    else:
        print(f"  ⚠ PnL ratio = {ratio:.2f} — fill model may differ from exchange")
    print()
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    # Step 1: Validate backtester
    validate_backtester()

    # Step 2: Load data
    print("=" * 60)
    print("CWFA RUN")
    print("=" * 60)
    data = load_price_data(PRICE_FILES)
    print(f"Loaded {len(data)} rows, products: {data['product'].unique().tolist()}")
    print(f"Days: {sorted(data['day'].unique().tolist())}")
    print()

    # Step 3: Import trader class
    sys.path.insert(0, TRADER_PATH)
    import importlib
    trader_mod = importlib.import_module(TRADER_MODULE)
    trader_class = trader_mod.Trader

    # Step 4: Run CWFA
    result = run_cwfa(
        trader_class = trader_class,
        price_data   = data,
        param_grid   = PARAM_GRID,
        K            = K,
        p            = P,
        objective    = OBJECTIVE,
        products     = PRODUCTS,
        purge        = PURGE,
        verbose      = True,
    )

    # Step 5: Print full results table
    print("\nFull results table:")
    print(f"{'Combo':>6} {'OOS blocks':>12} {'IS blocks':>12} "
          f"{'IS score':>10} {'OOS score':>10}  Best params")
    print("-" * 90)
    for r in result.combo_results:
        print(f"{r.combo_id:>6} {str(r.oos_blocks):>12} {str(r.is_blocks):>12} "
              f"{r.is_score:>10.4f} {r.oos_score:>10.4f}  {r.best_params}")

    print(f"\nOOS scores: {result.oos_scores}")
    print(f"Distribution: mean={result.oos_mean:.4f}, "
          f"std={result.oos_std:.4f}, "
          f"min={result.oos_scores.min():.4f}, "
          f"max={result.oos_scores.max():.4f}")

    print(f"\n{'='*60}")
    print(f"RECOMMENDED PARAMETERS: {result.recommended_params}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    main()
