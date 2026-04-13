"""
cwfa.py — Combinatorial Walk-Forward Analysis harness.

Usage pattern:
  1. Define a parameter grid (dict of param -> list of values)
  2. Call run_cwfa(trader_class, price_data, param_grid, ...)
  3. Inspect results: OOS distribution, parameter stability, best params

Theory recap:
  - Split data into K equal blocks (by tick count, respecting day boundaries)
  - For each combination of (K choose p) OOS blocks:
      - IS  = remaining K-p blocks, MINUS any block adjacent to an OOS block (purge)
      - Optimise parameters on IS (maximise objective)
      - Evaluate best params on OOS
  - Collect all OOS scores → distribution
  - Best parameters = those that appear most often in IS-optimal sets AND
    produce consistent OOS performance
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import json
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from simulator import run_backtest, load_price_data


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ComboResult:
    combo_id:    int
    oos_blocks:  Tuple
    is_blocks:   Tuple
    best_params: dict
    is_score:    float
    oos_score:   float


@dataclass
class CWFAResult:
    combo_results:      List[ComboResult]
    oos_scores:         np.ndarray
    oos_mean:           float
    oos_std:            float
    oos_sharpe:         float       # mean / std (proxy)
    best_params_counts: dict        # param combo -> count of times IS-optimal
    recommended_params: dict        # consensus best params
    param_stability:    dict        # per-param: how stable across IS optima


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------

def split_into_blocks(
    price_data: pd.DataFrame,
    K: int,
    product: str = None,
) -> List[pd.DataFrame]:
    """
    Split price_data into K roughly equal blocks.
    Respects day boundaries: blocks are cut at the closest day/timestamp
    boundary to the K-even split point.

    Returns list of K DataFrames.
    """
    # Use a single reference product for tick counting to avoid
    # double-counting when multiple products share a timestamp
    if product is None:
        product = price_data["product"].iloc[0]

    ref = price_data[price_data["product"] == product].copy()
    ref = ref.sort_values(["day", "timestamp"]).reset_index(drop=True)

    n = len(ref)
    block_size = n // K

    # Find block boundaries (indices into ref)
    boundaries = [i * block_size for i in range(K)] + [n]

    blocks = []
    for i in range(K):
        start_idx = boundaries[i]
        end_idx   = boundaries[i + 1]

        # Get the (day, timestamp) range for this block
        start_row = ref.iloc[start_idx]
        end_row   = ref.iloc[min(end_idx, n - 1)]

        # Filter full price_data (all products) for this range
        mask = (
            (price_data["day"] > start_row["day"]) |
            ((price_data["day"] == start_row["day"]) &
             (price_data["timestamp"] >= start_row["timestamp"]))
        ) & (
            (price_data["day"] < end_row["day"]) |
            ((price_data["day"] == end_row["day"]) &
             (price_data["timestamp"] <= end_row["timestamp"]))
        )
        blocks.append(price_data[mask].copy())

    return blocks


def purge_adjacent(
    all_blocks:  List[pd.DataFrame],
    oos_indices: Tuple[int, ...],
) -> List[pd.DataFrame]:
    """
    Remove blocks that are directly adjacent to any OOS block from IS.
    Returns list of IS DataFrames (may be smaller than K - p).
    """
    oos_set = set(oos_indices)
    adjacent = set()
    for idx in oos_indices:
        if idx - 1 >= 0:
            adjacent.add(idx - 1)
        if idx + 1 < len(all_blocks):
            adjacent.add(idx + 1)

    is_indices = [
        i for i in range(len(all_blocks))
        if i not in oos_set and i not in adjacent
    ]
    return [all_blocks[i] for i in is_indices], is_indices


# ---------------------------------------------------------------------------
# Parameter grid helpers
# ---------------------------------------------------------------------------

def grid_combinations(param_grid: Dict[str, List[Any]]) -> List[dict]:
    """Generate all combinations of parameters from a grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def apply_params_to_trader(trader_class, params: dict):
    """
    Return a trader instance with module-level constants patched.
    Works by monkey-patching the module where trader_class is defined.
    """
    import importlib, types

    module = sys.modules[trader_class.__module__]

    # Save originals
    originals = {}
    for k, v in params.items():
        if hasattr(module, k):
            originals[k] = getattr(module, k)
        setattr(module, k, v)

    trader = trader_class()

    # Restore (important: restore after instantiation so stateless params
    # are picked up at run() time too — need to keep them set during backtest)
    # We return the module and originals so the caller can restore later
    return trader, module, originals


def restore_params(module, originals: dict):
    for k, v in originals.items():
        setattr(module, k, v)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def sharpe_objective(result: dict) -> float:
    """Sharpe-like: mean tick-PnL / std tick-PnL. Penalises variance."""
    pnl = np.array([p for _, p in result["pnl_series"]])
    diffs = np.diff(pnl)
    if len(diffs) == 0 or diffs.std() == 0:
        return 0.0
    return float(diffs.mean() / (diffs.std() + 1e-9))


def final_pnl_objective(result: dict) -> float:
    return result["final_pnl"]


def calmar_objective(result: dict) -> float:
    """Final PnL / |max drawdown|. Returns 0 if drawdown is 0."""
    pnl = result["final_pnl"]
    dd  = abs(result["max_drawdown"])
    if dd < 1e-6:
        return pnl
    return pnl / dd


OBJECTIVES = {
    "sharpe":    sharpe_objective,
    "final_pnl": final_pnl_objective,
    "calmar":    calmar_objective,
}


# ---------------------------------------------------------------------------
# IS optimisation
# ---------------------------------------------------------------------------

def optimise_on_is(
    trader_class,
    is_data:     pd.DataFrame,
    param_grid:  Dict[str, List[Any]],
    objective:   Callable,
    products:    Optional[List[str]] = None,
    verbose:     bool = False,
) -> Tuple[dict, float]:
    """
    Try every parameter combination on IS data.
    Returns (best_params, best_score).
    """
    combos = grid_combinations(param_grid)
    if verbose:
        print(f"    IS optimisation: {len(combos)} combos over {len(is_data)} rows")

    best_score  = -np.inf
    best_params = combos[0]

    for params in combos:
        trader, module, originals = apply_params_to_trader(trader_class, params)
        try:
            # Keep params set during backtest
            for k, v in params.items():
                setattr(module, k, v)
            result = run_backtest(trader, is_data, products=products)
            score  = objective(result)
        except Exception as e:
            if verbose:
                print(f"      Error with params {params}: {e}")
            score = -np.inf
        finally:
            restore_params(module, originals)

        if score > best_score:
            best_score  = score
            best_params = params

    return best_params, best_score


# ---------------------------------------------------------------------------
# Main CWFA runner
# ---------------------------------------------------------------------------

def run_cwfa(
    trader_class,
    price_data:      pd.DataFrame,
    param_grid:      Dict[str, List[Any]],
    K:               int  = 4,
    p:               int  = 2,
    objective:       str  = "sharpe",
    products:        Optional[List[str]] = None,
    purge:           bool = True,
    verbose:         bool = True,
) -> CWFAResult:
    """
    Run Combinatorial Walk-Forward Analysis.

    Args:
        trader_class : class with a .run(TradingState) method
        price_data   : DataFrame from load_price_data()
        param_grid   : {param_name: [val1, val2, ...]}
        K            : number of blocks to split data into
        p            : number of OOS blocks per combination
        objective    : 'sharpe', 'final_pnl', or 'calmar'
        products     : list of product names (None = all)
        purge        : whether to purge adjacent blocks from IS
        verbose      : print progress

    Returns CWFAResult.
    """
    obj_fn = OBJECTIVES.get(objective, final_pnl_objective)

    if verbose:
        print(f"CWFA: K={K}, p={p}, objective={objective}")
        print(f"  Splitting data into {K} blocks...")

    blocks = split_into_blocks(price_data, K)

    # Generate all C(K, p) OOS combinations
    all_combos = list(itertools.combinations(range(K), p))
    if verbose:
        print(f"  {len(all_combos)} OOS combinations (C({K},{p}))")
        n_grid = len(grid_combinations(param_grid))
        print(f"  {n_grid} parameter combinations in grid")
        print(f"  Total backtests: {len(all_combos) * n_grid} (IS) + {len(all_combos)} (OOS)")

    combo_results: List[ComboResult] = []
    param_count: Dict[str, int] = {}

    for combo_id, oos_idx in enumerate(all_combos):
        if verbose:
            print(f"\n  Combo {combo_id+1}/{len(all_combos)}: OOS blocks {oos_idx}")

        # Build IS and OOS DataFrames
        oos_data = pd.concat([blocks[i] for i in oos_idx], ignore_index=True)

        if purge:
            is_block_list, is_indices = purge_adjacent(blocks, oos_idx)
        else:
            is_block_list = [blocks[i] for i in range(K) if i not in set(oos_idx)]
            is_indices    = [i for i in range(K) if i not in set(oos_idx)]

        if not is_block_list:
            if verbose:
                print("    No IS data after purge — skipping")
            continue

        is_data = pd.concat(is_block_list, ignore_index=True)

        if verbose:
            print(f"    IS blocks: {is_indices} ({len(is_data)} rows), "
                  f"OOS: {oos_idx} ({len(oos_data)} rows)")

        # Optimise on IS
        best_params, is_score = optimise_on_is(
            trader_class, is_data, param_grid, obj_fn,
            products=products, verbose=verbose,
        )

        # Evaluate on OOS
        trader, module, originals = apply_params_to_trader(trader_class, best_params)
        try:
            for k, v in best_params.items():
                setattr(module, k, v)
            oos_result = run_backtest(trader, oos_data, products=products)
            oos_score  = obj_fn(oos_result)
        except Exception as e:
            if verbose:
                print(f"    OOS eval error: {e}")
            oos_score = np.nan
        finally:
            restore_params(module, originals)

        if verbose:
            print(f"    Best IS params: {best_params}")
            print(f"    IS score: {is_score:.4f} | OOS score: {oos_score:.4f}")

        combo_results.append(ComboResult(
            combo_id    = combo_id,
            oos_blocks  = oos_idx,
            is_blocks   = tuple(is_indices),
            best_params = best_params,
            is_score    = is_score,
            oos_score   = oos_score,
        ))

        # Track param occurrence
        key = json.dumps(best_params, sort_keys=True)
        param_count[key] = param_count.get(key, 0) + 1

    # Aggregate OOS scores
    oos_scores = np.array([r.oos_score for r in combo_results
                           if not np.isnan(r.oos_score)])

    oos_mean   = float(oos_scores.mean()) if len(oos_scores) > 0 else np.nan
    oos_std    = float(oos_scores.std())  if len(oos_scores) > 0 else np.nan
    oos_sharpe = (oos_mean / (oos_std + 1e-9)) if oos_std > 0 else oos_mean

    # Recommended params = most frequently IS-optimal
    if param_count:
        best_key = max(param_count, key=param_count.get)
        recommended_params = json.loads(best_key)
    else:
        recommended_params = {}

    # Parameter stability: for each param, how much do the IS-optimal values vary?
    param_stability = {}
    if combo_results:
        all_params_used = [r.best_params for r in combo_results]
        for param_name in (list(combo_results[0].best_params.keys()) if combo_results else []):
            vals = [p[param_name] for p in all_params_used if param_name in p]
            try:
                numeric_vals = [float(v) for v in vals]
                param_stability[param_name] = {
                    "values":    vals,
                    "mean":      float(np.mean(numeric_vals)),
                    "std":       float(np.std(numeric_vals)),
                    "unique":    list(set(vals)),
                    "stable":    np.std(numeric_vals) < 0.01 * abs(np.mean(numeric_vals) + 1e-9),
                }
            except (TypeError, ValueError):
                param_stability[param_name] = {
                    "values": vals,
                    "unique": list(set(vals)),
                }

    if verbose:
        print(f"\n{'='*60}")
        print(f"CWFA SUMMARY")
        print(f"{'='*60}")
        print(f"  Combinations run:  {len(combo_results)}")
        print(f"  OOS score mean:    {oos_mean:.4f}")
        print(f"  OOS score std:     {oos_std:.4f}")
        print(f"  OOS Sharpe proxy:  {oos_sharpe:.4f}")
        print(f"  Recommended params: {recommended_params}")
        print(f"\n  Parameter stability:")
        for pname, info in param_stability.items():
            if "std" in info:
                stable_tag = "✓ stable" if info["stable"] else "⚠ variable"
                print(f"    {pname}: mean={info['mean']:.3f}, std={info['std']:.3f} — {stable_tag}")
            else:
                print(f"    {pname}: {info['unique']}")

    return CWFAResult(
        combo_results      = combo_results,
        oos_scores         = oos_scores,
        oos_mean           = oos_mean,
        oos_std            = oos_std,
        oos_sharpe         = oos_sharpe,
        best_params_counts = param_count,
        recommended_params = recommended_params,
        param_stability    = param_stability,
    )
