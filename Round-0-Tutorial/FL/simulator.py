"""
simulator.py — local exchange simulator for IMC Prosperity price data.

Fill model (matches IMC Prosperity behaviour):
  - Trader posts a BUY order at price P:
      fills against existing asks at prices <= P, up to order quantity
  - Trader posts a SELL order at price P:
      fills against existing bids at prices >= P, up to |order quantity|

PnL = realised cash flow + mark-to-market on open position at end-of-day
      (valued at final mid price).

The position limit is enforced: orders that would breach the limit are
clipped silently (same as the exchange does).
"""

import pandas as pd
import numpy as np
import sys, os, importlib, json, io
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from datamodel import OrderDepth, Order, Trade, TradingState


POSITION_LIMIT = 80   # shared across products in tutorial


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_price_data(paths: List[str]) -> pd.DataFrame:
    """
    Load one or more prices_round_X_day_Y.csv files and return a single
    sorted DataFrame with columns:
      day, timestamp, product, bid_price_1..3, bid_volume_1..3,
      ask_price_1..3, ask_volume_1..3, mid_price
    Timestamps are kept relative (0–199900 per day); the day column
    distinguishes them.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep=";")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return combined


def build_order_depth(row: pd.Series) -> OrderDepth:
    od = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv > 0:
            od.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av) and av > 0:
            # IMC convention: sell_orders volumes are negative
            od.sell_orders[int(ap)] = -int(av)
    return od


# ---------------------------------------------------------------------------
# Fill engine
# ---------------------------------------------------------------------------

def simulate_fills(
    orders:       List[Order],
    order_depth:  OrderDepth,
    position:     int,
    tick_index:   int = 0,
    block_size:   int = 5,
) -> Tuple[List[Trade], int, float]:
    """
    Block fill model.

    Fills bids for `block_size` consecutive ticks, then asks for
    `block_size` ticks, cycling. With block_size=5 and typical
    order size 5-6, position peaks at ~25-30 — within the range
    where skew trigger (15–40) and unwind logic activate.

    This is deterministic, reproducible, and exposes the inventory
    management parameters (skew trigger, velocity threshold, unwind
    timing) to meaningful stress. Absolute PnL is inflated but
    relative CWFA rankings are valid.

    Fill condition: quote must still improve the market (bid > best_bid
    or ask < best_ask) — suppressed quotes (velocity / halt logic) do
    NOT fill even if it's their block's turn.
    """
    fills: List[Trade] = []
    pos = position
    cash = 0.0

    best_bid = max(order_depth.buy_orders.keys())  if order_depth.buy_orders  else None
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

    # Determine which side is active this tick
    block = (tick_index // block_size) % 2   # 0 = bid side, 1 = ask side
    fill_bid_this_block = (block == 0)

    for order in orders:
        qty = order.quantity   # positive = buy, negative = sell

        if qty > 0 and fill_bid_this_block:
            if best_bid is not None and order.price > best_bid:
                cap = POSITION_LIMIT - pos
                fill_qty = min(qty, cap)
                if fill_qty > 0:
                    fills.append(Trade(order.symbol, order.price, fill_qty,
                                       buyer="SUBMISSION"))
                    pos  += fill_qty
                    cash -= fill_qty * order.price

        elif qty < 0 and not fill_bid_this_block:
            if best_ask is not None and order.price < best_ask:
                sell_qty = abs(qty)
                cap = POSITION_LIMIT + pos
                fill_qty = min(sell_qty, cap)
                if fill_qty > 0:
                    fills.append(Trade(order.symbol, order.price, -fill_qty,
                                       seller="SUBMISSION"))
                    pos  -= fill_qty
                    cash += fill_qty * order.price

    return fills, pos, cash


# ---------------------------------------------------------------------------
# Single backtest run
# ---------------------------------------------------------------------------

def run_backtest(
    trader,
    price_data: pd.DataFrame,
    products:   Optional[List[str]] = None,
    verbose:    bool = False,
) -> dict:
    """
    Run trader against price_data.

    Returns a dict with:
      pnl_series   : list of (timestamp, total_pnl)
      final_pnl    : float
      max_drawdown : float
      per_product  : dict  product -> {final_pnl, trades}
      trade_log    : list of Trade objects
    """
    if products is None:
        products = price_data["product"].unique().tolist()

    # State
    positions    : Dict[str, int]   = {p: 0 for p in products}
    cash         : Dict[str, float] = {p: 0.0 for p in products}
    trader_data  : str              = ""
    all_trades   : List[Trade]      = []
    pnl_series   : List[Tuple]      = []

    # Build a lookup of mid prices: (day, timestamp, product) -> mid
    mid_lookup: Dict[tuple, float] = {}
    for _, row in price_data[price_data["product"].isin(products)].iterrows():
        if pd.notna(row["mid_price"]):
            mid_lookup[(int(row["day"]), int(row["timestamp"]), row["product"])] = float(row["mid_price"])

    # Build sorted tick list: list of (day, timestamp)
    ref_prod = products[0]
    tick_index = (
        price_data[price_data["product"] == ref_prod][["day", "timestamp"]]
        .drop_duplicates()
        .sort_values(["day", "timestamp"])
        .reset_index(drop=True)
    )
    tick_list = list(zip(tick_index["day"], tick_index["timestamp"]))

    # Group all rows for fast lookup
    grouped_dict: Dict[tuple, pd.DataFrame] = {}
    for (day, ts), grp in price_data[price_data["product"].isin(products)].groupby(["day", "timestamp"]):
        grouped_dict[(int(day), int(ts))] = grp

    for tick_i, (day, ts) in enumerate(tick_list):
        tick_df = grouped_dict.get((int(day), int(ts)))
        if tick_df is None:
            continue

        # Build order_depths and mids for this tick
        order_depths: Dict[str, OrderDepth] = {}
        mids: Dict[str, Optional[float]] = {}
        for _, row in tick_df.iterrows():
            prod = row["product"]
            order_depths[prod] = build_order_depth(row)
            mids[prod] = float(row["mid_price"]) if pd.notna(row["mid_price"]) else None

        state = TradingState(
            timestamp     = int(ts),
            listings      = {p: {"symbol": p, "denomination": "XIRECS"} for p in products},
            order_depths  = order_depths,
            own_trades    = {p: [] for p in products},
            market_trades = {p: [] for p in products},
            position      = dict(positions),
            observations  = {},
            traderData    = trader_data,
        )

        try:
            result, conversions, new_trader_data = trader.run(state)
        except Exception as e:
            if verbose:
                print(f"  Trader error at ts={ts}: {e}")
            result, new_trader_data = {}, trader_data

        trader_data = new_trader_data if isinstance(new_trader_data, str) else json.dumps(new_trader_data)

        # Process orders
        for prod, orders in result.items():
            if prod not in products:
                continue
            fills, new_pos, cash_delta = simulate_fills(
                orders, order_depths[prod], positions[prod],
                tick_index=tick_i,
            )
            positions[prod] = new_pos
            cash[prod]     += cash_delta
            all_trades.extend(fills)

        # Compute total PnL: realised cash + mark-to-market
        total_pnl = 0.0
        for prod in products:
            mid = mids.get(prod)
            if mid is not None:
                total_pnl += cash[prod] + positions[prod] * mid

        pnl_series.append((int(ts), total_pnl))

    # Final PnL
    final_pnl = pnl_series[-1][1] if pnl_series else 0.0

    # Max drawdown
    pnl_vals = np.array([p for _, p in pnl_series])
    running_max = np.maximum.accumulate(pnl_vals)
    drawdowns = pnl_vals - running_max
    max_drawdown = float(drawdowns.min())

    # Per-product breakdown
    last_mids: Dict[str, float] = {}
    for prod in products:
        for i in range(len(tick_list) - 1, -1, -1):
            day, ts = tick_list[i]
            m = mid_lookup.get((int(day), int(ts), prod))
            if m is not None:
                last_mids[prod] = m
                break

    per_product = {}
    for prod in products:
        mid = last_mids.get(prod, 0.0)
        per_product[prod] = {
            "final_pnl":       cash[prod] + positions[prod] * mid,
            "final_position":  positions[prod],
        }

    return {
        "pnl_series":   pnl_series,
        "final_pnl":    final_pnl,
        "max_drawdown": max_drawdown,
        "per_product":  per_product,
        "trade_log":    all_trades,
        "positions":    positions,
    }
