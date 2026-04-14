import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\indra\Documents\New project")
DOWNLOADS = Path(r"C:\Users\indra\Downloads")
KEVIN_ROOT = ROOT / "imc-prosperity-4-backtester"
KEVIN_PACKAGE = KEVIN_ROOT / "prosperity4bt"

PRICE_PATHS = {
    -2: DOWNLOADS / "prices_round_1_day_-2.csv",
    -1: DOWNLOADS / "prices_round_1_day_-1.csv",
    0: DOWNLOADS / "prices_round_1_day_0.csv",
}
TRADE_PATHS = {
    -2: DOWNLOADS / "trades_round_1_day_-2.csv",
    -1: DOWNLOADS / "trades_round_1_day_-1.csv",
    0: DOWNLOADS / "trades_round_1_day_0.csv",
}

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
LIMITS = {"INTARIAN_PEPPER_ROOT": 80, "ASH_COATED_OSMIUM": 80}
PEPPER_TREND_PER_TICK = 1000.0 / 10000.0


@dataclass
class QuoteDecision:
    bid_price: Optional[int] = None
    bid_size: int = 0
    ask_price: Optional[int] = None
    ask_size: int = 0


@dataclass
class Metrics:
    product: str
    strategy: str
    engine: str
    scope: str
    pnl: float
    sharpe: float
    max_drawdown: float
    fill_rate: float
    trade_count: int
    submitted_orders: int
    filled_orders: int
    max_abs_position: int
    pnl_series: list[float]


def read_price_day(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    numeric_cols = [c for c in df.columns if c != "product"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def read_trade_day(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "price", "quantity"])
    df = pd.read_csv(path, sep=";")
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "price", "quantity"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype(int)
    return df


def load_round1_csv_data(price_paths: dict[int, Path], trade_paths: dict[int, Path]) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    return (
        {day: read_price_day(path) for day, path in price_paths.items()},
        {day: read_trade_day(path) for day, path in trade_paths.items()},
    )


def _book_side(row: pd.Series, side: str) -> list[tuple[int, int]]:
    if side == "bid":
        price_cols = ["bid_price_1", "bid_price_2", "bid_price_3"]
        vol_cols = ["bid_volume_1", "bid_volume_2", "bid_volume_3"]
    else:
        price_cols = ["ask_price_1", "ask_price_2", "ask_price_3"]
        vol_cols = ["ask_volume_1", "ask_volume_2", "ask_volume_3"]
    levels: list[tuple[int, int]] = []
    for price_col, vol_col in zip(price_cols, vol_cols):
        price = row.get(price_col)
        vol = row.get(vol_col)
        if pd.isna(price) or pd.isna(vol):
            continue
        levels.append((int(price), int(abs(vol))))
    return levels


def row_to_snapshot(row: pd.Series, trades_at_ts: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    bids = _book_side(row, "bid")
    asks = _book_side(row, "ask")
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    mid = float(row["mid_price"])
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
        bid_vol = bids[0][1]
        ask_vol = asks[0][1]
        total = bid_vol + ask_vol
        micro = (best_bid * ask_vol + best_ask * bid_vol) / total if total else mid
    else:
        spread = 999
        bid_vol = 0
        ask_vol = 0
        micro = mid
    total_book = sum(v for _, v in bids) + sum(v for _, v in asks)
    imbalance = 0.0 if total_book == 0 else (sum(v for _, v in bids) - sum(v for _, v in asks)) / total_book
    return {
        "day": int(row["day"]),
        "timestamp": int(row["timestamp"]),
        "product": row["product"],
        "mid": mid,
        "spread": int(spread),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "best_bid_vol": int(bid_vol),
        "best_ask_vol": int(ask_vol),
        "bids": bids,
        "asks": asks,
        "imbalance": float(imbalance),
        "microprice": float(micro),
        "trades": trades_at_ts if trades_at_ts is not None else pd.DataFrame(columns=["timestamp", "symbol", "price", "quantity"]),
    }


def round_price(value: float) -> int:
    return int(round(value))


def clip_size(side: str, requested: int, position: int, limit: int) -> int:
    if requested <= 0:
        return 0
    if side == "buy":
        return max(0, min(requested, limit - position))
    return max(0, min(requested, limit + position))


def inventory_skew(position: int, limit: int, width: float, scale: float = 0.9) -> float:
    return scale * width * (position / max(limit, 1))


def init_product_state(product: str) -> dict[str, Any]:
    return {
        "product": product,
        "day": None,
        "day_open": None,
        "day_start_ts": None,
        "last_mid": None,
        "last_spread": None,
        "last_imbalance": 0.0,
        "vol_window": [],
        "breakout_wait": 0,
        "breakout_direction": 0,
    }


def reset_for_new_day(state: dict[str, Any], snapshot: dict[str, Any]) -> None:
    state["day"] = snapshot["day"]
    state["day_open"] = snapshot["mid"]
    state["day_start_ts"] = snapshot["timestamp"]
    state["last_mid"] = snapshot["mid"]
    state["last_spread"] = snapshot["spread"]
    state["last_imbalance"] = snapshot["imbalance"]
    state["vol_window"] = []
    state["breakout_wait"] = 0
    state["breakout_direction"] = 0


def ensure_state(state: dict[str, Any], snapshot: dict[str, Any]) -> None:
    if state["day"] != snapshot["day"] or state["day_open"] is None:
        reset_for_new_day(state, snapshot)
        return
    last_mid = state["last_mid"]
    if last_mid is not None:
        state["vol_window"].append(abs(snapshot["mid"] - last_mid))
        if len(state["vol_window"]) > 20:
            state["vol_window"] = state["vol_window"][-20:]
    state["last_mid"] = snapshot["mid"]
    state["last_spread"] = snapshot["spread"]
    state["last_imbalance"] = snapshot["imbalance"]


def pepper_fair_value(state: dict[str, Any], snapshot: dict[str, Any]) -> float:
    ticks_elapsed = (snapshot["timestamp"] - state["day_start_ts"]) / 100.0
    return float(state["day_open"] + PEPPER_TREND_PER_TICK * ticks_elapsed)


def pepper_hold_wide(snapshot: dict[str, Any], state: dict[str, Any]) -> QuoteDecision:
    fair = pepper_fair_value(state, snapshot)
    return QuoteDecision(round_price(fair - 40), 0, round_price(fair + 40), 0)


def pepper_trend_rider(snapshot: dict[str, Any], state: dict[str, Any], position: int) -> QuoteDecision:
    ensure_state(state, snapshot)
    best_ask = snapshot["best_ask"] or round_price(snapshot["mid"] + 6)
    if position < 80:
        return QuoteDecision(best_ask + 2, min(20, 80 - position), None, 0)
    return pepper_hold_wide(snapshot, state)


def pepper_dip_accumulator(snapshot: dict[str, Any], state: dict[str, Any], position: int) -> QuoteDecision:
    ensure_state(state, snapshot)
    fair = pepper_fair_value(state, snapshot)
    deviation = snapshot["mid"] - fair
    target = 80 if deviation <= 2 else 60
    best_ask = snapshot["best_ask"] or round_price(snapshot["mid"] + 6)
    if position < target:
        return QuoteDecision(best_ask + 1, min(16, target - position), None, 0)
    return pepper_hold_wide(snapshot, state)


def pepper_signal_ramp(snapshot: dict[str, Any], state: dict[str, Any], position: int) -> QuoteDecision:
    ensure_state(state, snapshot)
    signal = snapshot["imbalance"] + 0.5 * ((snapshot["microprice"] - snapshot["mid"]) / max(snapshot["spread"], 1))
    target = 50 if snapshot["timestamp"] < 2000 else 80
    if signal > 0.05:
        target = 80
    best_ask = snapshot["best_ask"] or round_price(snapshot["mid"] + 6)
    if position < target:
        step = 12 if snapshot["timestamp"] < 2000 else 18
        return QuoteDecision(best_ask + 2, min(step, target - position), None, 0)
    return pepper_hold_wide(snapshot, state)


def osmium_fixed_anchor_mm(snapshot: dict[str, Any], state: dict[str, Any], position: int) -> QuoteDecision:
    ensure_state(state, snapshot)
    anchor = 10000.0
    half = 8.0
    skew = inventory_skew(position, LIMITS[snapshot["product"]], half)
    return QuoteDecision(
        round_price(anchor - half - skew),
        clip_size("buy", 6, position, LIMITS[snapshot["product"]]),
        round_price(anchor + half - skew),
        clip_size("sell", 6, position, LIMITS[snapshot["product"]]),
    )


def osmium_volatility_adaptive(snapshot: dict[str, Any], state: dict[str, Any], position: int) -> QuoteDecision:
    ensure_state(state, snapshot)
    anchor = 10000.0
    vol = float(np.mean(state["vol_window"][-20:])) if state["vol_window"] else 3.6
    if vol < 3.0:
        half, size = 6.0, 8
    elif vol < 4.2:
        half, size = 8.0, 6
    else:
        half, size = 11.0, 4
    skew = inventory_skew(position, LIMITS[snapshot["product"]], half)
    return QuoteDecision(
        round_price(anchor - half - skew),
        clip_size("buy", size, position, LIMITS[snapshot["product"]]),
        round_price(anchor + half - skew),
        clip_size("sell", size, position, LIMITS[snapshot["product"]]),
    )


def osmium_narrow_spread_breakout(snapshot: dict[str, Any], state: dict[str, Any], position: int) -> QuoteDecision:
    ensure_state(state, snapshot)
    anchor = 10000.0
    if snapshot["spread"] < 10:
        state["breakout_wait"] = 2
        state["breakout_direction"] = 1 if snapshot["microprice"] > snapshot["mid"] or snapshot["imbalance"] > 0.15 else -1 if snapshot["microprice"] < snapshot["mid"] or snapshot["imbalance"] < -0.15 else 0
    if state["breakout_wait"] > 0:
        state["breakout_wait"] -= 1
        if state["breakout_direction"] > 0:
            return QuoteDecision(
                round_price((snapshot["best_ask"] or snapshot["mid"]) - 1),
                clip_size("buy", 8, position, LIMITS[snapshot["product"]]),
                round_price(anchor + 10),
                clip_size("sell", 2, position, LIMITS[snapshot["product"]]),
            )
        if state["breakout_direction"] < 0:
            return QuoteDecision(
                round_price(anchor - 10),
                clip_size("buy", 2, position, LIMITS[snapshot["product"]]),
                round_price((snapshot["best_bid"] or snapshot["mid"]) + 1),
                clip_size("sell", 8, position, LIMITS[snapshot["product"]]),
            )
    half = 8.0
    skew = inventory_skew(position, LIMITS[snapshot["product"]], half)
    return QuoteDecision(
        round_price(anchor - half - skew),
        clip_size("buy", 5, position, LIMITS[snapshot["product"]]),
        round_price(anchor + half - skew),
        clip_size("sell", 5, position, LIMITS[snapshot["product"]]),
    )


STRATEGIES: dict[str, dict[str, Callable[[dict[str, Any], dict[str, Any], int], QuoteDecision]]] = {
    "INTARIAN_PEPPER_ROOT": {
        "pepper_trend_rider": pepper_trend_rider,
        "pepper_dip_accumulator": pepper_dip_accumulator,
        "pepper_signal_ramp": pepper_signal_ramp,
    },
    "ASH_COATED_OSMIUM": {
        "osmium_fixed_anchor_mm": osmium_fixed_anchor_mm,
        "osmium_volatility_adaptive": osmium_volatility_adaptive,
        "osmium_narrow_spread_breakout": osmium_narrow_spread_breakout,
    },
}


class PassiveResearchBacktester:
    def __init__(self, prices_by_day: dict[int, pd.DataFrame], trades_by_day: dict[int, pd.DataFrame], queue_share: float = 0.35):
        self.prices_by_day = prices_by_day
        self.trades_by_day = trades_by_day
        self.queue_share = queue_share
        self.snapshots: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for day, price_df in prices_by_day.items():
            trade_df = trades_by_day[day]
            for product in PRODUCTS:
                day_prices = price_df.loc[price_df["product"] == product].sort_values("timestamp").reset_index(drop=True)
                trades_by_ts = {ts: frame.reset_index(drop=True) for ts, frame in trade_df.loc[trade_df["symbol"] == product].groupby("timestamp")}
                self.snapshots[(day, product)] = [row_to_snapshot(row, trades_by_ts.get(int(row["timestamp"]))) for _, row in day_prices.iterrows()]

    def run(self, product: str, strategy_name: str, days: list[int], scope: str) -> Metrics:
        strategy = STRATEGIES[product][strategy_name]
        product_state = init_product_state(product)
        position = 0
        cash = 0.0
        pnl_series = [0.0]
        submitted_orders = 0
        filled_orders = 0
        trade_count = 0
        max_abs_position = 0

        for day in days:
            day_snapshots = self.snapshots[(day, product)]
            reset_for_new_day(product_state, day_snapshots[0])
            for idx in range(len(day_snapshots) - 1):
                snapshot = day_snapshots[idx]
                next_snapshot = day_snapshots[idx + 1]
                decision = strategy(snapshot, product_state, position)
                submitted = []
                if decision.bid_price is not None and decision.bid_size > 0:
                    submitted.append(("buy", int(decision.bid_price), int(decision.bid_size)))
                if decision.ask_price is not None and decision.ask_size > 0:
                    submitted.append(("sell", int(decision.ask_price), int(decision.ask_size)))
                submitted_orders += len(submitted)
                for side, price, qty in submitted:
                    fill_qty, fill_price = self._simulate_fill(side, price, qty, next_snapshot)
                    if fill_qty <= 0:
                        continue
                    filled_orders += 1
                    trade_count += 1
                    if side == "buy":
                        position += fill_qty
                        cash -= fill_qty * fill_price
                    else:
                        position -= fill_qty
                        cash += fill_qty * fill_price
                max_abs_position = max(max_abs_position, abs(position))
                pnl_series.append(cash + position * next_snapshot["mid"])

        pnl_arr = np.asarray(pnl_series, dtype=float)
        pnl_changes = np.diff(pnl_arr)
        sharpe = float(np.mean(pnl_changes) / (np.std(pnl_changes) + 1e-9)) if len(pnl_changes) else 0.0
        peaks = np.maximum.accumulate(pnl_arr)
        drawdowns = np.where(peaks != 0, (peaks - pnl_arr) / np.maximum(np.abs(peaks), 1.0), 0.0)
        return Metrics(
            product=product,
            strategy=strategy_name,
            engine="custom_csv",
            scope=scope,
            pnl=float(pnl_arr[-1]),
            sharpe=sharpe,
            max_drawdown=float(np.max(drawdowns)) if len(drawdowns) else 0.0,
            fill_rate=float(filled_orders / submitted_orders) if submitted_orders else 0.0,
            trade_count=trade_count,
            submitted_orders=submitted_orders,
            filled_orders=filled_orders,
            max_abs_position=max_abs_position,
            pnl_series=pnl_arr.tolist(),
        )

    def _simulate_fill(self, side: str, price: int, qty: int, next_snapshot: dict[str, Any]) -> tuple[int, float]:
        if qty <= 0:
            return 0, 0.0
        trades = next_snapshot["trades"]
        if side == "buy":
            if next_snapshot["best_ask"] is not None and next_snapshot["best_ask"] <= price:
                cross_volume = sum(v for p, v in next_snapshot["asks"] if p <= price)
                return min(qty, max(1, cross_volume)), float(next_snapshot["best_ask"])
            marketable = 0 if trades.empty else int(trades.loc[trades["price"] <= price, "quantity"].sum())
            return (min(qty, max(1, int(round(marketable * self.queue_share)))), float(price)) if marketable > 0 else (0, 0.0)
        if next_snapshot["best_bid"] is not None and next_snapshot["best_bid"] >= price:
            cross_volume = sum(v for p, v in next_snapshot["bids"] if p >= price)
            return min(qty, max(1, cross_volume)), float(next_snapshot["best_bid"])
        marketable = 0 if trades.empty else int(trades.loc[trades["price"] >= price, "quantity"].sum())
        return (min(qty, max(1, int(round(marketable * self.queue_share)))), float(price)) if marketable > 0 else (0, 0.0)


def metrics_to_frame(results: list[Metrics]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product": r.product,
                "strategy": r.strategy,
                "engine": r.engine,
                "scope": r.scope,
                "pnl": r.pnl,
                "sharpe": r.sharpe,
                "max_drawdown": r.max_drawdown,
                "fill_rate": r.fill_rate,
                "trade_count": r.trade_count,
                "submitted_orders": r.submitted_orders,
                "filled_orders": r.filled_orders,
                "max_abs_position": r.max_abs_position,
            }
            for r in results
        ]
    )


def plot_metric_curves(results: list[Metrics], product: str, engine: str, scope: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = [r for r in results if r.product == product and r.engine == engine and r.scope == scope]
    fig, ax = plt.subplots(figsize=(12, 6))
    for result in subset:
        ax.plot(result.pnl_series, label=f"{result.strategy} | pnl={result.pnl:.1f} | sr={result.sharpe:.2f}")
    ax.set_title(f"{product} | {engine} | {scope}")
    ax.set_xlabel("tick")
    ax.set_ylabel("PnL")
    ax.grid(alpha=0.25)
    ax.legend()
    path = output_dir / f"{product.lower()}_{engine}_{scope}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_selection_table(custom_results: list[Metrics], kevin_results: list[Metrics]) -> pd.DataFrame:
    rows = []
    for product in PRODUCTS:
        custom_val = [r for r in custom_results if r.product == product and r.scope == "validation"]
        kevin_val = [r for r in kevin_results if r.product == product and r.scope == "validation"]
        for strategy in sorted({r.strategy for r in custom_val + kevin_val}):
            c = next((r for r in custom_val if r.strategy == strategy), None)
            k = next((r for r in kevin_val if r.strategy == strategy), None)
            score = 0.0
            if c is not None:
                score += c.pnl + 500.0 * c.sharpe - 2000.0 * c.max_drawdown
            if k is not None:
                score += k.pnl + 500.0 * k.sharpe - 2000.0 * k.max_drawdown
            rows.append(
                {
                    "product": product,
                    "strategy": strategy,
                    "custom_validation_pnl": None if c is None else c.pnl,
                    "custom_validation_sharpe": None if c is None else c.sharpe,
                    "kevin_validation_pnl": None if k is None else k.pnl,
                    "kevin_validation_sharpe": None if k is None else k.sharpe,
                    "selection_score": score,
                }
            )
    return pd.DataFrame(rows).sort_values(["product", "selection_score"], ascending=[True, False]).reset_index(drop=True)


def choose_best_strategies(selection_table: pd.DataFrame) -> dict[str, str]:
    return {
        product: str(
            selection_table.loc[selection_table["product"] == product]
            .sort_values("selection_score", ascending=False)
            .iloc[0]["strategy"]
        )
        for product in PRODUCTS
    }


class NotebookKevinTrader:
    def __init__(self, pepper_strategy: Optional[str], osmium_strategy: Optional[str]):
        self.pepper_strategy = pepper_strategy
        self.osmium_strategy = osmium_strategy

    def run(self, state):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        product_states = data.get("product_states", {product: init_product_state(product) for product in PRODUCTS})
        orders = {}
        for product, strategy_name in [("INTARIAN_PEPPER_ROOT", self.pepper_strategy), ("ASH_COATED_OSMIUM", self.osmium_strategy)]:
            if strategy_name is None or product not in state.order_depths:
                continue
            depth = state.order_depths[product]
            bids = sorted(depth.buy_orders.items(), reverse=True)
            asks = sorted(depth.sell_orders.items())
            best_bid = bids[0][0] if bids else None
            best_ask = asks[0][0] if asks else None
            mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else float(best_bid or best_ask or (11500 if product == "INTARIAN_PEPPER_ROOT" else 10000))
            spread = best_ask - best_bid if best_bid is not None and best_ask is not None else 999
            best_bid_vol = 0 if not bids else bids[0][1]
            best_ask_vol = 0 if not asks else abs(asks[0][1])
            total = best_bid_vol + best_ask_vol
            micro = (best_bid * best_ask_vol + best_ask * best_bid_vol) / total if total and best_bid is not None and best_ask is not None else mid
            total_book = sum(v for _, v in bids) + sum(abs(v) for _, v in asks)
            imbalance = 0.0 if total_book == 0 else (sum(v for _, v in bids) - sum(abs(v) for _, v in asks)) / total_book
            snapshot = {
                "day": 0,
                "timestamp": state.timestamp,
                "product": product,
                "mid": mid,
                "spread": spread,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "best_bid_vol": best_bid_vol,
                "best_ask_vol": best_ask_vol,
                "bids": [(int(p), int(v)) for p, v in bids],
                "asks": [(int(p), int(abs(v))) for p, v in asks],
                "imbalance": float(imbalance),
                "microprice": float(micro),
                "trades": pd.DataFrame(),
            }
            decision = STRATEGIES[product][strategy_name](snapshot, product_states.setdefault(product, init_product_state(product)), state.position.get(product, 0))
            product_orders = []
            if decision.bid_size > 0:
                product_orders.append(self._order(product, int(decision.bid_price), int(decision.bid_size)))
            if decision.ask_size > 0:
                product_orders.append(self._order(product, int(decision.ask_price), -int(decision.ask_size)))
            if product_orders:
                orders[product] = product_orders
        return orders, {}, json.dumps({"product_states": product_states}, separators=(",", ":"))

    def _order(self, product: str, price: int, quantity: int):
        if "datamodel" in sys.modules:
            return sys.modules["datamodel"].Order(product, price, quantity)
        from prosperity4bt.datamodel import Order
        return Order(product, price, quantity)


def summarize_kevin_result(result: Any, product: str, strategy: str, scope: str) -> Metrics:
    rows = [row for row in result.activity_logs if row.symbol == product]
    pnl_series = [float(row.profit_loss) for row in rows] or [0.0]
    pnl_arr = np.asarray(pnl_series, dtype=float)
    pnl_changes = np.diff(pnl_arr)
    peaks = np.maximum.accumulate(pnl_arr)
    drawdowns = np.where(peaks != 0, (peaks - pnl_arr) / np.maximum(np.abs(peaks), 1.0), 0.0)
    trades = [trade for trade in result.trades if trade.trade.symbol == product]
    return Metrics(
        product=product,
        strategy=strategy,
        engine="kevin_backtester",
        scope=scope,
        pnl=float(pnl_arr[-1]),
        sharpe=float(np.mean(pnl_changes) / (np.std(pnl_changes) + 1e-9)) if len(pnl_changes) else 0.0,
        max_drawdown=float(np.max(drawdowns)) if len(drawdowns) else 0.0,
        fill_rate=float(len(trades) / max(1, len(rows))),
        trade_count=len(trades),
        submitted_orders=len(rows),
        filled_orders=len(trades),
        max_abs_position=0,
        pnl_series=pnl_arr.tolist(),
    )


def run_kevin_backtests(round_days: dict[str, list[int]]) -> list[Metrics]:
    sys.path.insert(0, str(KEVIN_ROOT))
    sys.path.insert(0, str(KEVIN_PACKAGE))
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import FileSystemReader

    reader = FileSystemReader(KEVIN_PACKAGE / "resources")
    results: list[Metrics] = []
    for product in PRODUCTS:
        for strategy_name in STRATEGIES[product]:
            for scope, days in round_days.items():
                merged_series: list[float] = []
                total_pnl = 0.0
                total_trades = 0
                filled_orders = 0
                submitted_orders = 0
                max_dd = 0.0
                for day in days:
                    trader = NotebookKevinTrader(strategy_name if product == "INTARIAN_PEPPER_ROOT" else None, strategy_name if product == "ASH_COATED_OSMIUM" else None)
                    summary = summarize_kevin_result(TestRunner(trader, reader, 1, day, False, False).run(), product, strategy_name, scope)
                    total_pnl += summary.pnl
                    total_trades += summary.trade_count
                    filled_orders += summary.filled_orders
                    submitted_orders += summary.submitted_orders
                    max_dd = max(max_dd, summary.max_drawdown)
                    merged_series.extend(summary.pnl_series if not merged_series else [merged_series[-1] + x for x in summary.pnl_series])
                pnl_arr = np.asarray(merged_series if merged_series else [0.0], dtype=float)
                pnl_changes = np.diff(pnl_arr)
                results.append(
                    Metrics(
                        product=product,
                        strategy=strategy_name,
                        engine="kevin_backtester",
                        scope=scope,
                        pnl=total_pnl,
                        sharpe=float(np.mean(pnl_changes) / (np.std(pnl_changes) + 1e-9)) if len(pnl_changes) else 0.0,
                        max_drawdown=max_dd,
                        fill_rate=float(filled_orders / submitted_orders) if submitted_orders else 0.0,
                        trade_count=total_trades,
                        submitted_orders=submitted_orders,
                        filled_orders=filled_orders,
                        max_abs_position=0,
                        pnl_series=pnl_arr.tolist(),
                    )
                )
    return results


def run_custom_backtests(round_days: dict[str, list[int]]) -> list[Metrics]:
    prices, trades = load_round1_csv_data(PRICE_PATHS, TRADE_PATHS)
    engine = PassiveResearchBacktester(prices, trades)
    return [engine.run(product, strategy_name, days, scope) for product in PRODUCTS for strategy_name in STRATEGIES[product] for scope, days in round_days.items()]


def export_submission_file(selected: dict[str, str], output_path: Path) -> Path:
    content = f'''import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ModuleNotFoundError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

LIMITS = {{"INTARIAN_PEPPER_ROOT": 80, "ASH_COATED_OSMIUM": 80}}
PEPPER_TREND_PER_TICK = 1000.0 / 10000.0
PEPPER_STRATEGY = "{selected["INTARIAN_PEPPER_ROOT"]}"
OSMIUM_STRATEGY = "{selected["ASH_COATED_OSMIUM"]}"

def round_price(value: float) -> int:
    return int(round(value))

def clip_size(side: str, requested: int, position: int, limit: int) -> int:
    if requested <= 0:
        return 0
    if side == "buy":
        return max(0, min(requested, limit - position))
    return max(0, min(requested, limit + position))

def inventory_skew(position: int, limit: int, width: float, scale: float = 0.9) -> float:
    return scale * width * (position / max(limit, 1))

def init_product_state(product: str) -> dict[str, Any]:
    return {{"product": product, "day_open": None, "day_start_ts": None, "last_mid": None, "last_spread": None, "last_imbalance": 0.0, "vol_window": [], "breakout_wait": 0, "breakout_direction": 0}}

def build_snapshot(product: str, depth: OrderDepth, timestamp: int) -> dict[str, Any]:
    bids = sorted(depth.buy_orders.items(), reverse=True)
    asks = sorted(depth.sell_orders.items())
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else float(best_bid or best_ask or (11500 if product == "INTARIAN_PEPPER_ROOT" else 10000))
    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else 999
    bid_vol = 0 if not bids else bids[0][1]
    ask_vol = 0 if not asks else abs(asks[0][1])
    total = bid_vol + ask_vol
    micro = (best_bid * ask_vol + best_ask * bid_vol) / total if total and best_bid is not None and best_ask is not None else mid
    total_book = sum(v for _, v in bids) + sum(abs(v) for _, v in asks)
    imbalance = 0.0 if total_book == 0 else (sum(v for _, v in bids) - sum(abs(v) for _, v in asks)) / total_book
    return {{"product": product, "timestamp": timestamp, "mid": mid, "spread": spread, "best_bid": best_bid, "best_ask": best_ask, "best_bid_vol": bid_vol, "best_ask_vol": ask_vol, "imbalance": imbalance, "microprice": micro}}

def ensure_state(state: dict[str, Any], snapshot: dict[str, Any]) -> None:
    if state["day_open"] is None or snapshot["timestamp"] < (state["day_start_ts"] or 0):
        state["day_open"] = snapshot["mid"]
        state["day_start_ts"] = snapshot["timestamp"]
        state["last_mid"] = snapshot["mid"]
        state["last_spread"] = snapshot["spread"]
        state["last_imbalance"] = snapshot["imbalance"]
        state["vol_window"] = []
        state["breakout_wait"] = 0
        state["breakout_direction"] = 0
        return
    last_mid = state["last_mid"]
    if last_mid is not None:
        state["vol_window"].append(abs(snapshot["mid"] - last_mid))
        if len(state["vol_window"]) > 20:
            state["vol_window"] = state["vol_window"][-20:]
    state["last_mid"] = snapshot["mid"]
    state["last_spread"] = snapshot["spread"]
    state["last_imbalance"] = snapshot["imbalance"]

def pepper_fair_value(state: dict[str, Any], snapshot: dict[str, Any]) -> float:
    return float(state["day_open"] + PEPPER_TREND_PER_TICK * ((snapshot["timestamp"] - state["day_start_ts"]) / 100.0))

def pepper_hold_wide(snapshot: dict[str, Any], state: dict[str, Any]):
    fair = pepper_fair_value(state, snapshot)
    return round_price(fair - 40), 0, round_price(fair + 40), 0

def pepper_trend_rider(snapshot: dict[str, Any], state: dict[str, Any], position: int):
    ensure_state(state, snapshot)
    best_ask = snapshot["best_ask"] or round_price(snapshot["mid"] + 6)
    if position < 80:
        return best_ask + 2, min(20, 80 - position), None, 0
    return pepper_hold_wide(snapshot, state)

def pepper_dip_accumulator(snapshot: dict[str, Any], state: dict[str, Any], position: int):
    ensure_state(state, snapshot)
    fair = pepper_fair_value(state, snapshot)
    deviation = snapshot["mid"] - fair
    target = 80 if deviation <= 2 else 60
    best_ask = snapshot["best_ask"] or round_price(snapshot["mid"] + 6)
    if position < target:
        return best_ask + 1, min(16, target - position), None, 0
    return pepper_hold_wide(snapshot, state)

def pepper_signal_ramp(snapshot: dict[str, Any], state: dict[str, Any], position: int):
    ensure_state(state, snapshot)
    signal = snapshot["imbalance"] + 0.5 * ((snapshot["microprice"] - snapshot["mid"]) / max(snapshot["spread"], 1))
    target = 50 if snapshot["timestamp"] < 2000 else 80
    if signal > 0.05:
        target = 80
    best_ask = snapshot["best_ask"] or round_price(snapshot["mid"] + 6)
    if position < target:
        step = 12 if snapshot["timestamp"] < 2000 else 18
        return best_ask + 2, min(step, target - position), None, 0
    return pepper_hold_wide(snapshot, state)

def osmium_fixed_anchor_mm(snapshot: dict[str, Any], state: dict[str, Any], position: int):
    ensure_state(state, snapshot)
    half = 8.0
    skew = inventory_skew(position, LIMITS[snapshot["product"]], half)
    return round_price(10000 - half - skew), clip_size("buy", 6, position, LIMITS[snapshot["product"]]), round_price(10000 + half - skew), clip_size("sell", 6, position, LIMITS[snapshot["product"]])

def osmium_volatility_adaptive(snapshot: dict[str, Any], state: dict[str, Any], position: int):
    ensure_state(state, snapshot)
    vol = sum(state["vol_window"]) / len(state["vol_window"]) if state["vol_window"] else 3.6
    half, size = (6.0, 8) if vol < 3.0 else (8.0, 6) if vol < 4.2 else (11.0, 4)
    skew = inventory_skew(position, LIMITS[snapshot["product"]], half)
    return round_price(10000 - half - skew), clip_size("buy", size, position, LIMITS[snapshot["product"]]), round_price(10000 + half - skew), clip_size("sell", size, position, LIMITS[snapshot["product"]])

def osmium_narrow_spread_breakout(snapshot: dict[str, Any], state: dict[str, Any], position: int):
    ensure_state(state, snapshot)
    if snapshot["spread"] < 10:
        state["breakout_wait"] = 2
        state["breakout_direction"] = 1 if snapshot["microprice"] > snapshot["mid"] or snapshot["imbalance"] > 0.15 else -1 if snapshot["microprice"] < snapshot["mid"] or snapshot["imbalance"] < -0.15 else 0
    if state["breakout_wait"] > 0:
        state["breakout_wait"] -= 1
        if state["breakout_direction"] > 0:
            return round_price((snapshot["best_ask"] or snapshot["mid"]) - 1), clip_size("buy", 8, position, LIMITS[snapshot["product"]]), 10010, clip_size("sell", 2, position, LIMITS[snapshot["product"]])
        if state["breakout_direction"] < 0:
            return 9990, clip_size("buy", 2, position, LIMITS[snapshot["product"]]), round_price((snapshot["best_bid"] or snapshot["mid"]) + 1), clip_size("sell", 8, position, LIMITS[snapshot["product"]])
    half = 8.0
    skew = inventory_skew(position, LIMITS[snapshot["product"]], half)
    return round_price(10000 - half - skew), clip_size("buy", 5, position, LIMITS[snapshot["product"]]), round_price(10000 + half - skew), clip_size("sell", 5, position, LIMITS[snapshot["product"]])

STRATEGY_MAP = {{"pepper_trend_rider": pepper_trend_rider, "pepper_dip_accumulator": pepper_dip_accumulator, "pepper_signal_ramp": pepper_signal_ramp, "osmium_fixed_anchor_mm": osmium_fixed_anchor_mm, "osmium_volatility_adaptive": osmium_volatility_adaptive, "osmium_narrow_spread_breakout": osmium_narrow_spread_breakout}}

class Trader:
    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {{}}
        except Exception:
            data = {{}}
        product_states = data.get("product_states", {{"INTARIAN_PEPPER_ROOT": init_product_state("INTARIAN_PEPPER_ROOT"), "ASH_COATED_OSMIUM": init_product_state("ASH_COATED_OSMIUM")}})
        orders = {{}}
        for product, strategy_name in [("INTARIAN_PEPPER_ROOT", PEPPER_STRATEGY), ("ASH_COATED_OSMIUM", OSMIUM_STRATEGY)]:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            snapshot = build_snapshot(product, depth, state.timestamp)
            bid_price, bid_size, ask_price, ask_size = STRATEGY_MAP[strategy_name](snapshot, product_states.setdefault(product, init_product_state(product)), state.position.get(product, 0))
            product_orders = []
            if bid_size > 0:
                product_orders.append(Order(product, int(bid_price), int(bid_size)))
            if ask_size > 0:
                product_orders.append(Order(product, int(ask_price), -int(ask_size)))
            if product_orders:
                orders[product] = product_orders
        return orders, {{}}, json.dumps({{"product_states": product_states}}, separators=(",", ":"))
'''
    output_path.write_text(content, encoding="utf-8")
    return output_path


def export_notebook(notebook_path: Path, submission_path: Path, charts_dir: Path) -> Path:
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# IMC Prosperity Round 1 rebuild\nThis notebook runs all requested round 1 backtests on your CSV data and on Kevin Fu's backtester, then exports a submission-ready trader file."),
        nbf.v4.new_code_cell("from pathlib import Path\nfrom round1_rebuild import run_custom_backtests, run_kevin_backtests, metrics_to_frame, plot_metric_curves, build_selection_table, choose_best_strategies, export_submission_file\nROUND_DAYS = {'train': [-2, -1], 'validation': [0]}\nCHARTS_DIR = Path(r'" + str(charts_dir) + "')\nSUBMISSION_PATH = Path(r'" + str(submission_path) + "')"),
        nbf.v4.new_code_cell("custom_results = run_custom_backtests(ROUND_DAYS)\ncustom_df = metrics_to_frame(custom_results)\ncustom_df.sort_values(['product', 'scope', 'pnl'], ascending=[True, True, False])"),
        nbf.v4.new_code_cell("kevin_results = run_kevin_backtests(ROUND_DAYS)\nkevin_df = metrics_to_frame(kevin_results)\nkevin_df.sort_values(['product', 'scope', 'pnl'], ascending=[True, True, False])"),
        nbf.v4.new_code_cell("for engine_name, results in [('custom_csv', custom_results), ('kevin_backtester', kevin_results)]:\n    for product in ['INTARIAN_PEPPER_ROOT', 'ASH_COATED_OSMIUM']:\n        for scope in ['train', 'validation']:\n            plot_metric_curves(results, product, engine_name, scope, CHARTS_DIR)"),
        nbf.v4.new_code_cell("selection_df = build_selection_table(custom_results, kevin_results)\nselection_df"),
        nbf.v4.new_code_cell("selected = choose_best_strategies(selection_df)\nselected"),
        nbf.v4.new_code_cell("export_submission_file(selected, SUBMISSION_PATH)\nSUBMISSION_PATH"),
    ]
    nbf.write(nb, notebook_path)
    return notebook_path


def main() -> None:
    round_days = {"train": [-2, -1], "validation": [0]}
    custom_results = run_custom_backtests(round_days)
    kevin_results = run_kevin_backtests(round_days)
    selection = build_selection_table(custom_results, kevin_results)
    chosen = choose_best_strategies(selection)
    charts_dir = ROOT / "round1_charts"
    for engine_name, results in [("custom_csv", custom_results), ("kevin_backtester", kevin_results)]:
        for product in PRODUCTS:
            for scope in round_days:
                plot_metric_curves(results, product, engine_name, scope, charts_dir)
    export_submission_file(chosen, ROOT / "round1_submission.py")
    export_notebook(ROOT / "round1_submission_workbook.ipynb", ROOT / "round1_submission.py", charts_dir)
    print(metrics_to_frame(custom_results).sort_values(["product", "scope", "pnl"], ascending=[True, True, False]))
    print(metrics_to_frame(kevin_results).sort_values(["product", "scope", "pnl"], ascending=[True, True, False]))
    print(selection)
    print(chosen)


if __name__ == "__main__":
    main()
