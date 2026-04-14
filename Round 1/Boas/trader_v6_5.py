import json
from sqlite3 import Time
from typing import List, Dict

# =============================================================================
# trader v6.5
#
# round 1 conclusion:
#   - intarian pepper root is best traded as deterministic trend capture, not
#     symmetric market making. the trend is structural and the best live result
#     from the current research stack is to accumulate long inventory early,
#     add more aggressively on dips versus the trend line, and then hold that
#     inventory instead of fighting the drift.
#   - ash coated osmium remains a classic fixed-anchor mean reversion product.
#     the current best live choice is still simple anchor market making around
#     10000 with inventory skew and a hard stop on further accumulation when
#     inventory gets too large.
#
# version 6.5 implements the currently selected live pair:
#   - pepper: pepper_dip_accumulator
#   - osmium: fixed_anchor_mm
#
# pepper logic:
#   1. track a per-day opening price and apply the correct trend slope of
#      1000 units per 10000 ticks = 0.1 per tick.
#   2. compute fair value = day_open + structural trend drift.
#   3. if live mid is at or below fair+2, target max long inventory (80).
#      if price is stretched above fair+2, target a smaller holding (60).
#   4. cross the ask slightly to ensure the position is actually built.
#   5. once target inventory is reached, stop quoting bids and leave only a
#      very wide hold quote. this preserves the directional exposure.
#
# osmium logic:
#   1. fixed anchor at 10000.
#   2. quote around anchor with half-spread 8.
#   3. inventory skew discourages further accumulation on the crowded side.
#   4. hard halt stops new buying above +70 and new selling below -70.
# =============================================================================

POSITION_LIMIT = 80

# pepper: deterministic trend model
PEP_TREND_PER_TICK      = 1000.0 / 10000.0
PEP_LONG_TARGET_FULL    = 80
PEP_LONG_TARGET_REDUCED = 60
PEP_DIP_THRESHOLD       = 2.0
PEP_ENTRY_CROSS         = 1
PEP_ENTRY_SIZE          = 16
PEP_HOLD_WIDE           = 40

# osmium: fixed-anchor market making
OSM_ANCHOR              = 10000
OSM_HALF_SPREAD         = 8
OSM_BASE_SIZE           = 6
OSM_SKEW_TRIGGER        = 40
OSM_SKEW_STEP           = 1
OSM_HARD_HALT           = 70

class Order:

    def __init__(self, symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

class TradingState(object):

    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings,
                 order_depths,
                 own_trades,
                 market_trades,
                 position,
                 observations):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
    
    
class OrderDepth:

    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class Trader:

    def run(self, state: TradingState):
        trader_state = self._load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        if 'INTARIAN_PEPPER_ROOT' in state.order_depths:
            result['INTARIAN_PEPPER_ROOT'], trader_state = self._trade_pepper(
                state.order_depths['INTARIAN_PEPPER_ROOT'],
                state.position.get('INTARIAN_PEPPER_ROOT', 0),
                state.timestamp,
                trader_state,
            )

        if 'ASH_COATED_OSMIUM' in state.order_depths:
            result['ASH_COATED_OSMIUM'] = self._trade_osmium(
                state.order_depths['ASH_COATED_OSMIUM'],
                state.position.get('ASH_COATED_OSMIUM', 0),
            )

        return result, 0, json.dumps(trader_state)


    def _trade_pepper(self, order_depth: OrderDepth, position: int, timestamp: int, trader_state: dict):
        orders: List[Order] = []

        if not order_depth.sell_orders and not order_depth.buy_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        mid = self._mid_price(best_bid, best_ask, 11500.0)

        # New day detection: Prosperity timestamps restart from 0 each day.
        last_ts = trader_state.get('pep_last_timestamp')
        if trader_state['pep_day_open'] is None or last_ts is None or timestamp < last_ts:
            trader_state['pep_day_open'] = mid
            trader_state['pep_day_start_ts'] = timestamp

        trader_state['pep_last_timestamp'] = timestamp

        fair = self._pepper_fair_value(timestamp, trader_state)
        deviation = mid - fair
        target = PEP_LONG_TARGET_FULL if deviation <= PEP_DIP_THRESHOLD else PEP_LONG_TARGET_REDUCED

        bid_cap = self._remaining_capacity(position, 'buy')
        ask_cap = self._remaining_capacity(position, 'sell')

        if best_ask is not None and position < target and bid_cap > 0:
            buy_size = min(PEP_ENTRY_SIZE, target - position, bid_cap)
            if buy_size > 0:
                orders.append(Order('INTARIAN_PEPPER_ROOT', int(best_ask + PEP_ENTRY_CROSS), buy_size))

        # Keep a very wide hold quote only after the target inventory is built.
        # This mirrors the current best backtested logic: own the trend rather
        # than provide normal symmetric liquidity against it.
        if best_ask is not None and ask_cap > 0 and position >= target:
            hold_ask = int(round(fair + PEP_HOLD_WIDE))
            orders.append(Order('INTARIAN_PEPPER_ROOT', hold_ask, -min(ask_cap, 1)))

        return orders, trader_state


    def _trade_osmium(self, order_depth: OrderDepth, position: int):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        bid_price = OSM_ANCHOR - OSM_HALF_SPREAD
        ask_price = OSM_ANCHOR + OSM_HALF_SPREAD

        want_bid = True
        want_ask = True

        if position >= OSM_SKEW_TRIGGER:
            levels = (position - OSM_SKEW_TRIGGER) // OSM_SKEW_STEP
            bid_price -= levels
        elif position <= -OSM_SKEW_TRIGGER:
            levels = (-position - OSM_SKEW_TRIGGER) // OSM_SKEW_STEP
            ask_price += levels

        if position >= OSM_HARD_HALT:
            want_bid = False
        elif position <= -OSM_HARD_HALT:
            want_ask = False

        if bid_price >= ask_price:
            bid_price = OSM_ANCHOR - OSM_HALF_SPREAD
            ask_price = OSM_ANCHOR + OSM_HALF_SPREAD

        bid_cap = self._remaining_capacity(position, 'buy')
        ask_cap = self._remaining_capacity(position, 'sell')

        if want_bid and bid_cap > 0:
            orders.append(Order('ASH_COATED_OSMIUM', int(bid_price), min(bid_cap, OSM_BASE_SIZE)))

        if want_ask and ask_cap > 0:
            orders.append(Order('ASH_COATED_OSMIUM', int(ask_price), -min(ask_cap, OSM_BASE_SIZE)))

        return orders


    def _pepper_fair_value(self, timestamp: int, trader_state: dict) -> float:
        ticks_elapsed = (timestamp - trader_state['pep_day_start_ts']) / 100.0
        return float(trader_state['pep_day_open'] + PEP_TREND_PER_TICK * ticks_elapsed)


    def _mid_price(self, best_bid, best_ask, fallback: float) -> float:
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return fallback


    def _remaining_capacity(self, position: int, side: str) -> int:
        if side == 'buy':
            return max(0, POSITION_LIMIT - position)
        return max(0, POSITION_LIMIT + position)


    def _load_state(self, trader_data: str) -> dict:
        default = {
            'pep_day_open': None,
            'pep_day_start_ts': 0,
            'pep_last_timestamp': None,
        }
        if not trader_data:
            return default
        try:
            loaded = json.loads(trader_data)
            for key, value in default.items():
                if key not in loaded:
                    loaded[key] = value
            return loaded
        except Exception:
            return default
