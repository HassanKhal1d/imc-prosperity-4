import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# ── Session constants ─────────────────────────────────────────────────────────
POSITION_LIMIT      = 80
SESSION_LENGTH      = 200000

# ── Unwind schedule (three phases replace the old two-phase binary close) ─────
#
#   PHASE 0 : normal trading                 ts <  190000
#   PHASE 1 : soft unwind (passive + skew)   ts >= 190000  and  ts < 196000
#   PHASE 2 : progressive cross              ts >= 196000  and  ts < 199500
#   PHASE 3 : final safety dump              ts >= 199500
#
# The key fixes vs v3:
#   - PHASE 1 now starts 9000 ticks earlier (195000 -> 190000).
#   - During PHASE 1 the MM quotes ONE side only: the side that reduces inventory.
#     This stops the algo from re-building position (v3 bug: bought +4 @ ts=195500).
#   - PHASE 2 crosses at mid-inside prices, NOT at the raw bid/ask.
#     EMERALDS short: BUY at FV-1 (9999) not at ASK (10008). Saves ~7 ticks each.
#     TOMATOES long:  SELL at EMA+1, not at BID-6.
#   - PHASE 3 is the last-resort dump, limited to the final 5 ticks (500 ts),
#     rather than the final 100 ticks (10000 ts) in v3.
#
SOFT_UNWIND_TS      = 190000   # Phase 1 start: one-sided passive quoting
PROGRESSIVE_TS      = 196000   # Phase 2 start: cross spread at cheap prices
HARD_CLOSE_TS       = 199500   # Phase 3 start: last-resort market cross

# ── Emeralds parameters ───────────────────────────────────────────────────────
EM_FV               = 10000
EM_INSIDE           = 7        # half-spread inside existing book
EM_SKEW_TRIGGER     = 30       # position above this triggers bid/ask skew
EM_SKEW_STEP        = 1        # price tick drop per extra unit of position
EM_HARD_HALT        = 60       # stop quoting the risk-adding side entirely
EM_PROG_CROSS_SLIP  = 1        # ticks inside FV we cross at during Phase 2
#   Phase 2 logic for EMERALDS:
#     short: BUY at EM_FV - EM_PROG_CROSS_SLIP  (9999 vs 10008 in v3)
#     long:  SELL at EM_FV + EM_PROG_CROSS_SLIP (10001 vs 9992 in v3)
EM_PROG_CHUNK       = 3        # max units to cross per tick in Phase 2

# ── Tomatoes parameters ───────────────────────────────────────────────────────
TOM_EMA_SPAN        = 9
TOM_ALPHA           = 2 / (TOM_EMA_SPAN + 1)
TOM_QUOTE_OFFSET    = 5
TOM_SKEW_OFFSET     = 7
TOM_TIGHT_OFFSET    = 3
TOM_VELOCITY_WIN    = 10
TOM_VELOCITY_THR    = 2.0
TOM_PROG_CROSS_SLIP = 1        # ticks inside EMA we cross at during Phase 2
TOM_PROG_CHUNK      = 3        # max units to cross per tick in Phase 2

# ── Soft-unwind position threshold ───────────────────────────────────────────
#   During Phase 1 we stop quoting the inventory-adding side once abs(pos) > this.
LATE_SKEW_POS       = 5        # tightened from v3's 10


class Trader:

    def run(self, state: TradingState):
        trader_state = self._load_state(state.traderData)
        ts     = state.timestamp
        pos_em = state.position.get('EMERALDS', 0)
        pos_tm = state.position.get('TOMATOES',  0)

        result: Dict[str, List[Order]] = {}

        if 'EMERALDS' in state.order_depths:
            result['EMERALDS'] = self._trade_emeralds(
                state.order_depths['EMERALDS'], pos_em, ts
            )

        if 'TOMATOES' in state.order_depths:
            result['TOMATOES'], trader_state = self._trade_tomatoes(
                state.order_depths['TOMATOES'], pos_tm, ts, trader_state
            )

        return result, 0, json.dumps(trader_state)


    # ── Emeralds ──────────────────────────────────────────────────────────────

    def _trade_emeralds(self, order_depth: OrderDepth, position: int, ts: int):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid

        # ── Phase 3: last-resort market cross ─────────────────────────────────
        if ts >= HARD_CLOSE_TS:
            return self._hard_close('EMERALDS', order_depth, position,
                                    best_bid, best_ask)

        # ── Phase 2: progressive cross at near-FV prices ──────────────────────
        if ts >= PROGRESSIVE_TS and position != 0:
            return self._progressive_close_emeralds(
                order_depth, position, best_bid, best_ask
            )

        # ── Phase 1: soft unwind -- quote one side only ───────────────────────
        in_soft_unwind = ts >= SOFT_UNWIND_TS

        # Opportunistic fill when spread compresses to 8 (bid or ask touches FV)
        if spread == 8:
            if best_bid == EM_FV and position > 0:
                qty = min(position, order_depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order('EMERALDS', best_bid, -qty))
                return orders
            elif best_ask == EM_FV and position < 0:
                qty = min(-position, -order_depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order('EMERALDS', best_ask, qty))
                return orders

        # ── Normal / Phase 1 passive quoting ─────────────────────────────────
        skew_trig  = LATE_SKEW_POS if in_soft_unwind else EM_SKEW_TRIGGER
        bid_price  = EM_FV - EM_INSIDE
        ask_price  = EM_FV + EM_INSIDE
        want_bid   = True
        want_ask   = True

        # In soft-unwind mode, only quote the side that REDUCES inventory
        if in_soft_unwind:
            if position > 0:
                want_bid = False     # long: do not add more longs
            elif position < 0:
                want_ask = False     # short: do not add more shorts

        # Inventory skew: push quotes away from the risky side
        if position >= skew_trig:
            levels    = (position - skew_trig) // EM_SKEW_STEP
            bid_price -= levels
        if position <= -skew_trig:
            levels    = (-position - skew_trig) // EM_SKEW_STEP
            ask_price += levels

        # Hard halt: stop quoting the risk-adding side at extreme positions
        if position >= EM_HARD_HALT:
            want_bid = False
        if position <= -EM_HARD_HALT:
            want_ask = False

        if bid_price >= ask_price:
            bid_price = EM_FV - EM_INSIDE
            ask_price = EM_FV + EM_INSIDE

        bid_cap = self._remaining_capacity(position, 'buy')
        ask_cap = self._remaining_capacity(position, 'sell')

        if want_bid and bid_cap > 0:
            orders.append(Order('EMERALDS', int(bid_price), min(bid_cap, 10)))

        if want_ask and ask_cap > 0:
            orders.append(Order('EMERALDS', int(ask_price), -min(ask_cap, 10)))

        return orders


    def _progressive_close_emeralds(self, order_depth, position,
                                     best_bid, best_ask):
        """
        Cross the spread cheaply, a few units at a time.

        Short (position < 0): post a BUY at EM_FV - EM_PROG_CROSS_SLIP.
          In Prosperity you fill at YOUR price, so posting 9999 means you pay
          9999 rather than 10008 -- saving 9 ticks vs the raw hard-close.
          The order will fill against any sell-side market trade priced <= 9999,
          or against any resting ask <= 9999.

        Long (position > 0): post a SELL at EM_FV + EM_PROG_CROSS_SLIP (10001).
          This is only 1 tick above FV instead of 8 ticks above.

        We limit chunk size so we spread cost across multiple ticks if needed.
        """
        orders = []
        chunk = min(EM_PROG_CHUNK, abs(position),
                    self._remaining_capacity(position,
                                             'sell' if position > 0 else 'buy'))
        if chunk <= 0:
            return orders

        if position < 0:
            buy_price = EM_FV - EM_PROG_CROSS_SLIP   # 9999: well inside spread
            if buy_price >= best_ask:
                # spread has compressed, take the ask directly
                buy_price = best_ask
            orders.append(Order('EMERALDS', int(buy_price), chunk))
        else:
            sell_price = EM_FV + EM_PROG_CROSS_SLIP  # 10001: just above FV
            if sell_price <= best_bid:
                sell_price = best_bid
            orders.append(Order('EMERALDS', int(sell_price), -chunk))

        return orders


    # ── Tomatoes ──────────────────────────────────────────────────────────────

    def _trade_tomatoes(self, order_depth, position, ts, trader_state):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2

        # Always update EMA so state stays fresh for the next tick
        ema = trader_state.get('tom_ema', None)
        if ema is None:
            ema = mid
        ema = TOM_ALPHA * mid + (1 - TOM_ALPHA) * ema
        trader_state['tom_ema'] = ema

        ema_history = trader_state.get('tom_ema_history', [])
        ema_history.append(ema)
        if len(ema_history) > TOM_VELOCITY_WIN + 1:
            ema_history = ema_history[-(TOM_VELOCITY_WIN + 1):]
        trader_state['tom_ema_history'] = ema_history

        # ── Phase 3: last-resort market cross ─────────────────────────────────
        if ts >= HARD_CLOSE_TS:
            return (self._hard_close('TOMATOES', order_depth, position,
                                     best_bid, best_ask),
                    trader_state)

        # ── Phase 2: progressive cross at near-EMA prices ─────────────────────
        if ts >= PROGRESSIVE_TS and position != 0:
            return (self._progressive_close_tomatoes(
                        order_depth, position, ema, best_bid, best_ask),
                    trader_state)

        # ── Phase 1: soft unwind -- one-sided quoting ─────────────────────────
        in_soft_unwind = ts >= SOFT_UNWIND_TS

        # Velocity / breakout detection
        velocity    = 0.0
        in_breakout = False
        if len(ema_history) >= TOM_VELOCITY_WIN:
            velocity    = ema_history[-1] - ema_history[-TOM_VELOCITY_WIN]
            in_breakout = abs(velocity) > TOM_VELOCITY_THR

        want_bid   = True
        want_ask   = True
        bid_offset = TOM_QUOTE_OFFSET
        ask_offset = TOM_QUOTE_OFFSET

        # In soft-unwind mode: only quote the side that reduces inventory,
        # and only if position is non-trivial.
        if in_soft_unwind:
            if position > 0:
                want_bid = False
            elif position < 0:
                want_ask = False

        # Normal inventory skew (applies outside soft-unwind too)
        if position >= EM_SKEW_TRIGGER:
            bid_offset = TOM_SKEW_OFFSET
        elif position <= -EM_SKEW_TRIGGER:
            ask_offset = TOM_SKEW_OFFSET

        if position >= EM_HARD_HALT:
            want_bid   = False
            ask_offset = TOM_TIGHT_OFFSET
        elif position <= -EM_HARD_HALT:
            want_ask   = False
            bid_offset = TOM_TIGHT_OFFSET

        # Breakout suppression (only outside soft-unwind to avoid interference)
        if not in_soft_unwind and in_breakout:
            if velocity > 0:
                if position > 0:
                    want_bid = False
            else:
                if position < 0:
                    want_ask = False
                want_bid = False

        bid_price = int(round(ema - bid_offset))
        ask_price = int(round(ema + ask_offset))

        if bid_price >= ask_price:
            bid_price = int(round(ema - TOM_QUOTE_OFFSET))
            ask_price = int(round(ema + TOM_QUOTE_OFFSET))

        if want_bid:
            cap = self._remaining_capacity(position, 'buy')
            if cap > 0:
                orders.append(Order('TOMATOES', bid_price, min(cap, 5)))

        if want_ask:
            cap = self._remaining_capacity(position, 'sell')
            if cap > 0:
                orders.append(Order('TOMATOES', ask_price, -min(cap, 5)))

        return orders, trader_state


    def _progressive_close_tomatoes(self, order_depth, position, ema,
                                     best_bid, best_ask):
        """
        Cross the spread at near-EMA prices rather than the raw bid/ask.

        Long (position > 0): SELL at round(ema) + TOM_PROG_CROSS_SLIP.
          EMA tracks the true fair value. Selling at EMA+1 costs ~1 tick vs
          the half-spread of 6-7 ticks charged by the hard close.

        Short (position < 0): BUY at round(ema) - TOM_PROG_CROSS_SLIP.
        """
        orders = []
        ema_int = int(round(ema))
        chunk = min(TOM_PROG_CHUNK, abs(position),
                    self._remaining_capacity(position,
                                             'sell' if position > 0 else 'buy'))
        if chunk <= 0:
            return orders

        if position > 0:
            sell_price = ema_int + TOM_PROG_CROSS_SLIP
            if sell_price <= best_bid:
                sell_price = best_bid
            orders.append(Order('TOMATOES', int(sell_price), -chunk))
        else:
            buy_price = ema_int - TOM_PROG_CROSS_SLIP
            if buy_price >= best_ask:
                buy_price = best_ask
            orders.append(Order('TOMATOES', int(buy_price), chunk))

        return orders


    # ── Shared helpers ────────────────────────────────────────────────────────

    def _hard_close(self, product, order_depth, position, best_bid, best_ask):
        """
        Last-resort full market cross. Only fires in the final 5 ticks.
        In v3 this fired over the final 100 ticks, leading to large positions
        being dumped at the worst possible price.
        """
        orders = []
        if position > 0:
            qty = min(position, order_depth.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        elif position < 0:
            qty = min(-position, abs(order_depth.sell_orders.get(best_ask, 0)))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        return orders


    def _remaining_capacity(self, position: int, side: str) -> int:
        if side == 'buy':
            return max(0, POSITION_LIMIT - position)
        else:
            return max(0, POSITION_LIMIT + position)


    def _load_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {'tom_ema': None, 'tom_ema_history': []}
        try:
            return json.loads(trader_data)
        except Exception:
            return {'tom_ema': None, 'tom_ema_history': []}
