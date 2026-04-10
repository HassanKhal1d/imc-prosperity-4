import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

POSITION_LIMIT   = 80
SESSION_LENGTH   = 200000
UNWIND_START_TS  = 195000
HARD_CLOSE_TS    = 199000

EM_FV            = 10000
EM_INSIDE        = 7
EM_SKEW_TRIGGER  = 30
EM_SKEW_STEP     = 1
EM_HARD_HALT     = 60

TOM_EMA_SPAN     = 9
TOM_ALPHA        = 2 / (TOM_EMA_SPAN + 1)
TOM_QUOTE_OFFSET = 5
TOM_SKEW_OFFSET  = 7
TOM_TIGHT_OFFSET = 3
TOM_VELOCITY_WIN = 10
TOM_VELOCITY_THR = 2.0

LATE_SKEW_POS = 10


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
                state.order_depths['TOMATOES'], pos_tm, trader_state
            )

        return result, 0, json.dumps(trader_state)


    def _trade_emeralds(self, order_depth: OrderDepth, position: int, ts: int):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid

        if ts >= HARD_CLOSE_TS:
            return self._close_position(
                'EMERALDS', order_depth, position, best_bid, best_ask
            )

        skew_trig = LATE_SKEW_POS if ts >= UNWIND_START_TS else EM_SKEW_TRIGGER

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

        bid_price = EM_FV - EM_INSIDE
        ask_price = EM_FV + EM_INSIDE

        want_bid = True
        want_ask = True

        if position >= skew_trig:
            levels = (position - skew_trig) // EM_SKEW_STEP
            bid_price -= levels
        if position <= -skew_trig:
            levels = (-position - skew_trig) // EM_SKEW_STEP
            ask_price += levels

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


    def _trade_tomatoes(self, order_depth, position, trader_state, ts):
    orders: List[Order] = []

    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders, trader_state

    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())

    if ts >= HARD_CLOSE_TS:
        return self._close_position(
            'TOMATOES', order_depth, position, best_bid, best_ask
        ), trader_state

    mid = (best_bid + best_ask) / 2

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

    in_breakout = False
    velocity = 0.0
    if len(ema_history) >= TOM_VELOCITY_WIN:
        velocity = ema_history[-1] - ema_history[-TOM_VELOCITY_WIN]
        in_breakout = abs(velocity) > TOM_VELOCITY_THR

    want_bid = True
    want_ask = True
    bid_offset = TOM_QUOTE_OFFSET
    ask_offset = TOM_QUOTE_OFFSET

    if position >= EM_SKEW_TRIGGER:
        bid_offset = TOM_SKEW_OFFSET
    elif position <= -EM_SKEW_TRIGGER:
        ask_offset = TOM_SKEW_OFFSET

    if position >= EM_HARD_HALT:
        want_bid = False
        ask_offset = TOM_TIGHT_OFFSET
    elif position <= -EM_HARD_HALT:
        want_ask = False
        bid_offset = TOM_TIGHT_OFFSET

    if in_breakout:
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


    def _remaining_capacity(self, position: int, side: str) -> int:
        if side == 'buy':
            return max(0, POSITION_LIMIT - position)
        else:
            return max(0, POSITION_LIMIT + position)


    def _close_position(self, product, order_depth, position, best_bid, best_ask):
        orders = []
        if position > 0:
            qty = min(position, order_depth.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
        elif position < 0:
            qty = min(-position, -order_depth.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        return orders


    def _load_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {'tom_ema': None, 'tom_ema_history': []}
        try:
            return json.loads(trader_data)
        except:
            return {'tom_ema': None, 'tom_ema_history': []}

