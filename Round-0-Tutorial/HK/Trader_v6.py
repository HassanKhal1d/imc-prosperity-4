import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# global constants: all tunable parameters

POSITION_LIMIT         = 80

# emeralds (unchanged)
EM_FV                  = 10000
EM_INSIDE              = 7
EM_SKEW_TRIGGER        = 40
EM_SKEW_STEP           = 1
EM_HARD_HALT           = 60

# tomatoes: fair value tracking (used for skew direction only, not for quoting)
TOM_EMA_SPAN           = 7
TOM_ALPHA              = 2 / (TOM_EMA_SPAN + 1)

# tomatoes: volatility estimation
TOM_VOL_WINDOW         = 20
TOM_VOL_CALM_THRESH    = 1.5
TOM_VOL_ACTIVE_THRESH  = 2.5

# tomatoes: regime-adaptive order sizing
TOM_SIZE_CALM          = 6
TOM_SIZE_ACTIVE        = 5
TOM_SIZE_VOLATILE      = 3

# tomatoes: book-anchored quoting offset from best bid/ask
# offset=1 means we quote at best_bid+1 and best_ask-1
TOM_QUOTE_OFFSET       = 1

# tomatoes: inventory management (tomato-specific, not reusing emerald params)
TOM_SKEW_TRIGGER       = 30
TOM_SKEW_STEP          = 1
TOM_HARD_HALT          = 50
TOM_UNWIND_OFFSET      = 3

# tomatoes: velocity breakout filter
TOM_VELOCITY_WIN       = 10
TOM_VELOCITY_THR       = 3.5

# tomatoes: inventory skew proportional factor
# for every unit of position beyond skew_trigger, shift quote by this much
TOM_SKEW_QUOTE_SHIFT   = 1


class Trader:

    def run(self, state: TradingState):
        trader_state = self._load_state(state.traderData)
        pos_em = state.position.get('EMERALDS', 0)
        pos_tm = state.position.get('TOMATOES', 0)

        result: Dict[str, List[Order]] = {}

        if 'EMERALDS' in state.order_depths:
            result['EMERALDS'] = self._trade_emeralds(
                state.order_depths['EMERALDS'], pos_em
            )

        if 'TOMATOES' in state.order_depths:
            result['TOMATOES'], trader_state = self._trade_tomatoes(
                state.order_depths['TOMATOES'], pos_tm, trader_state
            )

        return result, 0, json.dumps(trader_state)


    # emeralds: unchanged from v4

    def _trade_emeralds(self, order_depth: OrderDepth, position: int):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid

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

        if position >= EM_SKEW_TRIGGER:
            levels = (position - EM_SKEW_TRIGGER) // EM_SKEW_STEP
            bid_price -= levels
        if position <= -EM_SKEW_TRIGGER:
            levels = (-position - EM_SKEW_TRIGGER) // EM_SKEW_STEP
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


    # tomatoes: v6 redesign
    # core idea: keep v4 book-anchored quoting, add v5 regime and inventory infra

    def _trade_tomatoes(self, order_depth, position, trader_state):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2.0

        # update ema (used for skew direction and unwind pricing, not for quote center)
        ema = trader_state.get('tom_ema', None)
        if ema is None:
            ema = mid
        ema = TOM_ALPHA * mid + (1 - TOM_ALPHA) * ema
        trader_state['tom_ema'] = ema

        # update rolling volatility
        mid_history = trader_state.get('tom_mid_history', [])
        mid_history.append(mid)
        if len(mid_history) > TOM_VOL_WINDOW + 2:
            mid_history = mid_history[-(TOM_VOL_WINDOW + 2):]
        trader_state['tom_mid_history'] = mid_history

        vol = 0.0
        if len(mid_history) >= TOM_VOL_WINDOW + 1:
            returns = []
            for i in range(1, len(mid_history)):
                returns.append(mid_history[i] - mid_history[i - 1])
            returns = returns[-TOM_VOL_WINDOW:]
            if returns:
                mean_r = sum(returns) / len(returns)
                var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                vol = var_r ** 0.5
        trader_state['tom_vol'] = vol

        # update ema history for velocity filter
        ema_history = trader_state.get('tom_ema_history', [])
        ema_history.append(ema)
        if len(ema_history) > TOM_VELOCITY_WIN + 1:
            ema_history = ema_history[-(TOM_VELOCITY_WIN + 1):]
        trader_state['tom_ema_history'] = ema_history

        # classify volatility regime
        if vol < TOM_VOL_CALM_THRESH:
            regime = 'calm'
        elif vol < TOM_VOL_ACTIVE_THRESH:
            regime = 'active'
        else:
            regime = 'volatile'

        # velocity breakout detection (raised threshold vs v4 to reduce false signals)
        in_breakout = False
        velocity = 0.0
        if len(ema_history) >= TOM_VELOCITY_WIN:
            velocity = ema_history[-1] - ema_history[-TOM_VELOCITY_WIN]
            in_breakout = abs(velocity) > TOM_VELOCITY_THR

        # select order size based on regime
        if regime == 'calm':
            base_size = TOM_SIZE_CALM
        elif regime == 'active':
            base_size = TOM_SIZE_ACTIVE
        else:
            base_size = TOM_SIZE_VOLATILE

        # book-anchored quoting: always price relative to current best bid/ask
        # this guarantees we are always at best price in the book
        bid_price = best_bid + TOM_QUOTE_OFFSET
        ask_price = best_ask - TOM_QUOTE_OFFSET

        want_bid = True
        want_ask = True

        # inventory skew: shift quotes proportionally when position grows large
        # this gradually discourages adding to the overweight side
        if position > TOM_SKEW_TRIGGER:
            excess = position - TOM_SKEW_TRIGGER
            shift = excess * TOM_SKEW_QUOTE_SHIFT
            bid_price -= shift
        elif position < -TOM_SKEW_TRIGGER:
            excess = -position - TOM_SKEW_TRIGGER
            shift = excess * TOM_SKEW_QUOTE_SHIFT
            ask_price += shift

        # hard halt: stop quoting the overweight side, aggressively unwind
        if position >= TOM_HARD_HALT:
            want_bid = False
            ask_price = int(round(ema - TOM_UNWIND_OFFSET))
        elif position <= -TOM_HARD_HALT:
            want_ask = False
            bid_price = int(round(ema + TOM_UNWIND_OFFSET))

        # velocity filter: only suppress the side where position aligns with trend
        # v4 was too aggressive here (killed both sides sometimes)
        # v5 fixed this; v6 keeps the fix
        if in_breakout:
            if velocity > 0 and position > 0:
                want_bid = False
            elif velocity < 0 and position < 0:
                want_ask = False

        # sanity: prevent crossed quotes
        if bid_price >= ask_price:
            bid_price = best_bid + TOM_QUOTE_OFFSET
            ask_price = best_ask - TOM_QUOTE_OFFSET

        # place orders
        bid_cap = self._remaining_capacity(position, 'buy')
        ask_cap = self._remaining_capacity(position, 'sell')

        if want_bid and bid_cap > 0:
            qty = min(bid_cap, base_size)
            orders.append(Order('TOMATOES', int(bid_price), qty))

        if want_ask and ask_cap > 0:
            qty = min(ask_cap, base_size)
            orders.append(Order('TOMATOES', int(ask_price), -qty))

        return orders, trader_state


    def _remaining_capacity(self, position: int, side: str) -> int:
        if side == 'buy':
            return max(0, POSITION_LIMIT - position)
        else:
            return max(0, POSITION_LIMIT + position)


    def _load_state(self, trader_data: str) -> dict:
        default = {
            'tom_ema': None,
            'tom_ema_history': [],
            'tom_mid_history': [],
            'tom_vol': 0.0,
        }
        if not trader_data:
            return default
        try:
            loaded = json.loads(trader_data)
            for key in default:
                if key not in loaded:
                    loaded[key] = default[key]
            return loaded
        except Exception:
            return default
