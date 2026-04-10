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

TOM_LEAF_K       = 4.33
TOM_EXIT_BAND    = 0.8    

TOM_VEL_WIN      = 10
TOM_VEL_THR      = 2.0    

LATE_SKEW_POS    = 5     


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
                state.order_depths['TOMATOES'], pos_tm, trader_state, ts
            )

        return result, 0, json.dumps(trader_state)

    def _trade_emeralds(
        self, order_depth: OrderDepth, position: int, ts: int
    ) -> List[Order]:

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
                    orders.append(Order('EMERALDS', int(best_bid), -qty))
                return orders
            elif best_ask == EM_FV and position < 0:
                qty = min(-position, -order_depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order('EMERALDS', int(best_ask), qty))
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

        bid_cap = self._cap(position, 'buy')
        ask_cap = self._cap(position, 'sell')

        if want_bid and bid_cap > 0:
            orders.append(Order('EMERALDS', int(bid_price), min(bid_cap, 10)))
        if want_ask and ask_cap > 0:
            orders.append(Order('EMERALDS', int(ask_price), -min(ask_cap, 10)))

        return orders

    def _trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        trader_state: dict,
        ts: int,
    ):

        orders = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2.0

        if ts >= HARD_CLOSE_TS:
            orders = self._close_position(
                'TOMATOES', order_depth, position, best_bid, best_ask
            )
            return orders, trader_state

        ema = trader_state.get('tom_ema', None)
        if ema is None:
            ema = mid
        ema = TOM_ALPHA * mid + (1.0 - TOM_ALPHA) * ema
        trader_state['tom_ema'] = ema

        hist: list = trader_state.get('tom_ema_history', [])
        hist.append(ema)
        if len(hist) > TOM_VEL_WIN + 1:
            hist = hist[-(TOM_VEL_WIN + 1):]
        trader_state['tom_ema_history'] = hist

        velocity    = hist[-1] - hist[-TOM_VEL_WIN] if len(hist) >= TOM_VEL_WIN else 0.0
        in_breakout = abs(velocity) > TOM_VEL_THR


        residual = mid - ema 
        abs_res  = abs(residual)

        raw_target = TOM_LEAF_K * (abs_res ** 2)
        target_qty = min(int(raw_target), POSITION_LIMIT)

        edge_dir = 1 if residual < 0 else -1

        max_res  = trader_state.get('tom_max_res', 0.0)
        pos_sign = trader_state.get('tom_pos_sign', 0)

        current_sign = 1 if position > 3 else (-1 if position < -3 else 0)
        if current_sign != 0 and current_sign != pos_sign:
            max_res  = abs_res
            pos_sign = current_sign
        if abs_res > max_res:
            max_res = abs_res
        trader_state['tom_max_res']  = max_res
        trader_state['tom_pos_sign'] = pos_sign

        below_exit_band = abs_res < TOM_EXIT_BAND

        late_session = ts >= UNWIND_START_TS

        if in_breakout:
            desired = 0
        elif below_exit_band:
            desired = 0
        elif late_session:
            desired = edge_dir * min(target_qty, LATE_SKEW_POS)
        else:
            desired = edge_dir * target_qty

        delta = desired - position  

        bid_price = int(round(ema - TOM_QUOTE_OFFSET))
        ask_price = int(round(ema + TOM_QUOTE_OFFSET))

        if bid_price >= ask_price:
            bid_price = int(round(ema - TOM_QUOTE_OFFSET))
            ask_price = int(round(ema + TOM_QUOTE_OFFSET))

        abs_pos = abs(position)
        skew_trig_tom = LATE_SKEW_POS if late_session else 30
        if abs_pos >= skew_trig_tom:
            if position > 0:
                bid_price -= 2  
                ask_price -= 1  
            else:
                ask_price += 2   
                bid_price += 1   

        if delta > 0:
            bid_qty = min(delta, self._cap(position, 'buy'), 8)
            ask_qty = 0
        elif delta < 0:
            ask_qty = min(-delta, self._cap(position, 'sell'), 8)
            bid_qty = 0
        else:
            bid_qty = min(2, self._cap(position, 'buy'))
            ask_qty = min(2, self._cap(position, 'sell'))

        if bid_qty > 0:
            orders.append(Order('TOMATOES', bid_price, bid_qty))
        if ask_qty > 0:
            orders.append(Order('TOMATOES', ask_price, -ask_qty))

        return orders, trader_state


    def _close_position(
        self,
        symbol: str,
        order_depth: OrderDepth,
        position: int,
        best_bid: float,
        best_ask: float,
    ) -> List[Order]:

        orders = []
        if position > 0:
            qty = min(position, order_depth.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(symbol, int(best_bid), -qty))
        elif position < 0:
            qty = min(-position, -order_depth.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(symbol, int(best_ask), qty))
        return orders

    def _cap(self, position: int, side: str) -> int:

        if side == 'buy':
            return max(0, POSITION_LIMIT - position)
        return max(0, POSITION_LIMIT + position)

    def _load_state(self, trader_data: str) -> dict:

        defaults = {
            'tom_ema'        : None,
            'tom_ema_history': [],
            'tom_max_res'    : 0.0,
            'tom_pos_sign'   : 0,
        }
        if not trader_data:
            return defaults
        try:
            loaded = json.loads(trader_data)
            for k, v in defaults.items():
                if k not in loaded:
                    loaded[k] = v
            return loaded
        except (json.JSONDecodeError, TypeError):
            return defaults
