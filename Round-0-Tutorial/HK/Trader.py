import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


POSITION_LIMIT   = 80       
SKEW_TRIGGER     = 30        
HARD_HALT        = 60       


EM_FAIR_VALUE    = 10_000    
EM_QUOTE_NORMAL  = 8         
EM_QUOTE_NARROW  = 0         


TOM_EMA_SPAN     = 9         
TOM_ALPHA        = 2 / (TOM_EMA_SPAN + 1)
TOM_QUOTE_OFFSET = 5         
TOM_SKEW_OFFSET  = 7         
TOM_TIGHT_OFFSET = 3         
TOM_VELOCITY_WIN = 10       
TOM_VELOCITY_THR = 2.0       


class Trader:

    def bid(self):
        return 15

  
    def run(self, state: TradingState):
        
        trader_state = self._load_state(state.traderData)

        pos_em  = state.position.get('EMERALDS', 0)
        pos_tom = state.position.get('TOMATOES',  0)

        result: Dict[str, List[Order]] = {}

        if 'EMERALDS' in state.order_depths:
            result['EMERALDS'] = self._trade_emeralds(
                state.order_depths['EMERALDS'], pos_em
            )

        if 'TOMATOES' in state.order_depths:
            result['TOMATOES'], trader_state = self._trade_tomatoes(
                state.order_depths['TOMATOES'], pos_tom, trader_state
            )

        trader_data = json.dumps(trader_state)

        return result, 0, trader_data

    def _trade_emeralds(
        self, order_depth: OrderDepth, position: int
    ) -> List[Order]:
       
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid

        want_bid = True
        want_ask = True

        if position >= SKEW_TRIGGER:
            want_bid = False
        if position <= -SKEW_TRIGGER:
            want_ask = False

        if position >= HARD_HALT:
            want_bid = False
            want_ask = True   
        elif position <= -HARD_HALT:
            want_ask = False
            want_bid = True  

        if spread == 8:
        
            if best_ask == EM_FAIR_VALUE and want_bid:
                cap = self._remaining_capacity(position, 'buy')
                if cap > 0:
                    qty = min(cap, -order_depth.sell_orders[best_ask])
                    orders.append(Order('EMERALDS', EM_FAIR_VALUE, qty))

            elif best_bid == EM_FAIR_VALUE and want_ask:
                cap = self._remaining_capacity(position, 'sell')
                if cap > 0:
                    qty = min(cap, order_depth.buy_orders[best_bid])
                    orders.append(Order('EMERALDS', EM_FAIR_VALUE, -qty))

        else:
            bid_price = EM_FAIR_VALUE - EM_QUOTE_NORMAL   
            ask_price = EM_FAIR_VALUE + EM_QUOTE_NORMAL   

            if want_bid:
                cap = self._remaining_capacity(position, 'buy')
                if cap > 0:
                    l1_vol = order_depth.buy_orders.get(bid_price, 0)
                    qty = min(cap, max(l1_vol, 5))  
                    orders.append(Order('EMERALDS', bid_price, qty))

            if want_ask:
                cap = self._remaining_capacity(position, 'sell')
                if cap > 0:
                    l1_vol = abs(order_depth.sell_orders.get(ask_price, 0))
                    qty = min(cap, max(l1_vol, 5))
                    orders.append(Order('EMERALDS', ask_price, -qty))

        return orders

    def _trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        trader_state: dict,
    ):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2

        # Update EMA
        ema = trader_state.get('tom_ema', None)
        if ema is None:
            ema = mid  
        ema = TOM_ALPHA * mid + (1 - TOM_ALPHA) * ema
        trader_state['tom_ema'] = ema

        
        ema_history: list = trader_state.get('tom_ema_history', [])
        ema_history.append(ema)
        if len(ema_history) > TOM_VELOCITY_WIN + 1:
            ema_history = ema_history[-(TOM_VELOCITY_WIN + 1):]
        trader_state['tom_ema_history'] = ema_history

       
        in_breakout = False
        velocity    = 0.0
        if len(ema_history) >= TOM_VELOCITY_WIN:
            velocity    = ema_history[-1] - ema_history[-TOM_VELOCITY_WIN]
            in_breakout = abs(velocity) > TOM_VELOCITY_THR

        want_bid = True
        want_ask = True
        bid_offset = TOM_QUOTE_OFFSET  
        ask_offset = TOM_QUOTE_OFFSET 

        
        if position >= SKEW_TRIGGER:
            bid_offset = TOM_SKEW_OFFSET   
        elif position <= -SKEW_TRIGGER:
            ask_offset = TOM_SKEW_OFFSET

       
        if position >= HARD_HALT:
            want_bid   = False
            ask_offset = TOM_TIGHT_OFFSET 
        elif position <= -HARD_HALT:
            want_ask   = False
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

    def _load_state(self, trader_data: str) -> dict:

        if not trader_data:
            return {
                'tom_ema'        : None,
                'tom_ema_history': [],
            }
        try:
            return json.loads(trader_data)
        except (json.JSONDecodeError, TypeError):
            return {
                'tom_ema'        : None,
                'tom_ema_history': [],
            }
