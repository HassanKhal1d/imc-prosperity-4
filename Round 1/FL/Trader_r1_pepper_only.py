import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# =============================================================================
# round 1 strategy: pepper root only -- buy and hold
#
# buys intarian pepper root up to the position limit of 80 and holds.
# no other products are traded. this isolates the pepper pnl cleanly
# so the trend-capture thesis can be confirmed before adding osmium.
# =============================================================================

POSITION_LIMIT = 80


class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        pos_pep = state.position.get('INTARIAN_PEPPER_ROOT', 0)

        if 'INTARIAN_PEPPER_ROOT' in state.order_depths:
            result['INTARIAN_PEPPER_ROOT'] = self._buy_and_hold(
                state.order_depths['INTARIAN_PEPPER_ROOT'], pos_pep
            )

        return result, 0, ''


    def _buy_and_hold(self, order_depth: OrderDepth, position: int):
        orders: List[Order] = []

        remaining = POSITION_LIMIT - position
        if remaining <= 0:
            return orders

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            available = -order_depth.sell_orders[best_ask]
            qty = min(remaining, available)
            if qty > 0:
                orders.append(Order('INTARIAN_PEPPER_ROOT', best_ask, qty))

        elif order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order('INTARIAN_PEPPER_ROOT', best_bid, remaining))

        return orders
