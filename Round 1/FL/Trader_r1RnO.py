import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# =============================================================================
# round 1 strategy: buy and hold intarian pepper root
#
# rationale:
#   pepper root rises by exactly 1000 units per day (slope 0.001/timestamp).
#   buying the maximum position (80 units) at the start and holding it
#   captures ~984 units of pnl per unit per day after spread costs.
#   at 80 units that is ~78,720 per day -- far exceeding the 200,000 target
#   across the round.
#
#   osmium is mean-reverting around 10000 (identical to emeralds in round 0).
#   we run the same proven emerald market-making logic on it to earn
#   additional spread income on top of the pepper hold.
#
# pepper strategy: buy aggressively at every tick until position = 80.
#   once full, do nothing -- just hold and let the trend do the work.
#   we never sell pepper under any circumstances.
#
# osmium strategy: identical to the round 0 emerald approach.
#   quote at fv +/- 7 ticks. skew when inventory builds. halt at 60.
# =============================================================================

POSITION_LIMIT = 80

# osmium constants (same structure as round 0 emeralds)
OSM_FV           = 10000
OSM_INSIDE       = 7
OSM_SKEW_TRIGGER = 40
OSM_SKEW_STEP    = 1
OSM_HARD_HALT    = 60


class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        pos_pep = state.position.get('INTARIAN_PEPPER_ROOT', 0)
        pos_osm = state.position.get('ASH_COATED_OSMIUM', 0)

        if 'INTARIAN_PEPPER_ROOT' in state.order_depths:
            result['INTARIAN_PEPPER_ROOT'] = self._buy_and_hold_pepper(
                state.order_depths['INTARIAN_PEPPER_ROOT'], pos_pep
            )

        if 'ASH_COATED_OSMIUM' in state.order_depths:
            result['ASH_COATED_OSMIUM'] = self._trade_osmium(
                state.order_depths['ASH_COATED_OSMIUM'], pos_osm
            )

        return result, 0, ''


    def _buy_and_hold_pepper(self, order_depth: OrderDepth, position: int):
        """
        buy as many units of pepper as possible up to the position limit.
        once at limit, place no orders -- just hold.

        we buy by hitting the best ask. if the ask side is empty (rare,
        ~3.7% of ticks) we place a passive bid at the best bid price to
        catch any sellers. we never sell pepper under any circumstances.
        """
        orders: List[Order] = []

        remaining = POSITION_LIMIT - position
        if remaining <= 0:
            # already at limit -- hold, do nothing
            return orders

        if order_depth.sell_orders:
            # take liquidity: hit the best ask for as many units as we can
            best_ask = min(order_depth.sell_orders.keys())
            available = -order_depth.sell_orders[best_ask]  # negative qty in sell orders
            qty = min(remaining, available)
            if qty > 0:
                orders.append(Order('INTARIAN_PEPPER_ROOT', best_ask, qty))

        elif order_depth.buy_orders:
            # no ask side -- place a passive bid at best bid to signal intent
            # this tick is rare; we will catch up on the next tick
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order('INTARIAN_PEPPER_ROOT', best_bid, remaining))

        return orders


    def _trade_osmium(self, order_depth: OrderDepth, position: int):
        """
        market-make ash coated osmium around its fixed fair value of 10000.
        logic is identical to the proven round 0 emerald approach.
        osmium is stationary with std ~5.3 and mean ~10000 across all 3
        historical days, so the same fixed-anchor quoting works directly.
        """
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid

        # spread=8 passive unwind: identical to emerald special case
        if spread == 8:
            if best_bid == OSM_FV and position > 0:
                qty = min(position, order_depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order('ASH_COATED_OSMIUM', best_bid, -qty))
                return orders
            elif best_ask == OSM_FV and position < 0:
                qty = min(-position, -order_depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order('ASH_COATED_OSMIUM', best_ask, qty))
                return orders

        bid_price = OSM_FV - OSM_INSIDE
        ask_price = OSM_FV + OSM_INSIDE

        want_bid = True
        want_ask = True

        # inventory skew
        if position >= OSM_SKEW_TRIGGER:
            levels = (position - OSM_SKEW_TRIGGER) // OSM_SKEW_STEP
            bid_price -= levels
        if position <= -OSM_SKEW_TRIGGER:
            levels = (-position - OSM_SKEW_TRIGGER) // OSM_SKEW_STEP
            ask_price += levels

        # hard halt
        if position >= OSM_HARD_HALT:
            want_bid = False
        if position <= -OSM_HARD_HALT:
            want_ask = False

        if bid_price >= ask_price:
            bid_price = OSM_FV - OSM_INSIDE
            ask_price = OSM_FV + OSM_INSIDE

        bid_cap = max(0, POSITION_LIMIT - position)
        ask_cap = max(0, POSITION_LIMIT + position)

        if want_bid and bid_cap > 0:
            orders.append(Order('ASH_COATED_OSMIUM', int(bid_price), min(bid_cap, 10)))

        if want_ask and ask_cap > 0:
            orders.append(Order('ASH_COATED_OSMIUM', int(ask_price), -min(ask_cap, 10)))

        return orders
