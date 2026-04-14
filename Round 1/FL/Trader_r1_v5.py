import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# =============================================================================
# round 1 strategy v5: pepper buy-and-hold + osmium market-making
#
# changes from v4:
#   option 4 (micro price as quote centre) REVERTED.
#   results showed it lowered avg sell price from 10006 to 10005.27,
#   costing ~73 pnl. the micro price was shifting quotes downward
#   during sell pressure -- the exact wrong direction for this market.
#
# active changes vs v2 baseline:
#   - OSM_INSIDE = 6          (from v3: tighter quoting, more fills)
#   - OSM_NARROW_SPREAD_SKIP  (from v3: skip <14 tick spread ticks)
#   - OSM_SKEW_TRIGGER = 30   (from v4: earlier skew, residual near flat)
#   - regime-adaptive sizing  (from v4: calm=12, active=8, volatile=5)
#
# option 3 (regime sizing) is kept but flagged for recalibration.
# the vol thresholds (3.0/4.2) may be too sensitive -- 31% of ticks
# in the last run were classified volatile on a low-range day.
# recalibration will be done separately once we understand the pattern.
# =============================================================================

POSITION_LIMIT = 80

OSM_FV                 = 10000
OSM_INSIDE             = 6
OSM_SKEW_TRIGGER       = 30
OSM_SKEW_STEP          = 1
OSM_HARD_HALT          = 60
OSM_NARROW_SPREAD_SKIP = 14

OSM_VOL_WINDOW         = 20
OSM_VOL_CALM           = 3.0
OSM_VOL_ACTIVE         = 4.2

OSM_SIZE_CALM          = 12
OSM_SIZE_ACTIVE        = 8
OSM_SIZE_VOLATILE      = 5


class Trader:

    def run(self, state: TradingState):
        trader_state = self._load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        pos_pep = state.position.get('INTARIAN_PEPPER_ROOT', 0)
        pos_osm = state.position.get('ASH_COATED_OSMIUM', 0)

        if 'INTARIAN_PEPPER_ROOT' in state.order_depths:
            result['INTARIAN_PEPPER_ROOT'] = self._buy_and_hold_pepper(
                state.order_depths['INTARIAN_PEPPER_ROOT'], pos_pep
            )

        if 'ASH_COATED_OSMIUM' in state.order_depths:
            result['ASH_COATED_OSMIUM'], trader_state = self._trade_osmium(
                state.order_depths['ASH_COATED_OSMIUM'], pos_osm, trader_state
            )

        return result, 0, json.dumps(trader_state)


    def _buy_and_hold_pepper(self, order_depth: OrderDepth, position: int):
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


    def _trade_osmium(self, order_depth: OrderDepth, position: int,
                      trader_state: dict):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid
        mid      = (best_bid + best_ask) / 2.0

        # update rolling volatility
        mid_hist = trader_state.get('osm_mid_hist', [])
        mid_hist.append(mid)
        if len(mid_hist) > OSM_VOL_WINDOW + 2:
            mid_hist = mid_hist[-(OSM_VOL_WINDOW + 2):]
        trader_state['osm_mid_hist'] = mid_hist

        vol = 0.0
        if len(mid_hist) >= OSM_VOL_WINDOW + 1:
            rets = [mid_hist[i] - mid_hist[i-1] for i in range(1, len(mid_hist))]
            rets = rets[-OSM_VOL_WINDOW:]
            mean_r = sum(rets) / len(rets)
            vol = (sum((r - mean_r)**2 for r in rets) / len(rets)) ** 0.5

        # regime-adaptive sizing
        if vol < OSM_VOL_CALM:
            base_size = OSM_SIZE_CALM
        elif vol < OSM_VOL_ACTIVE:
            base_size = OSM_SIZE_ACTIVE
        else:
            base_size = OSM_SIZE_VOLATILE

        # spread=8 passive unwind
        if spread == 8:
            if best_bid == OSM_FV and position > 0:
                qty = min(position, order_depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order('ASH_COATED_OSMIUM', best_bid, -qty))
                return orders, trader_state
            elif best_ask == OSM_FV and position < 0:
                qty = min(-position, -order_depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order('ASH_COATED_OSMIUM', best_ask, qty))
                return orders, trader_state

        # narrow spread skip
        if spread < OSM_NARROW_SPREAD_SKIP:
            return orders, trader_state

        # quote around fixed fair value 10000
        bid_price = OSM_FV - OSM_INSIDE
        ask_price = OSM_FV + OSM_INSIDE

        want_bid = True
        want_ask = True

        # inventory skew
        if position >= OSM_SKEW_TRIGGER:
            excess = (position - OSM_SKEW_TRIGGER) // OSM_SKEW_STEP
            bid_price -= excess
        if position <= -OSM_SKEW_TRIGGER:
            excess = (-position - OSM_SKEW_TRIGGER) // OSM_SKEW_STEP
            ask_price += excess

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
            orders.append(Order('ASH_COATED_OSMIUM', int(bid_price),
                                min(bid_cap, base_size)))
        if want_ask and ask_cap > 0:
            orders.append(Order('ASH_COATED_OSMIUM', int(ask_price),
                                -min(ask_cap, base_size)))

        return orders, trader_state


    def _load_state(self, trader_data: str) -> dict:
        default = {'osm_mid_hist': []}
        if not trader_data:
            return default
        try:
            loaded = json.loads(trader_data)
            for k in default:
                if k not in loaded:
                    loaded[k] = default[k]
            return loaded
        except Exception:
            return default
