import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# ============================================================================
# round 1 strategy v1-HK: osmium take-and-make + pepper smart accumulate
#
# pnl target: 11744 xirecs (competition best)
# current v5 baseline: 8962 xirecs (osmium 1579 + pepper 7383)
#
# key changes from v5 (each is a falsifiable hypothesis):
#
# [H1] osmium: take mispriced levels (ask < fv, bid > fv)
#   evidence: sample data shows 3000-3800 xirecs/day from taking alone
#   v5 gap: v5 never crosses the spread to take edge. this is the single
#   largest source of missed alpha. the narrow spread ticks that v5 skips
#   (spread < 14) are almost always mispriced and contain the best edge.
#   falsifiable: disable TAKE_ENABLED and compare pnl
#
# [H2] osmium: remove narrow spread skip filter
#   evidence: 7.7% of ticks have spread < 14. of those, 56% are mispriced.
#   v5 skips ALL of these. removing the filter lets us take the edge.
#   falsifiable: set SKIP_NARROW_SPREAD = True and compare pnl
#
# [H3] osmium: reservation price inventory skew (avellaneda-stoikov)
#   evidence: v5 only skews the bid side and leaves ask at fv+6 regardless
#   of position. this means at position 60, the ask is still at 10006
#   and inventory clears very slowly. the A-S approach shifts BOTH sides,
#   making the ask much closer to fv when long, clearing inventory faster.
#   falsifiable: set GAMMA = 0 to revert to symmetric quoting
#
# [H4] osmium: take at fv for inventory clearing
#   evidence: when position is +40 and bid=10000 appears, selling at fv
#   clears 40 units at break-even. v5 only does this for spread=8.
#   falsifiable: disable TAKE_AT_FV and compare drawdown
#
# [H5] pepper: passive bidding at best_bid + 1
#   evidence: 50.9% of pepper trades are sell-aggressor (hitting the bid).
#   bidding at bid+1 intercepts this flow, saving ~12 ticks per unit vs
#   hitting the ask. even 5 fills of 8 units saves ~480 xirecs.
#   falsifiable: disable PASSIVE_BID and compare entry cost
#
# [H6] pepper: multi-level ask sweeping
#   evidence: fills position 8 ticks faster but costs 131 more due to
#   taking worse levels. net negative on sample data. DISABLED by default.
#   included as toggle for backtesting.
#   falsifiable: enable MULTI_LEVEL_BUY and compare avg fill price
#
# unchanged from v5:
#   - pepper is fundamentally buy-and-hold (drift = 0.1/tick = 1000/day)
#   - osmium fv = 10000 (adf p ~ 0, advisor confirmed)
#   - position limit = 80 per product
# ============================================================================

POSITION_LIMIT = 80

# -- osmium component flags (toggle for a/b testing) --
TAKE_ENABLED       = True    # [H1] take mispriced levels
SKIP_NARROW_SPREAD = False   # [H2] False = take edge in narrow spreads
TAKE_AT_FV         = True    # [H4] clear inventory at fair value

# -- osmium parameters --
OSM_FV      = 10000   # static fair value (adf confirmed stationary)
OSM_INSIDE  = 6       # half spread for passive quotes (ticks from fv)
OSM_GAMMA   = 0.10    # [H3] inventory skew intensity (0 = no skew)
OSM_MM_SIZE = 15      # passive mm order size per side
OSM_HARD_LIMIT = 75   # stop passive mm buys/sells beyond this position

# -- pepper component flags --
PASSIVE_BID      = True    # [H5] also bid at best_bid + 1
MULTI_LEVEL_BUY  = False   # [H6] sweep all ask levels (disabled: net negative)


class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        pos_pep = state.position.get('INTARIAN_PEPPER_ROOT', 0)
        pos_osm = state.position.get('ASH_COATED_OSMIUM', 0)

        if 'INTARIAN_PEPPER_ROOT' in state.order_depths:
            result['INTARIAN_PEPPER_ROOT'] = self._trade_pepper(
                state.order_depths['INTARIAN_PEPPER_ROOT'], pos_pep
            )

        if 'ASH_COATED_OSMIUM' in state.order_depths:
            result['ASH_COATED_OSMIUM'] = self._trade_osmium(
                state.order_depths['ASH_COATED_OSMIUM'], pos_osm
            )

        # no state needed across ticks for this strategy
        return result, 0, ""


    # ================================================================
    # osmium: take-and-make market making
    # ================================================================
    def _trade_osmium(self, depth: OrderDepth, position: int):
        orders: List[Order] = []

        # track remaining capacity in each direction
        # all buy orders combined must not exceed (limit - position)
        # all sell orders combined must not exceed (limit + position)
        buy_room = POSITION_LIMIT - position
        sell_room = POSITION_LIMIT + position

        # -- phase 1: take mispriced levels for profit --
        # buy everything offered below fair value
        # sell everything bid above fair value
        # this is the highest edge per tick and v5 completely ignores it

        if TAKE_ENABLED:
            # buy: sweep all ask levels strictly below fv
            if depth.sell_orders:
                for ask_price in sorted(depth.sell_orders.keys()):
                    if ask_price >= OSM_FV:
                        break
                    if buy_room <= 0:
                        break
                    ask_vol = abs(depth.sell_orders[ask_price])
                    qty = min(ask_vol, buy_room)
                    if qty > 0:
                        orders.append(Order('ASH_COATED_OSMIUM', ask_price, qty))
                        buy_room -= qty

            # sell: sweep all bid levels strictly above fv
            if depth.buy_orders:
                for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                    if bid_price <= OSM_FV:
                        break
                    if sell_room <= 0:
                        break
                    bid_vol = depth.buy_orders[bid_price]
                    qty = min(bid_vol, sell_room)
                    if qty > 0:
                        orders.append(Order('ASH_COATED_OSMIUM', bid_price, -qty))
                        sell_room -= qty

        # -- phase 2: clear inventory at fair value (break even) --
        # when long, sell at bid = fv to reduce position
        # when short, buy at ask = fv to reduce position
        # this frees up capacity for more profitable taking

        if TAKE_AT_FV:
            if position > 0 and depth.buy_orders:
                if OSM_FV in depth.buy_orders:
                    bid_vol = depth.buy_orders[OSM_FV]
                    qty = min(bid_vol, position, sell_room)
                    if qty > 0:
                        orders.append(Order('ASH_COATED_OSMIUM', OSM_FV, -qty))
                        sell_room -= qty

            if position < 0 and depth.sell_orders:
                if OSM_FV in depth.sell_orders:
                    ask_vol = abs(depth.sell_orders[OSM_FV])
                    qty = min(ask_vol, -position, buy_room)
                    if qty > 0:
                        orders.append(Order('ASH_COATED_OSMIUM', OSM_FV, qty))
                        buy_room -= qty

        # -- phase 3: passive market making with inventory skew --
        # use the avellaneda-stoikov reservation price to shift both
        # quotes based on inventory. when long, both quotes shift down
        # making the ask more attractive and the bid less attractive.
        # this is strictly better than v5's approach of only shifting
        # the bid while leaving the ask fixed at fv+6.

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread = best_ask - best_bid

        # optional narrow spread skip (v5 legacy, disabled by default)
        if SKIP_NARROW_SPREAD and spread < 14:
            return orders

        # reservation price: shifted fair value based on inventory
        # positive position -> reservation below fv -> quotes shift down
        # negative position -> reservation above fv -> quotes shift up
        reservation = OSM_FV - OSM_GAMMA * position

        # passive quote prices
        bid_price = int(reservation - OSM_INSIDE)
        ask_price = int(reservation + OSM_INSIDE)

        # safety clamps: never buy above fv or sell below fv on passive quotes
        # buying above fv = guaranteed loss in a mean reverting market
        # selling below fv = guaranteed loss
        bid_price = min(bid_price, OSM_FV - 1)
        ask_price = max(ask_price, OSM_FV + 1)

        # ensure bid < ask (sanity check after clamping)
        if bid_price >= ask_price:
            bid_price = OSM_FV - OSM_INSIDE
            ask_price = OSM_FV + OSM_INSIDE

        # hard position limits for passive mm
        # stop adding to positions that are already extreme
        want_bid = abs(position) < OSM_HARD_LIMIT or position < 0
        want_ask = abs(position) < OSM_HARD_LIMIT or position > 0

        # size for passive quotes
        bid_size = min(OSM_MM_SIZE, buy_room)
        ask_size = min(OSM_MM_SIZE, sell_room)

        if want_bid and bid_size > 0:
            orders.append(Order('ASH_COATED_OSMIUM', bid_price, bid_size))

        if want_ask and ask_size > 0:
            orders.append(Order('ASH_COATED_OSMIUM', ask_price, -ask_size))

        return orders


    # ================================================================
    # pepper root: smart accumulation + hold
    # ================================================================
    def _trade_pepper(self, depth: OrderDepth, position: int):
        orders: List[Order] = []
        remaining = POSITION_LIMIT - position

        if remaining <= 0:
            return orders

        # -- component 1: take asks to fill position --
        # the drift is 0.1 per tick (1000 per day), so every tick we
        # delay buying costs us 0.1 * remaining in missed drift.
        # buy at the best ask to fill as quickly as possible.

        if depth.sell_orders:
            if MULTI_LEVEL_BUY:
                # [H6] sweep all ask levels (fills faster but costs more)
                for ask_price in sorted(depth.sell_orders.keys()):
                    if remaining <= 0:
                        break
                    ask_vol = abs(depth.sell_orders[ask_price])
                    qty = min(ask_vol, remaining)
                    if qty > 0:
                        orders.append(
                            Order('INTARIAN_PEPPER_ROOT', ask_price, qty)
                        )
                        remaining -= qty
            else:
                # single level: take only the best ask (cheaper on average)
                best_ask = min(depth.sell_orders.keys())
                ask_vol = abs(depth.sell_orders[best_ask])
                qty = min(ask_vol, remaining)
                if qty > 0:
                    orders.append(
                        Order('INTARIAN_PEPPER_ROOT', best_ask, qty)
                    )
                    remaining -= qty

        # -- component 2: passive bid at best_bid + 1 --
        # captures sell-aggressor flow at a discount.
        # 50.9% of pepper trades are sell-side, so our bid will
        # intercept some of this flow, saving ~12 ticks per unit
        # vs hitting the ask (which costs full spread).
        #
        # the total buy quantity (take + passive) must respect the
        # position limit. we only bid with leftover capacity.

        if PASSIVE_BID and remaining > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            # bid 1 tick above the current best bid for queue priority
            bid_price = best_bid + 1
            orders.append(
                Order('INTARIAN_PEPPER_ROOT', bid_price, remaining)
            )

        # -- no sell logic --
        # holding is strictly optimal for a consistently drifting asset.
        # each unit held earns 0.1 per tick = 1000 per day in drift.
        # selling to capture spread (13 ticks) loses 1000/13 ~ 77 ticks
        # of drift exposure. the spread capture does not compensate.

        return orders
