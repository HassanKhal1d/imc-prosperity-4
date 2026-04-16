import json
from typing import List, Dict

POSITION_LIMIT = 80

OSM_FV = 10000
OSM_GAMMA = 0.10
OSM_HARD_LIMIT = 75

CALM_THRESH = 3.7
VOL_THRESH = 5.0

MM_SIZE_CALM = 18
MM_SIZE_ACTIVE = 12
MM_SIZE_VOL = 6

JUMP_THRESH = 2.5
JUMP_SIZE = 20
JUMP_MAX_POS = 60

MIN_INSIDE = 2

PASSIVE_BID     = True   # bid at best_bid+1 to capture sell-aggressor flow
MULTI_LEVEL_BUY = False  # empirically net-negative: disabled


class Order:

    def __init__(self, symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"


class OrderDepth:

    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class TradingState(object):

    def __init__(self,
                 traderData: str,
                 timestamp,
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

class Trader:

    def run(self, state: TradingState):
        result = {}

        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except:
            ts = {}

        osm_mid_hist = ts.get("osm_mid", [])
        osm_chg_hist = ts.get("osm_chg", [])
        pending_jump = ts.get("pending_jump", 0)

        pos_pep = state.position.get('INTARIAN_PEPPER_ROOT', 0)
        pos = state.position.get("ASH_COATED_OSMIUM", 0)

        # ── Update history ──
        if "ASH_COATED_OSMIUM" in state.order_depths:
            mid = self._mid(state.order_depths["ASH_COATED_OSMIUM"])
            if mid is not None:
                if osm_mid_hist:
                    osm_chg_hist.append(mid - osm_mid_hist[-1])
                osm_mid_hist.append(mid)

                osm_mid_hist = osm_mid_hist[-50:]
                osm_chg_hist = osm_chg_hist[-20:]

        z = self._z(osm_mid_hist)
        jump = self._jump(osm_chg_hist)
        regime = self._regime(osm_chg_hist)

        # detect NEW jump (store, don't trade yet)
        if jump > JUMP_THRESH:
            pending_jump = 1
        elif jump < -JUMP_THRESH:
            pending_jump = -1

        orders = []
        if "ASH_COATED_OSMIUM" in state.order_depths:
            orders = self._trade_osmium(
                state.order_depths["ASH_COATED_OSMIUM"],
                pos, z, regime, pending_jump, osm_chg_hist
            )

        if 'INTARIAN_PEPPER_ROOT' in state.order_depths:
            result['INTARIAN_PEPPER_ROOT'] = self._trade_pepper(
                state.order_depths['INTARIAN_PEPPER_ROOT'], pos_pep
            )

        # decay jump after attempt
        if pending_jump != 0:
            pending_jump = 0

        new_td = json.dumps({
            "osm_mid": osm_mid_hist,
            "osm_chg": osm_chg_hist,
            "pending_jump": pending_jump
        })

        result["ASH_COATED_OSMIUM"] = orders
        return result, 0, new_td

    # ─────────────────────────────────────
    # CORE LOGIC (v3 + upgrades)
    # ─────────────────────────────────────
    def _trade_pepper(self, depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        remaining = POSITION_LIMIT - position

        if remaining <= 0:
            return orders

        # Component 1: take best ask (fill as fast as possible)
        if depth.sell_orders and not MULTI_LEVEL_BUY:
            best_ask = min(depth.sell_orders)
            qty = min(abs(depth.sell_orders[best_ask]), remaining)
            if qty > 0:
                orders.append(Order('INTARIAN_PEPPER_ROOT', best_ask, qty))
                remaining -= qty

        elif depth.sell_orders and MULTI_LEVEL_BUY:
            for ap in sorted(depth.sell_orders):
                if remaining <= 0:
                    break
                qty = min(abs(depth.sell_orders[ap]), remaining)
                if qty > 0:
                    orders.append(Order('INTARIAN_PEPPER_ROOT', ap, qty))
                    remaining -= qty

        # Component 2: passive bid at best_bid+1 (intercept sell-aggressor flow)
        if PASSIVE_BID and remaining > 0 and depth.buy_orders:
            orders.append(Order('INTARIAN_PEPPER_ROOT', max(depth.buy_orders) + 1, remaining))

        return orders

    def _trade_osmium(self, depth, position, z, regime, pending_jump, chg_hist):
        orders = []

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        buy_room = POSITION_LIMIT - position
        sell_room = POSITION_LIMIT + position

        # ─────────────────────────────────
        # 1. FIXED JUMP LOGIC (REAL CONFIRMATION)
        # ─────────────────────────────────
        if pending_jump != 0 and len(chg_hist) >= 2:
            last_change = chg_hist[-1]

            # up jump → wait for negative tick → SELL
            if pending_jump == 1 and last_change < 0 and sell_room > 0:
                qty = min(JUMP_SIZE * 2, sell_room)  # boosted size
                orders.append(Order("ASH_COATED_OSMIUM", best_bid, -qty))
                return orders

            # down jump → wait for positive tick → BUY
            if pending_jump == -1 and last_change > 0 and buy_room > 0:
                qty = min(JUMP_SIZE * 2, buy_room)
                orders.append(Order("ASH_COATED_OSMIUM", best_ask, qty))
                return orders

        # ─────────────────────────────────
        # 2. TAKE MISPRICING (UNCHANGED + SIZE BOOST)
        # ─────────────────────────────────
        size_mult = 2.0 if abs(z) > 2.5 else 1.0

        if depth.sell_orders:
            for ap in sorted(depth.sell_orders):
                if ap >= OSM_FV or buy_room <= 0:
                    break
                qty = int(min(abs(depth.sell_orders[ap]), buy_room) * size_mult)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ap, qty))
                    buy_room -= qty

        if depth.buy_orders:
            for bp in sorted(depth.buy_orders, reverse=True):
                if bp <= OSM_FV or sell_room <= 0:
                    break
                qty = int(min(depth.buy_orders[bp], sell_room) * size_mult)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bp, -qty))
                    sell_room -= qty

        # ─────────────────────────────────
        # 3. PASSIVE MM (UNCHANGED CORE)
        # ─────────────────────────────────
        reservation = OSM_FV - OSM_GAMMA * position

        our_bid = min(best_bid + 1, int(reservation) - 1)
        our_ask = max(best_ask - 1, int(reservation) + 1)

        our_bid = min(our_bid, OSM_FV - 1)
        our_ask = max(our_ask, OSM_FV + 1)

        if our_bid >= our_ask - MIN_INSIDE:
            return orders

        if regime == "calm":
            base_size = MM_SIZE_CALM
        elif regime == "active":
            base_size = MM_SIZE_ACTIVE
        else:
            base_size = MM_SIZE_VOL

        want_bid = abs(position) < OSM_HARD_LIMIT or position < 0
        want_ask = abs(position) < OSM_HARD_LIMIT or position > 0

        bid_size = min(base_size, buy_room)
        ask_size = min(base_size, sell_room)

        if want_bid and bid_size > 0:
            orders.append(Order("ASH_COATED_OSMIUM", our_bid, bid_size))

        if want_ask and ask_size > 0:
            orders.append(Order("ASH_COATED_OSMIUM", our_ask, -ask_size))

        return orders

    # ─────────────────────────────────────
    # SIGNALS
    # ─────────────────────────────────────

    def _mid(self, depth):
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2

    def _z(self, hist):
        if len(hist) < 10:
            return 0
        dev = [x - OSM_FV for x in hist]
        mean = sum(dev) / len(dev)
        var = sum((x - mean) ** 2 for x in dev) / len(dev)
        std = var ** 0.5
        if std < 1e-6:
            return 0
        return (dev[-1] - mean) / std

    def _jump(self, chg_hist):
        if len(chg_hist) < 5:
            return 0
        mean = sum(chg_hist) / len(chg_hist)
        var = sum((x - mean) ** 2 for x in chg_hist) / len(chg_hist)
        std = var ** 0.5
        if std < 1e-6:
            return 0
        return (chg_hist[-1] - mean) / std

    def _regime(self, chg_hist):
        if len(chg_hist) < 5:
            return "active"
        mean = sum(chg_hist) / len(chg_hist)
        var = sum((x - mean) ** 2 for x in chg_hist) / len(chg_hist)
        vol = var ** 0.5

        if vol < CALM_THRESH:
            return "calm"
        elif vol < VOL_THRESH:
            return "active"
        return "volatile"