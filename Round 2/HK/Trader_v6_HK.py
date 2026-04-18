"""
Trader_v6.py — IMC Prosperity 4, Round 1
==========================================
CRITICAL BUG FIX LOG vs Trader_v5.py
======================================

ROOT CAUSE INVESTIGATION (from Log_4.json v5 result = 10357 vs 10.3k = 10348)
──────────────────────────────────────────────────────────────────────────────
Symptom : v5 OSM PnL = 2974 vs 10.3k OSM PnL = 2965 → only +9 improvement.
Expected: ~500–1000+ improvement from new signals.

POST-MORTEM — Three confirmed bugs destroying OSM PnL:

━━━ BUG 1 (CRITICAL — primary drag) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title   : Jump fader creates GUARANTEED immediate mark-to-market loss on every fill.
Evidence: Simulation of 35 jump fills in v5 → total fill PnL = -3585 ticks.
          ZERO positive fills out of 35 (0%).
          Disabling jumps improves simulation by +970 ticks/day.
Mechanism: SELL at best_bid fills at bid = mid − half_spread.
           Immediate MTM = JUMP_SIZE × (bid − mid) = 20 × (−8) = −160 ticks/fill.
           For fill to profit, price must revert > half_spread (8 ticks).
           But: >85% of "jumps" are normal bid-ask bounce (z>2.5 at std=2.7 ticks),
           where reversion is only 1–4 ticks. Net: loses 4–7 ticks per fill.
           99.5% of z>2.5 events come from CALM regime with std=2.7 (not 4–5).
           Any normal bid-ask bounce (8 tick return / 2.7 std = z=3.0) triggers.
Fix     : DISABLE jump fading entirely. The z-score jump fader is incompatible with
          tight-spread, high-frequency market making. It converts spread-neutral
          passive fills into guaranteed losers via aggressive execution cost.
          
━━━ BUG 2 (MODERATE — secondary drag) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title   : Reservation-price skew (lag-1 and one-sided) does NOT change quotes 83–92%
          of the time because quotes are constrained by book, not reservation.
Evidence: `our_bid = min(best_bid+1, int(res)-1)` — 84% of ticks the book constraint
          dominates (best_bid+1 <= int(res)-1). Reservation skew only activates at
          large positions (|pos| > 50) where gamma dominates.
          Lag-1 skew changed bid price 11.1% of ticks, avg shift 0.2 ticks.
          One-sided book skew changed bid price only 1.1% of ticks.
Fix     : Replace reservation-price skew with SIZE skew.
          After predicted direction: increase quote size on that side by 1.4×.
          After opposite prediction: decrease quote size on that side by 0.7×.
          Size skew works regardless of book state and is always effective.

━━━ BUG 3 (MINOR — regime mismatch) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title   : Regime thresholds from v5 (5.0/6.5) may cause underquoting vs original.
Evidence: The new thresholds correctly label 80% of ticks as "calm" (vs 60% before),
          increasing MM_SIZE from 12 → 18 for more ticks. This is correct.
          No change needed here — this improvement is valid.

CONFIRMED IMPROVEMENTS KEPT FROM V5:
  ✓ Regime thresholds: CALM_THRESH=5.0, VOL_THRESH=6.5  (calibrated q80/q95)
  ✓ prev_state tracking for one-sided book signals
  ✓ One-sided book passive quote (passive fill opportunity, small size)
  ✓ Spread compression mode MM_SIZE_COMPRESS=25

NEW IN V6:
  + Jump fader: DISABLED (replaced by passive FV take which is already in Layer 2)
  + Lag-1 SIZE skew: after uptick → increase ask_size, decrease bid_size
  + One-sided SIZE skew: after only-bid → increase bid_size; after only-ask → ask_size
  + Larger base sizes in calm/active regime (more aggressive passive MM)

EXPECTED IMPROVEMENT:
  +970 ticks/day from jump fader removal (simulation estimate)
  +~200 ticks/day from size skew improvements (theoretical estimate)
  Total: +1000-1200 ticks/day OSM improvement expected
"""

import json
from typing import List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80
OSM_FV         = 10000

# ── Regime thresholds (calibrated q80/q95 over 60k ticks) ────────────────────
CALM_THRESH = 5.0    # rolling_std(10) q80 = 5.01
VOL_THRESH  = 6.5    # rolling_std(10) q95 = 6.49

# ── BASE MM sizes ─────────────────────────────────────────────────────────────
# Increased slightly from v5 since we're no longer burning position budget on
# aggressive jump fills. More inventory headroom → can quote larger.
MM_SIZE_CALM     = 20   # was 18 → +2 (extra fill capacity in dominant regime)
MM_SIZE_ACTIVE   = 14   # was 12 → +2
MM_SIZE_VOL      =  6   # unchanged — volatile regime = defensive
MM_SIZE_COMPRESS = 28   # was 25 → +3 (spread compression = cheap liquidity)

# ── Size skew multipliers (Bug 2 fix) ─────────────────────────────────────────
# After UP tick (lag-1 autocorr = −0.49 → 74% chance of DOWN next):
#   Sell side: favored → increase ask_size by factor
#   Buy side:  disfavored → decrease bid_size by factor
LAG1_SKEW_FACTOR_FAV   = 1.5   # multiply size on favoured side
LAG1_SKEW_FACTOR_UNFAV = 0.7   # multiply size on unfavoured side

# After ONE-SIDED BOOK event (95%+ confidence next tick reverts):
#   prev=only_bid  → price rises → BUY side favoured
#   prev=only_ask  → price falls → SELL side favoured
ONESIDED_SKEW_FAV   = 1.6
ONESIDED_SKEW_UNFAV = 0.7

# ── Inventory management ─────────────────────────────────────────────────────
OSM_GAMMA    = 0.10   # inv skew: reservation = FV − γ × pos (optimal = 8.09/80)
OSM_HARD_LIM = 70     # start size reduction earlier (was 75) — tighter inventory

# ── One-sided book passive quote ──────────────────────────────────────────────
# Post a small passive order on the MISSING side.
# Size intentionally small: we don't know if a bot will fill us.
ONESIDED_PASSIVE_SIZE = 8  # conservative (was 8, unchanged)

# ── Spread thresholds ─────────────────────────────────────────────────────────
SPREAD_COMPRESS = 12   # below → compression mode
MIN_INSIDE      = 2    # minimum bid-ask separation for our quotes


# ─────────────────────────────────────────────────────────────────────────────
# DATAMODEL STUBS
# ─────────────────────────────────────────────────────────────────────────────

class Order:
    def __init__(self, symbol, price: int, quantity: int) -> None:
        self.symbol   = symbol
        self.price    = price
        self.quantity = quantity

    def __str__(self)  -> str: return f"({self.symbol}, {self.price}, {self.quantity})"
    def __repr__(self) -> str: return self.__str__()


class OrderDepth:
    def __init__(self):
        self.buy_orders:  Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class TradingState(object):
    def __init__(self, traderData, timestamp, listings,
                 order_depths, own_trades, market_trades, position, observations):
        self.traderData    = traderData
        self.timestamp     = timestamp
        self.listings      = listings
        self.order_depths  = order_depths
        self.own_trades    = own_trades
        self.market_trades = market_trades
        self.position      = position
        self.observations  = observations


# ─────────────────────────────────────────────────────────────────────────────
# TRADER
# ─────────────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        result = {}

        # ── Restore persistent state ──────────────────────────────────────────
        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        osm_mid_hist = ts.get("osm_mid",    [])   # mid-price history (last 50)
        osm_chg_hist = ts.get("osm_chg",    [])   # return history (last 20)
        prev_state   = ts.get("prev_state", 0)    # 0=normal, 1=only_bid, -1=only_ask

        pos_pep = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        pos_osm = state.position.get("ASH_COATED_OSMIUM",    0)

        # ── Update OSM history & classify book state ──────────────────────────
        cur_state = 0
        if "ASH_COATED_OSMIUM" in state.order_depths:
            depth   = state.order_depths["ASH_COATED_OSMIUM"]
            has_bid = bool(depth.buy_orders)
            has_ask = bool(depth.sell_orders)

            if has_bid and has_ask:
                mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
                cur_state = 0
            elif has_bid:
                mid = float(max(depth.buy_orders))
                cur_state = 1
            elif has_ask:
                mid = float(min(depth.sell_orders))
                cur_state = -1
            else:
                mid = None

            if mid is not None:
                if osm_mid_hist:
                    osm_chg_hist.append(mid - osm_mid_hist[-1])
                osm_mid_hist.append(mid)

            osm_mid_hist = osm_mid_hist[-50:]
            osm_chg_hist = osm_chg_hist[-20:]

        # ── Compute regime ────────────────────────────────────────────────────
        regime = self._regime(osm_chg_hist)

        # ── Place OSM orders ──────────────────────────────────────────────────
        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self._trade_osm(
                state.order_depths["ASH_COATED_OSMIUM"],
                pos_osm, regime, prev_state, osm_chg_hist
            )

        # ── Place PEPPER ROOT orders ───────────────────────────────────────────
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self._trade_pepper(
                state.order_depths["INTARIAN_PEPPER_ROOT"], pos_pep
            )

        # ── Persist state ─────────────────────────────────────────────────────
        new_td = json.dumps({
            "osm_mid":    osm_mid_hist,
            "osm_chg":    osm_chg_hist,
            "prev_state": cur_state,
        })

        return result, 0, new_td

    # =========================================================================
    # OSM TRADING LOGIC
    # =========================================================================

    def _trade_osm(self, depth: OrderDepth, position: int,
                   regime: str, prev_state: int,
                   chg_hist: list) -> List[Order]:
        """
        Two-layer strategy. Jump fader REMOVED (Bug 1 fix).

        Layer 1: FV mispricing take
          Buy any asks priced < FV. Sell any bids priced > FV.
          Safe, unambiguous alpha — no spread cost because we're pricing vs fair value.

        Layer 2: Passive market making with SIZE skew (Bug 2 fix)
          Quote inside the spread. Size is skewed based on:
            A) Lag-1 autocorrelation signal (−0.49 autocorr, 74% accuracy)
            B) Previous one-sided book state (95%+ accuracy)
          Reservoir price uses inventory skew (gamma=0.10, empirically optimal).
        """
        orders   = []
        has_bid  = bool(depth.buy_orders)
        has_ask  = bool(depth.sell_orders)

        # ── ONE-SIDED BOOK: passive quote on missing side (unchanged from v5) ─
        if not has_bid or not has_ask:
            buy_room  = POSITION_LIMIT - position
            sell_room = POSITION_LIMIT + position

            if has_bid and not has_ask and buy_room > 0:
                # Only bids → price rises next tick (95.5% confidence)
                # Post small passive BUY above best_bid
                best_bid = max(depth.buy_orders)
                px = min(best_bid + 2, OSM_FV + 5)
                orders.append(Order("ASH_COATED_OSMIUM", px,
                                    min(ONESIDED_PASSIVE_SIZE, buy_room)))

            elif has_ask and not has_bid and sell_room > 0:
                # Only asks → price falls next tick (95.6% confidence)
                # Post small passive SELL below best_ask
                best_ask = min(depth.sell_orders)
                px = max(best_ask - 2, OSM_FV - 5)
                orders.append(Order("ASH_COATED_OSMIUM", px,
                                    -min(ONESIDED_PASSIVE_SIZE, sell_room)))

            return orders

        # ── FULL TWO-SIDED BOOK ───────────────────────────────────────────────
        best_bid  = max(depth.buy_orders)
        best_ask  = min(depth.sell_orders)
        spread    = best_ask - best_bid
        buy_room  = POSITION_LIMIT - position
        sell_room = POSITION_LIMIT + position

        # ─────────────────────────────────────────────────────────────────────
        # LAYER 1: FV MISPRICING TAKE
        # Buy orders priced below FV (asks < 10000) — guaranteed edge at entry.
        # Sell orders priced above FV (bids > 10000) — guaranteed edge at entry.
        # These trades have ZERO immediate MTM loss (unlike aggressive jump fades).
        # ─────────────────────────────────────────────────────────────────────

        if depth.sell_orders:
            for ap in sorted(depth.sell_orders):
                if ap >= OSM_FV or buy_room <= 0:
                    break
                qty = min(abs(depth.sell_orders[ap]), buy_room)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ap, qty))
                    buy_room -= qty

        if depth.buy_orders:
            for bp in sorted(depth.buy_orders, reverse=True):
                if bp <= OSM_FV or sell_room <= 0:
                    break
                qty = min(depth.buy_orders[bp], sell_room)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bp, -qty))
                    sell_room -= qty

        # ─────────────────────────────────────────────────────────────────────
        # LAYER 2: PASSIVE MARKET MAKING with size skew
        # ─────────────────────────────────────────────────────────────────────

        # Reservation price (Avellaneda-Stoikov inventory skew)
        reservation = OSM_FV - OSM_GAMMA * position

        # Quote placement: 1 tick inside best bid/ask, clamped to FV±1
        our_bid = min(best_bid + 1, int(reservation) - 1)
        our_ask = max(best_ask - 1, int(reservation) + 1)
        our_bid = min(our_bid, OSM_FV - 1)
        our_ask = max(our_ask, OSM_FV + 1)

        # Ensure minimum spread
        if our_bid >= our_ask - MIN_INSIDE:
            return orders

        # ── A) Base size from spread/regime ──────────────────────────────────
        if spread < SPREAD_COMPRESS:
            base_size = MM_SIZE_COMPRESS   # 28: compression = cheap liquidity
        elif regime == "calm":
            base_size = MM_SIZE_CALM       # 20: 80% of ticks → most fills here
        elif regime == "active":
            base_size = MM_SIZE_ACTIVE     # 14
        else:
            base_size = MM_SIZE_VOL        #  6: volatile = reduce exposure

        # ── B) LAG-1 SIZE SKEW (Bug 2 fix) ───────────────────────────────────
        # Lag-1 autocorr = −0.49: after UP tick, next is DOWN with 74% confidence.
        # Scale ask_size UP and bid_size DOWN after uptick (and vice versa after downtick).
        # This works regardless of book state (unlike reservation-price skew).
        bid_mult = 1.0
        ask_mult = 1.0

        if chg_hist:
            last_ret = chg_hist[-1]
            if last_ret > 0:
                # After UP → expect DOWN → favour SELL side
                ask_mult = LAG1_SKEW_FACTOR_FAV     # 1.5× ask
                bid_mult = LAG1_SKEW_FACTOR_UNFAV   # 0.7× bid
            elif last_ret < 0:
                # After DOWN → expect UP → favour BUY side
                bid_mult = LAG1_SKEW_FACTOR_FAV     # 1.5× bid
                ask_mult = LAG1_SKEW_FACTOR_UNFAV   # 0.7× ask

        # ── C) ONE-SIDED BOOK SIZE SKEW (Bug 2 fix) ──────────────────────────
        # prev_state=1 (only-bid last tick) → price rises next tick (95.5%).
        #   Amplify buy side: 1.6× bid, 0.7× ask.
        # prev_state=-1 (only-ask last tick) → price falls next tick (95.6%).
        #   Amplify sell side: 1.6× ask, 0.7× bid.
        # One-sided signal is STRONGER than lag-1, so it takes priority.
        if prev_state == 1:
            bid_mult = ONESIDED_SKEW_FAV      # 1.6× bid (price rises)
            ask_mult = ONESIDED_SKEW_UNFAV    # 0.7× ask
        elif prev_state == -1:
            ask_mult = ONESIDED_SKEW_FAV      # 1.6× ask (price falls)
            bid_mult = ONESIDED_SKEW_UNFAV    # 0.7× bid

        # ── D) Final sizes with position guard ───────────────────────────────
        bid_size = int(base_size * bid_mult)
        ask_size = int(base_size * ask_mult)

        # Reduce size near hard position limit
        want_bid = abs(position) < OSM_HARD_LIM or position < 0
        want_ask = abs(position) < OSM_HARD_LIM or position > 0

        bid_size = min(bid_size, buy_room)
        ask_size = min(ask_size, sell_room)

        if want_bid and bid_size > 0:
            orders.append(Order("ASH_COATED_OSMIUM", our_bid,  bid_size))
        if want_ask and ask_size > 0:
            orders.append(Order("ASH_COATED_OSMIUM", our_ask, -ask_size))

        return orders

    # =========================================================================
    # PEPPER ROOT STRATEGY (unchanged — empirically optimal, already at max)
    # =========================================================================

    def _trade_pepper(self, depth: OrderDepth, position: int) -> List[Order]:
        """
        Buy to maximum long position as efficiently as possible.
        Aggress best ask for fastest fill. Passive bid at best_bid+1 for
        secondary fill from sell-aggressor bots.
        """
        orders    = []
        remaining = POSITION_LIMIT - position

        if remaining <= 0:
            return orders

        # Aggress best ask
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            qty = min(abs(depth.sell_orders[best_ask]), remaining)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, qty))
                remaining -= qty

        # Passive bid at best_bid+1
        if remaining > 0 and depth.buy_orders:
            orders.append(Order("INTARIAN_PEPPER_ROOT",
                                max(depth.buy_orders) + 1, remaining))

        return orders

    # =========================================================================
    # SIGNAL COMPUTATIONS
    # =========================================================================

    def _regime(self, chg_hist: list) -> str:
        """
        3-state volatility regime using CALIBRATED thresholds (from 60k tick EDA).
          calm     → rolling_std(10) < 5.0   (q80, ~80% of ticks)
          active   → rolling_std(10) ∈ [5.0, 6.5)  (~15%)
          volatile → rolling_std(10) ≥ 6.5   (q95, ~5%)

        Old thresholds (3.7/5.0) were q60/q80 → over-classified ticks as "active",
        under-sizing in the dominant regime. Fixed in v5, kept in v6.
        """
        if len(chg_hist) < 5:
            return "active"
        mean = sum(chg_hist) / len(chg_hist)
        var  = sum((x - mean) ** 2 for x in chg_hist) / len(chg_hist)
        vol  = var ** 0.5

        if vol < CALM_THRESH:
            return "calm"
        elif vol < VOL_THRESH:
            return "active"
        return "volatile"
