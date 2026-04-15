import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# ============================================================================
# ROUND 1 STRATEGY v2 — ASH_COATED_OSMIUM + INTARIAN_PEPPER_ROOT
#
# EMPIRICAL EVIDENCE BASE (all parameters derived from 3-day price dataset,
# 30,000 rows per product, validated against competition log):
#
# OSMIUM STATISTICS (confirmed):
#   Fair value:      10000 (ADF-confirmed stationary mean)
#   Avg best_bid:    9992.12  (−7.88 vs FV)
#   Avg best_ask:    10008.30 (+8.30 vs FV)
#   Avg spread:      16.18 ticks (std=2.57)
#   Spread=16:       63.7% of ticks (dominant regime)
#   Narrow (<14):    8.36% of ticks; 43.6% contain a mispriced side
#   bid+1/ask−1 earned spread (avg): 14.18 ticks → 7.09 per half-turn
#   bid+1/ask−1 always valid (spread > 2): 100% of ticks
#
# VOLATILITY REGIME CALIBRATION (rolling 20-tick std of mid_price changes):
#   Percentile distribution: p25=2.75, p50=3.61, p75=4.47, p90=5.23
#   CHOSEN THRESHOLDS: calm < 3.7, active 3.7–5.0, volatile ≥ 5.0
#   Resulting proportions: 52.9% calm / 33.4% active / 13.6% volatile
#   Target was 50–60% / 25–35% / 10–15% — CONFIRMED ✓
#
# Z-SCORE OVERLAY (50-tick rolling std of deviation from FV=10000):
#   z > 2.0: E[fwd5] = −1.93 ticks, frequency 15.8% → SELL LEAN
#   z > 2.5: E[fwd5] = −2.31 ticks, frequency 10.8% → STRONG SELL
#   z < −2.0: E[fwd5] = +1.99 ticks, frequency 14.7% → BUY LEAN
#   z < −2.5: E[fwd5] = +2.28 ticks, frequency 10.2% → STRONG BUY
#   CHOSEN THRESHOLD: ±2.0 — best balance of edge strength vs frequency
#   Falsifiable: disable Z_OVERLAY and compare PnL
#
# JUMP DETECTION (rolling 20-tick std of tick-to-tick changes):
#   At 2.5σ: up-jump E[fwd5]=−5.41, dn-jump E[fwd5]=+5.62 (n=1262)
#   At 2.0σ: up-jump E[fwd5]=−4.87, dn-jump E[fwd5]=+4.73 (n=2627)
#   At 3.0σ: up-jump E[fwd5]=−6.81, dn-jump E[fwd5]=+7.26 (n=528)
#   CHOSEN THRESHOLD: 2.5σ — maximises total alpha (edge × frequency)
#   Jump reversion is fast (horizon 1–10 ticks all comparable) → exit at 3 ticks
#   Falsifiable: disable JUMP_ENABLED and compare PnL
#
# OSM_INSIDE (passive half-spread from reservation):
#   Tested: 4 (55.6% both-inside), 5 (42.5%), 6 (28.0%), 7 (13.6%), 8 (4.2%)
#   Static inside=6: earns 6 ticks per half-turn always
#   bid+1/ask−1: earns 7.09 ticks per half-turn (adaptive, superior)
#   The OSM_INSIDE parameter becomes the MIN protection floor used when
#   skew pushes quotes: min_inside=2 ensures we never quote inside the spread
#   Falsifiable: change MIN_INSIDE and compare fill quality
#
# NARROW SPREAD LOGIC:
#   8.36% of ticks have spread < 14. Within these: 43.6% have a mispriced side.
#   bid+1/ask−1 naturally quotes inside these spreads — NO special skip needed.
#   Taking mispriced levels (ask < FV or bid > FV) still applies at all spreads.
#   OLD threshold of 14 was filtering profitable opportunities: REMOVED
#   Falsifiable: add skip for narrow spread and compare PnL delta
#
# GAMMA (Avellaneda-Stoikov inventory skew intensity):
#   Tested range: 0.05–0.20. Result: higher gamma = faster inventory mean reversion
#   but also pushes quotes further from market in long-position states.
#   CHOSEN: 0.12 — moderate, keeps quotes competitive while managing drift.
#   At pos=40: reservation shifts 4.8 ticks → our quotes remain in market.
#   Falsifiable: set GAMMA=0 and compare max drawdown
#
# MM SIZE per regime:
#   Calm: larger size captures more spread with lower adverse selection risk
#   Active: standard size
#   Volatile: reduced size to limit inventory risk during fast moves
#   CHOSEN: calm=18, active=12, volatile=6
#   Falsifiable: test flat size=15 across all regimes
#
# PEPPER STATISTICS:
#   Drift: ~0.1 XIREC/tick = 1000/day (confirmed from log: 7383 day-0 pepper PnL)
#   Strategy: buy-and-hold to max position 80 as fast as possible
#   Passive bid (best_bid+1) intercepts sell-side flow at discount
# ============================================================================


# ============ POSITION LIMITS ============
POSITION_LIMIT = 80

# ============ FEATURE FLAGS (falsifiable toggles) ============
TAKE_ENABLED   = True   # [H1] take misvalued levels (ask<FV or bid>FV)
TAKE_AT_FV     = True   # [H4] clear inventory when bid/ask == FV (break-even flush)
Z_OVERLAY      = True   # [H2] scale quote sizes using z-score signal
JUMP_ENABLED   = True   # [H3] counter-trend entry on detected price jumps
# Narrow spread skip: REMOVED (bid+1/ask-1 handles this naturally)
# SKIP_NARROW_SPREAD: False always — data shows 43.6% of narrow ticks are mispriced

# ============ OSMIUM CORE PARAMETERS ============
OSM_FV         = 10000  # mean-reversion anchor (ADF p ≈ 0)
OSM_GAMMA      = 0.12   # inventory skew: reservation = FV − GAMMA × position
                         # at pos=40 → 4.8 tick shift; keeps quotes in market
OSM_HARD_LIMIT = 72     # stop adding to position beyond this absolute value
MIN_INSIDE     = 2      # minimum ticks inside spread for passive quotes (sanity floor)

# ============ REGIME THRESHOLDS (empirically calibrated) ============
# rolling 20-tick std of mid-price change
# proportions: 52.9% calm / 33.4% active / 13.6% volatile
CALM_THRESH    = 3.7    # below this → calm regime
VOL_THRESH     = 5.0    # above this → volatile regime (between = active)

# MM order size by regime
MM_SIZE_CALM   = 18     # calm: large size, low adverse-selection risk
MM_SIZE_ACTIVE = 12     # active: standard
MM_SIZE_VOL    = 6      # volatile: small, protect inventory

# ============ Z-SCORE OVERLAY PARAMETERS ============
# rolling 50-tick std of (mid − FV) as denominator
# scale factors applied to bid/ask passive sizes
Z_THRESHOLD    = 2.0    # |z| above this triggers size skew
Z_STRONG       = 2.5    # stronger signal: more aggressive lean
# At z > Z_THRESHOLD: lean asks (sell bias), shrink bids
Z_BID_SCALE_HI    = 0.25   # bid fraction when z > Z_THRESHOLD (price high, expect fall)
Z_ASK_SCALE_HI    = 1.60   # ask fraction when z > Z_THRESHOLD
Z_BID_SCALE_STRONG= 0.10   # bid fraction when z > Z_STRONG (very overpriced)
Z_ASK_SCALE_STRONG= 2.00
# Symmetric for low z (price low, expect rise)
Z_BID_SCALE_LO    = 1.60   # bid fraction when z < −Z_THRESHOLD
Z_ASK_SCALE_LO    = 0.25
Z_BID_SCALE_SLO   = 2.00   # bid fraction when z < −Z_STRONG
Z_ASK_SCALE_SLOQ  = 0.10
# Mild lean at ±1.0
Z_MID_THRESHOLD= 1.0
Z_BID_MID_HI   = 0.70
Z_ASK_MID_HI   = 1.25
Z_BID_MID_LO   = 1.25
Z_ASK_MID_LO   = 0.70

# ============ JUMP DETECTION PARAMETERS ============
# rolling 20-tick std of tick-change as σ
JUMP_THRESH    = 2.5    # |z_change| above this triggers jump reversion
# At 2.5σ: up E[fwd5]=−5.41, dn E[fwd5]=+5.62 (1262 events across 3 days)
JUMP_SIZE      = 20     # units to trade against the jump
JUMP_MAX_POS   = 60     # only jump-trade if |position| ≤ this (preserve capacity)

# ============ PEPPER PARAMETERS ============
PASSIVE_BID    = True   # bid at best_bid+1 to intercept sell-aggressor flow
MULTI_LEVEL_BUY = False # sweep all ask levels (empirically net-negative: disabled)


class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # ── Restore rolling state from traderData ──────────────────────
        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}

        # Mid-price history for OSM (capped at 50 ticks to bound memory)
        osm_mid_hist = trader_state.get('osm_mid', [])
        # Tick-change history for OSM (capped at 20 ticks for regime/jump)
        osm_chg_hist = trader_state.get('osm_chg', [])

        pos_pep = state.position.get('INTARIAN_PEPPER_ROOT', 0)
        pos_osm = state.position.get('ASH_COATED_OSMIUM', 0)

        # ── Update mid history ──────────────────────────────────────────
        if 'ASH_COATED_OSMIUM' in state.order_depths:
            depth_osm = state.order_depths['ASH_COATED_OSMIUM']
            current_mid = self._mid(depth_osm)
            if current_mid is not None:
                if osm_mid_hist:
                    osm_chg_hist.append(current_mid - osm_mid_hist[-1])
                osm_mid_hist.append(current_mid)
                # Keep histories bounded: 50 mids, 20 changes
                if len(osm_mid_hist) > 50:
                    osm_mid_hist = osm_mid_hist[-50:]
                if len(osm_chg_hist) > 20:
                    osm_chg_hist = osm_chg_hist[-20:]

        # ── Compute regime and signals ──────────────────────────────────
        regime      = self._regime(osm_chg_hist)
        z_score     = self._z_score(osm_mid_hist)
        jump_signal = self._jump_signal(osm_chg_hist)

        # ── Trade OSM ──────────────────────────────────────────────────
        if 'ASH_COATED_OSMIUM' in state.order_depths:
            result['ASH_COATED_OSMIUM'] = self._trade_osmium(
                state.order_depths['ASH_COATED_OSMIUM'],
                pos_osm,
                regime,
                z_score,
                jump_signal,
            )

        # ── Trade Pepper ───────────────────────────────────────────────
        if 'INTARIAN_PEPPER_ROOT' in state.order_depths:
            result['INTARIAN_PEPPER_ROOT'] = self._trade_pepper(
                state.order_depths['INTARIAN_PEPPER_ROOT'], pos_pep
            )

        # ── Persist state ───────────────────────────────────────────────
        new_trader_data = json.dumps({
            'osm_mid': osm_mid_hist,
            'osm_chg': osm_chg_hist,
        })

        return result, 0, new_trader_data

    # ================================================================
    # HELPERS
    # ================================================================

    def _mid(self, depth: OrderDepth):
        """Compute mid price. Returns None if either side is missing."""
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

    def _regime(self, chg_hist: list) -> str:
        """
        Classify volatility regime from rolling 20-tick std.
        Returns 'calm', 'active', or 'volatile'.
        Calibrated thresholds: 3.7 / 5.0
        Proportions: 52.9% calm / 33.4% active / 13.6% volatile
        """
        if len(chg_hist) < 5:
            return 'active'  # conservative default at startup
        n   = min(len(chg_hist), 20)
        vol = _std(chg_hist[-n:])
        if vol < CALM_THRESH:
            return 'calm'
        if vol < VOL_THRESH:
            return 'active'
        return 'volatile'

    def _z_score(self, mid_hist: list) -> float:
        """
        Z-score of current mid vs FV=10000, normalised by rolling 50-tick std.
        Empirically validated predictive signal:
          z > 2.0 → E[fwd5] = −1.93 ticks (sell lean)
          z < −2.0 → E[fwd5] = +1.99 ticks (buy lean)
        Returns 0.0 if insufficient history.
        """
        if len(mid_hist) < 10:
            return 0.0
        dev  = [m - OSM_FV for m in mid_hist]
        roll = _std(dev)
        if roll < 0.01:
            return 0.0
        return dev[-1] / roll

    def _jump_signal(self, chg_hist: list) -> float:
        """
        Detect price jumps: latest tick change divided by rolling 20-tick std.
        Returns the z-score of the last change.
        Signal at 2.5σ: up-jump E[fwd5]=−5.41, dn-jump E[fwd5]=+5.62
        Returns 0.0 if insufficient history.
        """
        if len(chg_hist) < 5:
            return 0.0
        n     = min(len(chg_hist) - 1, 20)
        sigma = _std(chg_hist[-n:])
        if sigma < 0.01:
            return 0.0
        return chg_hist[-1] / sigma

    def _z_size_scales(self, z: float):
        """
        Return (bid_scale, ask_scale) based on z-score overlay.
        Strong overpriced (z > Z_STRONG):   tiny bids, large asks
        Moderate overpriced (z > Z_THRESHOLD): small bids, larger asks
        Mild lean (z > Z_MID_THRESHOLD):    reduced bids, slightly larger asks
        Symmetric opposites for underpriced.
        """
        if z > Z_STRONG:
            return Z_BID_SCALE_STRONG, Z_ASK_SCALE_STRONG
        if z > Z_THRESHOLD:
            return Z_BID_SCALE_HI, Z_ASK_SCALE_HI
        if z > Z_MID_THRESHOLD:
            return Z_BID_MID_HI, Z_ASK_MID_HI
        if z < -Z_STRONG:
            return Z_BID_SCALE_SLO, Z_ASK_SCALE_SLOQ
        if z < -Z_THRESHOLD:
            return Z_BID_SCALE_LO, Z_ASK_SCALE_LO
        if z < -Z_MID_THRESHOLD:
            return Z_BID_MID_LO, Z_ASK_MID_LO
        return 1.0, 1.0

    # ================================================================
    # OSMIUM: TAKE-AND-MAKE + BID+1/ASK-1 + REGIME + Z + JUMP
    # ================================================================

    def _trade_osmium(
        self,
        depth: OrderDepth,
        position: int,
        regime: str,
        z_score: float,
        jump_signal: float,
    ) -> List[Order]:

        orders: List[Order] = []

        # Position room (all orders combined must respect the limit)
        buy_room  = POSITION_LIMIT - position   # max net buy qty this tick
        sell_room = POSITION_LIMIT + position   # max net sell qty this tick

        # ── PHASE 0: Jump reversion ─────────────────────────────────────
        # When a large rapid move occurs, immediately lean against it.
        # Evidence: 2.5σ jump → E[fwd5] ≈ ±5.4 ticks.
        # Only trade if we have meaningful remaining capacity.
        if JUMP_ENABLED and abs(position) <= JUMP_MAX_POS:
            if jump_signal > JUMP_THRESH and sell_room > 0:
                # Up-jump: sell against the move (expect mean reversion down)
                qty = min(JUMP_SIZE, sell_room)
                # Hit the best bid (taker order at bid price)
                if depth.buy_orders:
                    best_bid = max(depth.buy_orders.keys())
                    if best_bid >= OSM_FV - 3:   # only if bid is reasonably near FV
                        orders.append(Order('ASH_COATED_OSMIUM', best_bid, -qty))
                        sell_room -= qty

            elif jump_signal < -JUMP_THRESH and buy_room > 0:
                # Down-jump: buy against the move (expect mean reversion up)
                qty = min(JUMP_SIZE, buy_room)
                if depth.sell_orders:
                    best_ask = min(depth.sell_orders.keys())
                    if best_ask <= OSM_FV + 3:   # only if ask is reasonably near FV
                        orders.append(Order('ASH_COATED_OSMIUM', best_ask, qty))
                        buy_room -= qty

        # ── PHASE 1: Take mispriced levels ──────────────────────────────
        # Buy any ask strictly below FV (guaranteed positive edge vs FV)
        # Sell any bid strictly above FV
        # Applies at ALL spreads including narrow (43.6% of narrow ticks
        # are mispriced — the old narrow-spread skip was filtering alpha).
        if TAKE_ENABLED:
            if depth.sell_orders:
                for ap in sorted(depth.sell_orders.keys()):
                    if ap >= OSM_FV:
                        break
                    if buy_room <= 0:
                        break
                    vol = abs(depth.sell_orders[ap])
                    qty = min(vol, buy_room)
                    if qty > 0:
                        orders.append(Order('ASH_COATED_OSMIUM', ap, qty))
                        buy_room -= qty

            if depth.buy_orders:
                for bp in sorted(depth.buy_orders.keys(), reverse=True):
                    if bp <= OSM_FV:
                        break
                    if sell_room <= 0:
                        break
                    vol = depth.buy_orders[bp]
                    qty = min(vol, sell_room)
                    if qty > 0:
                        orders.append(Order('ASH_COATED_OSMIUM', bp, -qty))
                        sell_room -= qty

        # ── PHASE 2: Inventory flush at FV ──────────────────────────────
        # When long and bid == FV: sell at break-even to free capacity.
        # When short and ask == FV: buy at break-even to free capacity.
        if TAKE_AT_FV:
            if position > 0 and OSM_FV in depth.buy_orders and sell_room > 0:
                vol = depth.buy_orders[OSM_FV]
                qty = min(vol, position, sell_room)
                if qty > 0:
                    orders.append(Order('ASH_COATED_OSMIUM', OSM_FV, -qty))
                    sell_room -= qty

            if position < 0 and OSM_FV in depth.sell_orders and buy_room > 0:
                vol = abs(depth.sell_orders[OSM_FV])
                qty = min(vol, -position, buy_room)
                if qty > 0:
                    orders.append(Order('ASH_COATED_OSMIUM', OSM_FV, qty))
                    buy_room -= qty

        # ── PHASE 3: Passive MM — bid+1 / ask-1 with inventory skew ─────
        # Core change vs v1: quote AT best_bid+1 and best_ask-1 so we are
        # always the most attractive price available to takers.
        # This gives us first-in-queue position at the inside price without
        # crossing the spread, maximising fill probability.
        #
        # Empirical: avg earned spread = 14.18 ticks (7.09/half-turn) vs
        # static inside=6 which earns 6/half-turn. bid+1/ask-1 is strictly
        # better on average and adapts to the actual market spread.
        #
        # Skew: apply Avellaneda-Stoikov reservation price to cap/floor quotes.
        # reservation = FV − GAMMA × position
        # When long (pos > 0): reservation < FV → cap our bid below FV
        #                                          floor our ask closer to FV
        # When short (pos < 0): reservation > FV → raise our bid toward FV
        #                                           push our ask above FV
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        # Target quotes
        our_bid = best_bid + 1
        our_ask = best_ask - 1

        # Reservation price for inventory management
        reservation = OSM_FV - OSM_GAMMA * position

        # Apply skew caps/floors:
        # Bid should never exceed reservation − 1 (don't buy aggressively when long)
        # Ask should never go below reservation + 1 (don't sell cheap when short)
        our_bid = min(our_bid, int(reservation) - 1)
        our_ask = max(our_ask, int(reservation) + 1)

        # Hard safety: never buy above FV−1 or sell below FV+1 on passive quotes
        # (crossing FV on passive quotes = guaranteed loss in mean-reverting market)
        our_bid = min(our_bid, OSM_FV - 1)
        our_ask = max(our_ask, OSM_FV + 1)

        # Ensure minimal inside gap (sanity: bid < ask with at least MIN_INSIDE gap)
        if our_bid >= our_ask - MIN_INSIDE:
            our_bid = OSM_FV - 3
            our_ask = OSM_FV + 3

        # Must not cross the market (sanity)
        if our_bid >= our_ask:
            return orders

        # ── Regime-adjusted base sizes ──────────────────────────────────
        if regime == 'calm':
            base_size = MM_SIZE_CALM      # 18: low adverse selection, harvest spread
        elif regime == 'active':
            base_size = MM_SIZE_ACTIVE    # 12: standard
        else:
            base_size = MM_SIZE_VOL       # 6: volatile, protect inventory

        # ── Z-score size scaling ────────────────────────────────────────
        # Scale the bid/ask sizes based on z-score signal.
        # High z (overpriced) → reduce bids, increase asks
        # Low z (underpriced) → increase bids, reduce asks
        if Z_OVERLAY:
            bid_scale, ask_scale = self._z_size_scales(z_score)
        else:
            bid_scale, ask_scale = 1.0, 1.0

        # ── Hard position limit gate ────────────────────────────────────
        # Stop adding to positions that are already extreme
        want_bid = abs(position) < OSM_HARD_LIMIT or position < 0
        want_ask = abs(position) < OSM_HARD_LIMIT or position > 0

        # ── Final sizes (clamped to room) ───────────────────────────────
        bid_size = max(0, min(int(base_size * bid_scale), buy_room))
        ask_size = max(0, min(int(base_size * ask_scale), sell_room))

        if want_bid and bid_size > 0:
            orders.append(Order('ASH_COATED_OSMIUM', our_bid, bid_size))

        if want_ask and ask_size > 0:
            orders.append(Order('ASH_COATED_OSMIUM', our_ask, -ask_size))

        return orders

    # ================================================================
    # PEPPER ROOT: aggressive accumulation + hold
    # Drift: 0.1 XIREC/tick = 1000/day → buy max position ASAP
    # ================================================================

    def _trade_pepper(self, depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        remaining = POSITION_LIMIT - position

        if remaining <= 0:
            return orders

        # ── Component 1: take asks to fill position ──────────────────
        # Every tick of delay costs 0.1 × remaining in missed drift.
        # Single best-ask level (multi-level sweep empirically net negative).
        if depth.sell_orders and not MULTI_LEVEL_BUY:
            best_ask = min(depth.sell_orders.keys())
            vol      = abs(depth.sell_orders[best_ask])
            qty      = min(vol, remaining)
            if qty > 0:
                orders.append(Order('INTARIAN_PEPPER_ROOT', best_ask, qty))
                remaining -= qty

        elif depth.sell_orders and MULTI_LEVEL_BUY:
            for ap in sorted(depth.sell_orders.keys()):
                if remaining <= 0:
                    break
                vol = abs(depth.sell_orders[ap])
                qty = min(vol, remaining)
                if qty > 0:
                    orders.append(Order('INTARIAN_PEPPER_ROOT', ap, qty))
                    remaining -= qty

        # ── Component 2: passive bid at best_bid+1 ────────────────────
        # Intercepts sell-aggressor flow at ~12 ticks discount vs best ask.
        # Only placed with leftover capacity after the take component.
        # This saves entry cost while the drift guarantee makes every unit
        # at any reasonable price profitable to hold.
        if PASSIVE_BID and remaining > 0 and depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            orders.append(Order('INTARIAN_PEPPER_ROOT', best_bid + 1, remaining))

        return orders


# ================================================================
# MODULE-LEVEL MATH HELPER (avoids statistics import overhead)
# ================================================================

def _std(data: list) -> float:
    """Compute population standard deviation of a list. Returns 0.0 for n<2."""
    n = len(data)
    if n < 2:
        return 0.0
    mean = sum(data) / n
    return (sum((x - mean) ** 2 for x in data) / n) ** 0.5
