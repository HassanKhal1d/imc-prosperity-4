import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# ============================================================================
# ROUND 1 STRATEGY v3 — ASH_COATED_OSMIUM + INTARIAN_PEPPER_ROOT
#
# ── CHANGES FROM v2 AND THE EMPIRICAL EVIDENCE BEHIND EACH ──────────────────
#
# [CONFIRMED] H4: OSM_INSIDE = 7 or 8 vs bid+1/ask-1
#   Test: computed expected_edge × priority for each approach on 27,644 ticks.
#   bid+1/ask-1: combined score 14.79 (7.18 bid + 7.60 ask)
#   inside=6:    combined score  7.57  (60.5% / 65.7% priority)
#   inside=7:    combined score  7.75  (51.4% / 59.3% priority)
#   inside=8:    combined score  7.62  (43.5% / 51.7% priority)
#   VERDICT: bid+1/ask-1 is 91% superior to the best fixed-inside.
#   Fixed quotes lose queue priority whenever the spread moves.
#   DECISION: Keep bid+1/ask-1 unchanged.
#
# [CONFIRMED] H5: Narrow spread (<14) logic
#   8.36% of ticks have spread < 14. bid+1/ask-1 quotes INSIDE even these:
#     spread=5 → bid+1=9998, ask-1=10001. Edge 2/1 ticks. Still profitable.
#   Taking misvalued levels still applies at all spreads.
#   Skipping narrow spreads would forfeit 8.36% of opportunities for zero gain.
#   DECISION: No narrow spread skip. bid+1/ask-1 handles them naturally.
#
# ─── NEW KEY FINDING: Z-SCORE OVERLAY HURTS PASSIVE MM ─────────────────────
#
# [CRITICAL FIX] Z-score removed from passive quote sizing.
#
#   Realistic backtest (fills from actual bot trade timestamps):
#     v2 with z-overlay:    73,630 simulated OSM PnL (3 days)
#     v3 without z-overlay: 92,961 simulated OSM PnL (+26.3% improvement)
#
#   ROOT CAUSE OF v2's ERROR:
#     When z > 2.0 (price at ~10004), v2 reduces bid size by 75% (scale=0.25).
#     But our passive bid is at 9993 — a full 11 ticks BELOW the current mid.
#     A bot sell hitting our 9993 bid earns us 7 ticks regardless of where the mid is.
#     The z-score predicts future mid direction, NOT the profitability of the passive
#     entry at 9993. A fill at 9993 is profitable with or without mean reversion.
#     So reducing our passive bid when z is high means we forfeit profitable fills
#     at exactly the moments when the spread between mid and our quote is widest.
#
#   Z-SCORE IS NOW KEPT FOR JUMP ENTRIES ONLY:
#     Jump entries are TAKER orders — we cross the spread and pay the spread cost.
#     For jumps, z-score provides useful confirmation: a down-jump when z > 2.5
#     is a stronger reversion signal (two independent signals agree).
#     A down-jump when z < -1 might be a momentum continuation instead.
#     DECISION: Gate jump entries with combined z + jump signal.
#
# ─── PARAMETER RETUNING ─────────────────────────────────────────────────────
#
# Systematic grid search over gamma × hard_limit × jump_thresh × jump_size
# on 3-day realistic simulation (1265 actual bot trades as fill events):
#
#   gamma:      0.10  (vs 0.12 in v2 — lighter skew, more fills at extreme positions)
#   hard_limit: 75    (vs 72 in v2 — allows marginally more inventory)
#   jump_thresh: 2.5  (confirmed from v2 — maximises edge × frequency product)
#   jump_size:  20    (vs 20 in v2 — unchanged)
#   jump_max_pos: 60  (unchanged)
#
#   mm_calm=18, mm_active=12, mm_volatile=6 (unchanged — regime sizes confirmed)
#
# ─── METRIC ANALYSIS: BEST OBSERVED (14352) vs v2 (10113) ──────────────────
#
#   Key metrics:
#     Best: PnL=14352, max_DD=609, recovery=23.57, avg_fill=2.32
#     v2:   PnL=10113, max_DD=553, recovery=18.29, avg_fill=5.97
#
#   Interpretation:
#     avg_fill 5.97 (v2) vs 2.32 (best): best fills smaller sizes, more frequently
#     Recovery ratio 23.57 vs 18.29: best earns 29% more PnL per unit of drawdown
#     Best takes MORE risk (DD=609>553) yet recovers FASTER (ratio higher)
#
#   Implied: best strategy does ~3.65x more fill events at 0.39x the size.
#   This is consistent with a strategy that cycles inventory faster, spending
#   less time at the position limit. Our backtest confirms: removing z-overlay
#   gives more fills, faster cycling, and higher total PnL.
#
#   The remaining gap (v3 target ~3200 OSM vs best ~7000 OSM) likely reflects:
#     1. Best using a different fill model that captures more bot volume
#     2. Best's strategy on Pepper potentially more aggressive
#     3. Multi-day compounding effects
#
# ─── UNCHANGED FROM v2 ──────────────────────────────────────────────────────
#   - bid+1/ask-1 adaptive quoting (empirically dominant over all fixed-inside)
#   - Regime detection: calm < 3.7, volatile ≥ 5.0 (52.9%/33.4%/13.6%)
#   - Take phase: sweep all asks < FV, all bids > FV
#   - FV flush: clear inventory at FV (break-even)
#   - Pepper strategy: take best ask + passive bid+1 (92.3% drift efficiency)
# ============================================================================


# ═══ POSITION LIMITS ══════════════════════════════════════════════════════════
POSITION_LIMIT = 80

# ═══ FEATURE FLAGS ════════════════════════════════════════════════════════════
TAKE_ENABLED   = True   # take misvalued levels (ask<FV or bid>FV)
TAKE_AT_FV     = True   # flush inventory at break-even when bid/ask == FV
JUMP_ENABLED   = True   # counter-trend entry on 2.5σ price jumps
# Z_OVERLAY removed: see analysis above — it was hurting passive MM by -26%
# H4/H5 tested and rejected: bid+1/ask-1 is empirically superior to all fixed-inside

# ═══ OSMIUM CORE PARAMETERS (retuned from grid search) ════════════════════════
OSM_FV          = 10000  # ADF-confirmed stationary mean
OSM_GAMMA       = 0.10   # A-S inventory skew (retuned from 0.12 — lighter, more fills)
                          # At pos=50: reservation shifts -5 ticks
OSM_HARD_LIMIT  = 75     # stop passive MM when |pos| ≥ this (retuned from 72)
MIN_INSIDE      = 2      # sanity floor: bid < ask − MIN_INSIDE always

# ═══ VOLATILITY REGIME THRESHOLDS (calibrated, unchanged) ════════════════════
# Rolling 20-tick std of mid-price changes
# Proportions: 52.9% calm / 33.4% active / 13.6% volatile
CALM_THRESH     = 3.7    # below = calm
VOL_THRESH      = 5.0    # above = volatile

# MM size per regime (unchanged from v2)
MM_SIZE_CALM    = 18
MM_SIZE_ACTIVE  = 12
MM_SIZE_VOL     = 6

# ═══ JUMP DETECTION (retuned, now gated by z-score for higher confidence) ════
# 2.5σ jump: up-jump E[fwd5]=−5.41, dn-jump E[fwd5]=+5.62 (n=1262 events)
JUMP_THRESH     = 2.5    # |z_change| threshold
JUMP_SIZE       = 20     # units per jump entry
JUMP_MAX_POS    = 60     # only trade jumps if |position| ≤ this
# NEW: z-score gate for jumps — only enter a reversion jump if z confirms direction
# Up-jump (expect fall) is stronger if z > 0 (price already above FV)
# Down-jump (expect rise) is stronger if z < 0 (price already below FV)
JUMP_Z_CONFIRM  = True   # require z-score to agree with jump direction
JUMP_Z_MIN      = 0.5    # minimum |z_score| for confirmed jump (weak gate)

# ═══ PEPPER PARAMETERS (unchanged) ═══════════════════════════════════════════
PASSIVE_BID     = True   # bid at best_bid+1 to capture sell-aggressor flow
MULTI_LEVEL_BUY = False  # empirically net-negative: disabled


class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # ── Restore rolling state ──────────────────────────────────────
        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        osm_mid_hist = ts.get('osm_mid', [])   # last 50 mids
        osm_chg_hist = ts.get('osm_chg', [])   # last 20 tick changes

        pos_pep = state.position.get('INTARIAN_PEPPER_ROOT', 0)
        pos_osm = state.position.get('ASH_COATED_OSMIUM', 0)

        # ── Update OSM mid history ─────────────────────────────────────
        if 'ASH_COATED_OSMIUM' in state.order_depths:
            current_mid = self._mid(state.order_depths['ASH_COATED_OSMIUM'])
            if current_mid is not None:
                if osm_mid_hist:
                    osm_chg_hist.append(current_mid - osm_mid_hist[-1])
                osm_mid_hist.append(current_mid)
                if len(osm_mid_hist) > 50:
                    osm_mid_hist = osm_mid_hist[-50:]
                if len(osm_chg_hist) > 20:
                    osm_chg_hist = osm_chg_hist[-20:]

        # ── Compute signals ────────────────────────────────────────────
        regime      = self._regime(osm_chg_hist)
        z_score     = self._z_score(osm_mid_hist)
        jump_signal = self._jump_signal(osm_chg_hist)

        # ── Trade OSM ─────────────────────────────────────────────────
        if 'ASH_COATED_OSMIUM' in state.order_depths:
            result['ASH_COATED_OSMIUM'] = self._trade_osmium(
                state.order_depths['ASH_COATED_OSMIUM'],
                pos_osm, regime, z_score, jump_signal,
            )

        # ── Trade Pepper ──────────────────────────────────────────────
        if 'INTARIAN_PEPPER_ROOT' in state.order_depths:
            result['INTARIAN_PEPPER_ROOT'] = self._trade_pepper(
                state.order_depths['INTARIAN_PEPPER_ROOT'], pos_pep
            )

        # ── Persist state ──────────────────────────────────────────────
        new_td = json.dumps({'osm_mid': osm_mid_hist, 'osm_chg': osm_chg_hist})
        return result, 0, new_td

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _mid(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

    def _regime(self, chg_hist: list) -> str:
        """
        3-regime classifier from rolling 20-tick std of mid changes.
        Calibrated proportions: 52.9% calm / 33.4% active / 13.6% volatile.
        """
        if len(chg_hist) < 5:
            return 'active'
        vol = _std(chg_hist[-min(len(chg_hist), 20):])
        if vol < CALM_THRESH:
            return 'calm'
        if vol < VOL_THRESH:
            return 'active'
        return 'volatile'

    def _z_score(self, mid_hist: list) -> float:
        """
        Z-score of current mid vs FV, normalised by rolling 50-tick std.
        Used only for jump-entry gating (NOT for passive MM sizing — see analysis).
        Empirical: z > 2.0 → E[fwd5] = -1.93 ticks (r = -0.32, p ≈ 0).
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
        Z-score of latest tick change relative to rolling 20-tick std.
        At 2.5σ: up-jump E[fwd5] = -5.41, dn-jump E[fwd5] = +5.62 (n=1262).
        """
        if len(chg_hist) < 5:
            return 0.0
        sigma = _std(chg_hist[-min(len(chg_hist) - 1, 20):])
        if sigma < 0.01:
            return 0.0
        return chg_hist[-1] / sigma

    # ─────────────────────────────────────────────────────────────────
    # OSM: TAKE-AND-MAKE + BID+1/ASK-1 + REGIME + JUMP
    # ─────────────────────────────────────────────────────────────────

    def _trade_osmium(
        self,
        depth: OrderDepth,
        position: int,
        regime: str,
        z_score: float,
        jump_signal: float,
    ) -> List[Order]:

        orders: List[Order] = []
        buy_room  = POSITION_LIMIT - position
        sell_room = POSITION_LIMIT + position

        # ── PHASE 0: Jump reversion ─────────────────────────────────
        # Entered as TAKER (we cross the spread). Z-score gating improves
        # signal quality: a jump in the same direction as z-score is stronger.
        #
        # Logic:
        #   Up-jump (price spiked up) + z > 0 (price above FV) → SELL confirmed
        #   Up-jump + z < -JUMP_Z_MIN → contradictory, skip
        #   Down-jump + z < 0 → BUY confirmed
        #   Down-jump + z > JUMP_Z_MIN → contradictory, skip
        if JUMP_ENABLED and abs(position) <= JUMP_MAX_POS:
            if jump_signal > JUMP_THRESH and sell_room > 0:
                # Skip if z contradicts (price is already low: up-jump likely noise)
                z_ok = (not JUMP_Z_CONFIRM) or (z_score > -JUMP_Z_MIN)
                if z_ok and depth.buy_orders:
                    best_bid = max(depth.buy_orders)
                    if best_bid >= OSM_FV - 3:
                        qty = min(JUMP_SIZE, sell_room)
                        orders.append(Order('ASH_COATED_OSMIUM', best_bid, -qty))
                        sell_room -= qty

            elif jump_signal < -JUMP_THRESH and buy_room > 0:
                z_ok = (not JUMP_Z_CONFIRM) or (z_score < JUMP_Z_MIN)
                if z_ok and depth.sell_orders:
                    best_ask = min(depth.sell_orders)
                    if best_ask <= OSM_FV + 3:
                        qty = min(JUMP_SIZE, buy_room)
                        orders.append(Order('ASH_COATED_OSMIUM', best_ask, qty))
                        buy_room -= qty

        # ── PHASE 1: Take misvalued levels ──────────────────────────
        # Sweep all asks strictly below FV (guaranteed edge vs 10000).
        # Sweep all bids strictly above FV (guaranteed edge).
        # Applies at ALL spreads — narrow spreads have 43.6% mispriced (EDA).
        if TAKE_ENABLED:
            if depth.sell_orders:
                for ap in sorted(depth.sell_orders):
                    if ap >= OSM_FV or buy_room <= 0:
                        break
                    qty = min(abs(depth.sell_orders[ap]), buy_room)
                    if qty > 0:
                        orders.append(Order('ASH_COATED_OSMIUM', ap, qty))
                        buy_room -= qty

            if depth.buy_orders:
                for bp in sorted(depth.buy_orders, reverse=True):
                    if bp <= OSM_FV or sell_room <= 0:
                        break
                    qty = min(depth.buy_orders[bp], sell_room)
                    if qty > 0:
                        orders.append(Order('ASH_COATED_OSMIUM', bp, -qty))
                        sell_room -= qty

        # ── PHASE 2: Inventory flush at FV ──────────────────────────
        # Sell at FV when long (break-even, frees capacity for future fills).
        # Buy at FV when short.
        if TAKE_AT_FV:
            if position > 0 and OSM_FV in depth.buy_orders and sell_room > 0:
                qty = min(depth.buy_orders[OSM_FV], position, sell_room)
                if qty > 0:
                    orders.append(Order('ASH_COATED_OSMIUM', OSM_FV, -qty))
                    sell_room -= qty

            if position < 0 and OSM_FV in depth.sell_orders and buy_room > 0:
                qty = min(abs(depth.sell_orders[OSM_FV]), -position, buy_room)
                if qty > 0:
                    orders.append(Order('ASH_COATED_OSMIUM', OSM_FV, qty))
                    buy_room -= qty

        # ── PHASE 3: Passive MM — bid+1/ask-1 with A-S skew ─────────
        # bid+1/ask-1 earns avg 7.09 ticks per half-turn (14.18 per round trip).
        # This is 91% better than any fixed inside approach due to:
        #   1. Always maintains inside-queue priority (100% of ticks)
        #   2. Adapts to wider spreads (18-19 ticks, 25% of time) for extra edge
        #   3. Correctly handles narrow spreads without special logic
        #
        # A-S inventory skew: shifts both quotes via reservation price.
        # gamma=0.10 (retuned): lighter skew → more passive fills at moderate positions.
        # At pos=50: reservation shifts -5 ticks → quotes shift with it.
        #
        # NOTE: Z-score NOT applied to passive sizing.
        # Reason: passive bid at 9993 earns 7 ticks regardless of current mid level.
        # Reducing bid size when mid is at 10004 (z>2) forfeits profitable fills
        # from bots selling at 9993 — the bot's decision is independent of z.
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        # Base quotes: inside by 1 tick from each side
        our_bid = best_bid + 1
        our_ask = best_ask - 1

        # A-S reservation price caps/floors
        reservation = OSM_FV - OSM_GAMMA * position
        our_bid = min(our_bid, int(reservation) - 1)
        our_ask = max(our_ask, int(reservation) + 1)

        # Hard safety: passive quotes must not cross FV
        our_bid = min(our_bid, OSM_FV - 1)
        our_ask = max(our_ask, OSM_FV + 1)

        # Sanity gap
        if our_bid >= our_ask - MIN_INSIDE:
            our_bid = OSM_FV - 3
            our_ask = OSM_FV + 3

        if our_bid >= our_ask:
            return orders

        # Regime-based size (z-score scaling removed)
        if regime == 'calm':
            base_size = MM_SIZE_CALM    # 18
        elif regime == 'active':
            base_size = MM_SIZE_ACTIVE  # 12
        else:
            base_size = MM_SIZE_VOL     # 6

        # Hard position limit gate
        want_bid = abs(position) < OSM_HARD_LIMIT or position < 0
        want_ask = abs(position) < OSM_HARD_LIMIT or position > 0

        bid_size = max(0, min(base_size, buy_room))
        ask_size = max(0, min(base_size, sell_room))

        if want_bid and bid_size > 0:
            orders.append(Order('ASH_COATED_OSMIUM', our_bid,  bid_size))

        if want_ask and ask_size > 0:
            orders.append(Order('ASH_COATED_OSMIUM', our_ask, -ask_size))

        return orders

    # ─────────────────────────────────────────────────────────────────
    # PEPPER ROOT: aggressive accumulation + hold
    # Drift: 0.1 XIREC/tick confirmed across all 3 days (0.1003, 0.0999, 0.1002)
    # Strategy: reach position limit 80 ASAP, then hold for maximum drift capture
    # ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL STD HELPER (avoids import overhead)
# ─────────────────────────────────────────────────────────────────────────────

def _std(data: list) -> float:
    """Population std. Returns 0.0 for n < 2."""
    n = len(data)
    if n < 2:
        return 0.0
    mean = sum(data) / n
    return (sum((x - mean) ** 2 for x in data) / n) ** 0.5
