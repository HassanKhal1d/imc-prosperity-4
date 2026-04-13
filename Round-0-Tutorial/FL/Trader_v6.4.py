import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# =============================================================================
# trader_v9 -- four targeted improvements over v6, all grounded in run data
#
# the three live runs (v6, v7, v8) established that spread capture is already
# near-optimal (~88% of theoretical). all pnl variance comes from inventory
# management -- how large positions build, how long they are held, and how
# aggressively they are unwound. these four changes attack that problem.
#
# change 1: skew trigger lowered from 30 to 25
#   the v6 run showed position peaked at -27 and never triggered the skew
#   mechanism at all (trigger=30 never fired). lowering to 25 means the
#   strategy starts discouraging accumulation and unwinding faster, 4 earlier
#   ticks into the inventory build. on the v6 run data this would have
#   activated the skew on 81 ticks (4% of the day) vs 0 ticks with the
#   old threshold. everything downstream (skew shift formula, hard halt) is
#   unchanged -- only the activation point moves.
#
# change 2: time-based unwind aggression
#   the -0.42 lag-1 autocorrelation and near-zero lag-3 autocorrelation means
#   mean reversion happens within 3 ticks or not at all. if the strategy is
#   still holding a large position after 20 consecutive ticks above the skew
#   trigger, the mean reversion window has statistically closed and it is now
#   holding pure directional risk.
#   fix: track consecutive ticks above trigger. after 20 ticks, shift the
#   unwind-side quote 1 extra tick toward mid. after 50 ticks, shift 2 extra
#   ticks. this makes fills faster on the unwind side as time passes without
#   the position normalising. on v6 run data: medium fires on 40 ticks, high
#   fires on 10 ticks.
#   the accumulation-side quote is not touched -- it already has the skew
#   shift pushing it away from mid.
#
# change 3: velocity protection on new trend-aligned exposure
#   the existing velocity filter suppresses the bid when already long in an
#   uptrend, and vice versa. but it does not protect against building new
#   exposure from a flat position. when velocity > 3.5 upward and position
#   is flat or short, posting a normal ask risks the fill creating or
#   extending a short into a continued uptrend.
#   fix: when breakout is detected, add 1 extra tick of offset on the side
#   that would create trend-aligned new inventory. the key distinction from
#   the existing filter: this applies when position is on the SAME side as
#   the trend (would add to it) rather than when position is already extended.
#   only fires when position <= 0 and velocity up, or position >= 0 and
#   velocity down (trend-aligned new inventory). when position is already
#   opposed to the trend (e.g. short in an uptrend), we want to buy back --
#   the bid is left untouched.
#   on v6 run data: fires on 17 ticks upside, 1 tick downside.
#
# change 4: asymmetric sizing when inventory is elevated
#   between the skew trigger and hard halt, both sides currently quote the
#   same base_size. when position is above trigger, we want fills on the
#   unwind side faster and fills on the accumulation side slower.
#   fix: when |position| > TOM_SKEW_TRIGGER, add TOM_UNWIND_SIZE_BOOST units
#   to the unwind side and subtract the same from the accumulation side.
#   spread capture per fill is unchanged (same price logic). only the fill
#   rate differs. minimum accumulation size is capped at 1 to keep quoting.
#   fires on the same 81 ticks as the skew trigger.
#
# what is unchanged from v6:
#   - ema-7 fair value estimator
#   - book-anchored quoting (offset=1)
#   - hard halt at position 50
#   - hard halt unwind price (ema - TOM_UNWIND_OFFSET)
#   - vol window, regime thresholds, regime sizes
#   - velocity breakout detection (threshold 3.5, window 10)
#   - existing velocity filter (suppress want_bid/want_ask)
#   - all emerald logic
# =============================================================================

POSITION_LIMIT          = 80

# emeralds: unchanged from v6
EM_FV                   = 10000
EM_INSIDE               = 7
EM_SKEW_TRIGGER         = 40
EM_SKEW_STEP            = 1
EM_HARD_HALT            = 60

# tomatoes: fair value tracking
TOM_EMA_SPAN            = 7
TOM_ALPHA               = 2 / (TOM_EMA_SPAN + 1)

# tomatoes: volatility estimation
TOM_VOL_WINDOW          = 20
TOM_VOL_CALM_THRESH     = 1.5
TOM_VOL_ACTIVE_THRESH   = 2.5

# tomatoes: regime-adaptive order sizing
TOM_SIZE_CALM           = 6
TOM_SIZE_ACTIVE         = 5
TOM_SIZE_VOLATILE       = 3

# tomatoes: book-anchored quoting offset
TOM_QUOTE_OFFSET        = 1

# tomatoes: inventory management
TOM_SKEW_TRIGGER        = 25     # [changed] was 30 -- v6 never triggered; 25 fires 81 ticks
TOM_SKEW_STEP           = 1
TOM_HARD_HALT           = 50
TOM_UNWIND_OFFSET       = 3
TOM_SKEW_QUOTE_SHIFT    = 1

# tomatoes: velocity breakout filter
TOM_VELOCITY_WIN        = 10
TOM_VELOCITY_THR        = 3.5

# [new] change 2: time-based unwind aggression
# after this many consecutive ticks above skew trigger, shift unwind side
# 1 tick (medium) or 2 ticks (high) closer to mid
TOM_UNWIND_TIME_MEDIUM  = 20     # data: fires on 40 ticks in v6 run
TOM_UNWIND_TIME_HIGH    = 50     # data: fires on 10 ticks in v6 run

# [new] change 3: velocity protection on new trend-aligned exposure
# extra ticks of offset on the side that would create inventory in trend direction
TOM_VEL_PROTECT_OFFSET  = 1

# [new] change 4: asymmetric sizing when inventory elevated
# unwind side gets +this many units, accumulation side gets -this many
TOM_UNWIND_SIZE_BOOST   = 2


class Trader:

    def run(self, state: TradingState):
        trader_state = self._load_state(state.traderData)
        pos_em = state.position.get('EMERALDS', 0)
        pos_tm = state.position.get('TOMATOES', 0)

        result: Dict[str, List[Order]] = {}

        if 'EMERALDS' in state.order_depths:
            result['EMERALDS'] = self._trade_emeralds(
                state.order_depths['EMERALDS'], pos_em
            )

        if 'TOMATOES' in state.order_depths:
            result['TOMATOES'], trader_state = self._trade_tomatoes(
                state.order_depths['TOMATOES'], pos_tm, trader_state
            )

        return result, 0, json.dumps(trader_state)


    # emeralds: identical to v6

    def _trade_emeralds(self, order_depth: OrderDepth, position: int):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid

        if spread == 8:
            if best_bid == EM_FV and position > 0:
                qty = min(position, order_depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order('EMERALDS', best_bid, -qty))
                return orders
            elif best_ask == EM_FV and position < 0:
                qty = min(-position, -order_depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order('EMERALDS', best_ask, qty))
                return orders

        bid_price = EM_FV - EM_INSIDE
        ask_price = EM_FV + EM_INSIDE

        want_bid = True
        want_ask = True

        if position >= EM_SKEW_TRIGGER:
            levels = (position - EM_SKEW_TRIGGER) // EM_SKEW_STEP
            bid_price -= levels
        if position <= -EM_SKEW_TRIGGER:
            levels = (-position - EM_SKEW_TRIGGER) // EM_SKEW_STEP
            ask_price += levels

        if position >= EM_HARD_HALT:
            want_bid = False
        if position <= -EM_HARD_HALT:
            want_ask = False

        if bid_price >= ask_price:
            bid_price = EM_FV - EM_INSIDE
            ask_price = EM_FV + EM_INSIDE

        bid_cap = self._remaining_capacity(position, 'buy')
        ask_cap = self._remaining_capacity(position, 'sell')

        if want_bid and bid_cap > 0:
            orders.append(Order('EMERALDS', int(bid_price), min(bid_cap, 10)))

        if want_ask and ask_cap > 0:
            orders.append(Order('EMERALDS', int(ask_price), -min(ask_cap, 10)))

        return orders


    def _trade_tomatoes(self, order_depth, position, trader_state):
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2.0

        # update ema (identical to v6)
        ema = trader_state.get('tom_ema', None)
        if ema is None:
            ema = mid
        ema = TOM_ALPHA * mid + (1 - TOM_ALPHA) * ema
        trader_state['tom_ema'] = ema

        # update rolling volatility (identical to v6)
        mid_history = trader_state.get('tom_mid_history', [])
        mid_history.append(mid)
        if len(mid_history) > TOM_VOL_WINDOW + 2:
            mid_history = mid_history[-(TOM_VOL_WINDOW + 2):]
        trader_state['tom_mid_history'] = mid_history

        vol = 0.0
        if len(mid_history) >= TOM_VOL_WINDOW + 1:
            returns = []
            for i in range(1, len(mid_history)):
                returns.append(mid_history[i] - mid_history[i - 1])
            returns = returns[-TOM_VOL_WINDOW:]
            if returns:
                mean_r = sum(returns) / len(returns)
                var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                vol = var_r ** 0.5
        trader_state['tom_vol'] = vol

        # update ema history for velocity filter (identical to v6)
        ema_history = trader_state.get('tom_ema_history', [])
        ema_history.append(ema)
        if len(ema_history) > TOM_VELOCITY_WIN + 1:
            ema_history = ema_history[-(TOM_VELOCITY_WIN + 1):]
        trader_state['tom_ema_history'] = ema_history

        # ---------------------------------------------------------------
        # change 2: update skew_ticks counter
        # counts consecutive ticks where |position| > TOM_SKEW_TRIGGER.
        # resets to 0 the moment position returns inside the trigger.
        # used to escalate unwind aggression over time.
        # ---------------------------------------------------------------
        skew_ticks = trader_state.get('tom_skew_ticks', 0)
        if abs(position) > TOM_SKEW_TRIGGER:
            skew_ticks += 1
        else:
            skew_ticks = 0
        trader_state['tom_skew_ticks'] = skew_ticks

        # classify volatility regime (identical to v6)
        if vol < TOM_VOL_CALM_THRESH:
            regime = 'calm'
        elif vol < TOM_VOL_ACTIVE_THRESH:
            regime = 'active'
        else:
            regime = 'volatile'

        # velocity breakout detection (identical to v6)
        in_breakout = False
        velocity = 0.0
        if len(ema_history) >= TOM_VELOCITY_WIN:
            velocity = ema_history[-1] - ema_history[-TOM_VELOCITY_WIN]
            in_breakout = abs(velocity) > TOM_VELOCITY_THR

        # select base size from regime (identical to v6)
        if regime == 'calm':
            base_size = TOM_SIZE_CALM
        elif regime == 'active':
            base_size = TOM_SIZE_ACTIVE
        else:
            base_size = TOM_SIZE_VOLATILE

        # ---------------------------------------------------------------
        # change 4: asymmetric sizing when inventory is elevated
        # when |position| > TOM_SKEW_TRIGGER the strategy wants to fill
        # faster on the unwind side and slower on the accumulation side.
        # unwind side: base_size + TOM_UNWIND_SIZE_BOOST
        # accumulation side: max(1, base_size - TOM_UNWIND_SIZE_BOOST)
        # when |position| <= trigger, both sides use base_size as normal.
        # ---------------------------------------------------------------
        if position > TOM_SKEW_TRIGGER:
            # long: want to sell (unwind) faster, buy (accumulate) slower
            ask_size = base_size + TOM_UNWIND_SIZE_BOOST
            bid_size = max(1, base_size - TOM_UNWIND_SIZE_BOOST)
        elif position < -TOM_SKEW_TRIGGER:
            # short: want to buy (unwind) faster, sell (accumulate) slower
            bid_size = base_size + TOM_UNWIND_SIZE_BOOST
            ask_size = max(1, base_size - TOM_UNWIND_SIZE_BOOST)
        else:
            bid_size = base_size
            ask_size = base_size

        # book-anchored quoting (identical to v6)
        bid_price = best_bid + TOM_QUOTE_OFFSET
        ask_price = best_ask - TOM_QUOTE_OFFSET

        want_bid = True
        want_ask = True

        # inventory skew: proportional shift when position grows large (identical to v6)
        # shifts the ACCUMULATION side away from mid to discourage more filling
        # the unwind side stays at offset=1 until the time-based block below
        if position > TOM_SKEW_TRIGGER:
            excess = position - TOM_SKEW_TRIGGER
            shift = excess * TOM_SKEW_QUOTE_SHIFT
            bid_price -= shift   # long: make bid less attractive (shift down)
        elif position < -TOM_SKEW_TRIGGER:
            excess = -position - TOM_SKEW_TRIGGER
            shift = excess * TOM_SKEW_QUOTE_SHIFT
            ask_price += shift   # short: make ask less attractive (shift up)

        # ---------------------------------------------------------------
        # change 2: time-based unwind aggression
        # when position has been above the skew trigger for many consecutive
        # ticks, mean reversion has likely failed (lag-3 autocorrelation is
        # near zero, meaning the edge expires within 3 ticks). start moving
        # the unwind-side quote progressively toward mid.
        #
        # long (want to sell): unwind side is the ask -- shift ask DOWN
        # short (want to buy): unwind side is the bid -- shift bid UP
        #
        # only applied between trigger and hard halt. hard halt block below
        # overrides everything for extreme positions.
        # ---------------------------------------------------------------
        if TOM_SKEW_TRIGGER < position < TOM_HARD_HALT:
            # long position: unwind by selling, shift ask toward mid over time
            if skew_ticks >= TOM_UNWIND_TIME_HIGH:
                ask_price -= 2    # high aggression: 2 ticks closer to mid
            elif skew_ticks >= TOM_UNWIND_TIME_MEDIUM:
                ask_price -= 1    # medium aggression: 1 tick closer to mid

        elif -TOM_HARD_HALT < position < -TOM_SKEW_TRIGGER:
            # short position: unwind by buying, shift bid toward mid over time
            if skew_ticks >= TOM_UNWIND_TIME_HIGH:
                bid_price += 2    # high aggression: 2 ticks closer to mid
            elif skew_ticks >= TOM_UNWIND_TIME_MEDIUM:
                bid_price += 1    # medium aggression: 1 tick closer to mid

        # hard halt: stop quoting the overweight side, aggressively unwind
        # (identical to v6 -- completely overrides everything above)
        if position >= TOM_HARD_HALT:
            want_bid = False
            ask_price = int(round(ema - TOM_UNWIND_OFFSET))
        elif position <= -TOM_HARD_HALT:
            want_ask = False
            bid_price = int(round(ema + TOM_UNWIND_OFFSET))

        # existing velocity filter (identical to v6)
        # suppresses the side that would add to already-existing trend-aligned inventory
        if in_breakout:
            if velocity > 0 and position > 0:
                want_bid = False
            elif velocity < 0 and position < 0:
                want_ask = False

        # ---------------------------------------------------------------
        # change 3: velocity protection on new trend-aligned exposure
        # distinct from the existing filter above. that filter suppresses
        # the bid when already long in an uptrend (position > 0). this
        # protection makes the dangerous side less attractive when position
        # is flat or opposed to the trend, preventing a SHORT from being
        # built from scratch during an uptrend, or a LONG from scratch
        # during a downtrend.
        #
        # only applies to the side that would CREATE trend-aligned inventory:
        #   velocity > 0 (uptrend) and position <= 0: selling makes us more
        #     short into an uptrend -- make ask harder to fill
        #   velocity < 0 (downtrend) and position >= 0: buying makes us more
        #     long into a downtrend -- make bid harder to fill
        #
        # when position is already opposed (e.g. short in uptrend), we WANT
        # to buy back -- the bid is intentionally left untouched in that case.
        # ---------------------------------------------------------------
        if in_breakout:
            if velocity > 0 and position <= 0:
                ask_price += TOM_VEL_PROTECT_OFFSET
            elif velocity < 0 and position >= 0:
                bid_price -= TOM_VEL_PROTECT_OFFSET

        # sanity: prevent crossed quotes (identical to v6)
        if bid_price >= ask_price:
            bid_price = best_bid + TOM_QUOTE_OFFSET
            ask_price = best_ask - TOM_QUOTE_OFFSET

        # place orders using asymmetric sizes
        bid_cap = self._remaining_capacity(position, 'buy')
        ask_cap = self._remaining_capacity(position, 'sell')

        if want_bid and bid_cap > 0:
            orders.append(Order('TOMATOES', int(bid_price), min(bid_cap, bid_size)))

        if want_ask and ask_cap > 0:
            orders.append(Order('TOMATOES', int(ask_price), -min(ask_cap, ask_size)))

        return orders, trader_state


    def _remaining_capacity(self, position: int, side: str) -> int:
        if side == 'buy':
            return max(0, POSITION_LIMIT - position)
        else:
            return max(0, POSITION_LIMIT + position)


    def _load_state(self, trader_data: str) -> dict:
        default = {
            'tom_ema': None,
            'tom_ema_history': [],
            'tom_mid_history': [],
            'tom_vol': 0.0,
            'tom_skew_ticks': 0,
        }
        if not trader_data:
            return default
        try:
            loaded = json.loads(trader_data)
            for key in default:
                if key not in loaded:
                    loaded[key] = default[key]
            return loaded
        except Exception:
            return default
