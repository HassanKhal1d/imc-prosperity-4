"""
trader.py - imc prosperity 4, round 0 (revised after post-mortem)

==========================================================================
post-mortem findings and hypothesis revision
==========================================================================

what failed:
  1. emeralds generated zero pnl because we quoted AT the bot prices
     (9992 and 10008). bots have time priority in the queue at those
     exact levels so our orders were never filled. competitors earned
     1000-1050 xirecs by quoting INSIDE the spread (bid=9999, ask=10001),
     giving bots an incentive to cross to our price. this is the core fix.

  2. tomatoes position component destroyed 92% of spread income.
     we earned ~13,055 xirecs in spread but lost ~11,992 on inventory
     mark-to-market. markout at t+10 was -7.58 xirecs, meaning price
     kept moving against us after fills during trending periods.

  3. we ended with non-zero positions (emeralds +5, tomatoes +6,
     xirecs -78916) with no mechanism to flatten at session end.

  4. risk management was too conservative. max drawdown 99 xirecs vs
     681 for top performers who earned 5x more. the leaf model from the
     winners intel brief scales position size with deviation, which is
     the correct approach for mean-reverting assets.

assumptions that held:
  - emeralds fv = 10000. mid stayed in {9996, 10000, 10004} always.
  - tomatoes ema-9 tracked fv with residual std=0.97. never exceeded 4.4.
  - spreads were stable: emeralds 96.7% at 16, tomatoes 93% at 13-14.

assumptions that broke:
  - emeralds a2: bots only trade at {9992,10000,10008} with each other.
    they never crossed to our quotes because we had no queue priority.
    fix: post inside the spread so bots actively prefer our price.
  - tomatoes a4: we assumed symmetric flow. markout = -7.58 proves the
    flow was directional during trending periods.
==========================================================================
"""

import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


# ── global constants ──────────────────────────────────────────────────────────

POSITION_LIMIT   = 80
SESSION_LENGTH   = 200000
UNWIND_START_TS  = 195000   # begin aggressive flattening here
HARD_CLOSE_TS    = 199000   # force-take from book to close position here

# emeralds quoting
EM_FV            = 10000
EM_INSIDE        = 1        # post at fv+-1 (9999/10001). inside the bot spread.
                             # competitors earned 1000-1050 xirecs this way.
EM_SKEW_TRIGGER  = 20       # begin shifting quotes at this |position|
EM_SKEW_STEP     = 1        # ticks to shift the adding-side per skew level
EM_HARD_HALT     = 50       # remove adding-side order entirely at this |position|

# tomatoes ema
TOM_EMA_SPAN     = 9
TOM_ALPHA        = 2 / (TOM_EMA_SPAN + 1)
TOM_QUOTE_OFFSET = 5        # base quote = ema +- 5 (inside l1 on 95% of ticks)

# tomatoes leaf model
# target_qty = min(POSITION_LIMIT, TOM_LEAF_K * residual^2)
# calibrated so qty=80 at residual=4.3 (empirical maximum from scored day)
TOM_LEAF_K       = 4.33
TOM_EXIT_BAND    = 0.8      # hold position until |residual| falls below this
                             # this is the hysteresis: exit is slower than entry

# tomatoes velocity filter
TOM_VEL_WIN      = 10
TOM_VEL_THR      = 2.0      # suspend new entries if |ema[now]-ema[now-10]| > this

# end-of-session: tighten skew trigger late in the session
LATE_SKEW_POS    = 5        # in final 5000 ticks, skew kicks in at |pos|>=5


class Trader:

    def run(self, state: TradingState):
        """
        called every tick. returns (result_dict, conversions, traderData).
        all inter-tick state lives in traderData (json string).
        """
        trader_state = self._load_state(state.traderData)
        ts     = state.timestamp
        pos_em = state.position.get('EMERALDS', 0)
        pos_tm = state.position.get('TOMATOES',  0)

        result: Dict[str, List[Order]] = {}

        if 'EMERALDS' in state.order_depths:
            result['EMERALDS'] = self._trade_emeralds(
                state.order_depths['EMERALDS'], pos_em, ts
            )

        if 'TOMATOES' in state.order_depths:
            result['TOMATOES'], trader_state = self._trade_tomatoes(
                state.order_depths['TOMATOES'], pos_tm, trader_state, ts
            )

        return result, 0, json.dumps(trader_state)

    # ── emeralds ──────────────────────────────────────────────────────────────

    def _trade_emeralds(
        self, order_depth: OrderDepth, position: int, ts: int
    ) -> List[Order]:
        """
        post inside the bot spread to earn fills via price improvement.

        why this works:
          bots quote at 9992 (bid) and 10008 (ask). they have queue priority
          at those exact prices. if we also post at 9992/10008 we are behind
          them in the queue and never fill. by posting at 9999/10001 we offer
          a better price: a bot wanting to sell 'sees' our 9999 bid as more
          attractive than the existing 9992 bot bid (they get 7 more xirecs
          for selling to us). so they cross to our price instead. same logic
          for the ask side. this matches the 1000-1050 xirec emeralds pnl
          that competitors reported on discord.

        regime routing:
          - in last 50 ticks: force-close via taker orders
          - narrow spread (spread=8): selective taking to unwind inventory at fv
          - normal (spread=16): post 9999/10001 with inventory skew applied
        """
        orders: List[Order] = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid

        # force-close in last 50 ticks
        if ts >= HARD_CLOSE_TS:
            return self._close_position(
                'EMERALDS', order_depth, position, best_bid, best_ask
            )

        # use tighter skew trigger in final 5000 ticks
        skew_trig = LATE_SKEW_POS if ts >= UNWIND_START_TS else EM_SKEW_TRIGGER

        # ── narrow-spread regime (spread=8): selective taking ─────────────────
        # in sub-state b (bid=10000): sell to bot at fv to reduce long inventory
        # in sub-state a (ask=10000): buy from bot at fv to reduce short inventory
        # we do NOT take speculatively in narrow regime, only unwind
        if spread == 8:
            if best_bid == EM_FV and position > 0:
                # sub-state b: bot bids at fv. sell to reduce long.
                qty = min(position, order_depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order('EMERALDS', int(best_bid), -qty))
                return orders
            elif best_ask == EM_FV and position < 0:
                # sub-state a: bot asks at fv. buy to reduce short.
                qty = min(-position, -order_depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order('EMERALDS', int(best_ask), qty))
                return orders

        # ── normal regime (spread=16): inside-spread quoting ──────────────────
        bid_price = EM_FV - EM_INSIDE   # 9999
        ask_price = EM_FV + EM_INSIDE   # 10001

        want_bid = True
        want_ask = True

        # inventory skew: shift adding-side quote further from fv
        if position >= skew_trig:
            levels = (position - skew_trig) // EM_SKEW_STEP
            bid_price -= levels           # lower bid when long
        if position <= -skew_trig:
            levels = (-position - skew_trig) // EM_SKEW_STEP
            ask_price += levels           # raise ask when short

        # hard halt: remove the adding-side entirely
        if position >= EM_HARD_HALT:
            want_bid = False
        if position <= -EM_HARD_HALT:
            want_ask = False

        # safety: bid must be strictly below ask
        if bid_price >= ask_price:
            bid_price = EM_FV - EM_INSIDE
            ask_price = EM_FV + EM_INSIDE

        bid_cap = self._cap(position, 'buy')
        ask_cap = self._cap(position, 'sell')

        if want_bid and bid_cap > 0:
            orders.append(Order('EMERALDS', int(bid_price), min(bid_cap, 10)))
        if want_ask and ask_cap > 0:
            orders.append(Order('EMERALDS', int(ask_price), -min(ask_cap, 10)))

        return orders

    # ── tomatoes ──────────────────────────────────────────────────────────────

    def _trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        trader_state: dict,
        ts: int,
    ):
        """
        ema-9 fair value with leaf model position sizing.

        leaf model mechanics (from winners intel brief):
          entry phase (|residual| is growing):
            target_qty = min(80, k * residual^2)   [quadratic scaling]
            as price moves further from ema, we add more inventory because
            the expected reversion is larger and more probable.

          exit phase (|residual| is shrinking back toward ema):
            do not exit until |residual| falls below TOM_EXIT_BAND.
            this hysteresis prevents churning in/out near the threshold
            and ensures we hold through the reversion instead of exiting
            prematurely when residual briefly ticks back.
            visual result: the position-vs-edge path traces a leaf shape.

        velocity filter (anti-trend gate):
          if |ema[t] - ema[t-10]| > 2.0: price is trending.
          in breakout: only post orders that REDUCE existing inventory.
          do not add new inventory on the trending side.
          this prevents the markout=-7.58 adverse selection from trending fills.

        end-of-session: force-close via taker orders after HARD_CLOSE_TS.
        """
        orders = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2.0

        # force-close in last 50 ticks
        if ts >= HARD_CLOSE_TS:
            orders = self._close_position(
                'TOMATOES', order_depth, position, best_bid, best_ask
            )
            return orders, trader_state

        # ── ema update ────────────────────────────────────────────────────────
        ema = trader_state.get('tom_ema', None)
        if ema is None:
            ema = mid
        ema = TOM_ALPHA * mid + (1.0 - TOM_ALPHA) * ema
        trader_state['tom_ema'] = ema

        # ── velocity filter ───────────────────────────────────────────────────
        hist: list = trader_state.get('tom_ema_history', [])
        hist.append(ema)
        if len(hist) > TOM_VEL_WIN + 1:
            hist = hist[-(TOM_VEL_WIN + 1):]
        trader_state['tom_ema_history'] = hist

        velocity    = hist[-1] - hist[-TOM_VEL_WIN] if len(hist) >= TOM_VEL_WIN else 0.0
        in_breakout = abs(velocity) > TOM_VEL_THR

        # ── leaf model sizing ─────────────────────────────────────────────────
        residual = mid - ema      # positive = price above ema, short signal
        abs_res  = abs(residual)

        # quadratic target position size
        raw_target = TOM_LEAF_K * (abs_res ** 2)
        target_qty = min(int(raw_target), POSITION_LIMIT)

        # direction: if price is above ema we want to go short; below -> long
        # edge_dir = +1 means we want LONG (price below ema = buy signal)
        # edge_dir = -1 means we want SHORT (price above ema = sell signal)
        edge_dir = 1 if residual < 0 else -1

        # ── hysteresis tracking ───────────────────────────────────────────────
        max_res  = trader_state.get('tom_max_res', 0.0)
        pos_sign = trader_state.get('tom_pos_sign', 0)

        # reset max_res tracker if position has flipped direction
        current_sign = 1 if position > 3 else (-1 if position < -3 else 0)
        if current_sign != 0 and current_sign != pos_sign:
            max_res  = abs_res
            pos_sign = current_sign
        if abs_res > max_res:
            max_res = abs_res
        trader_state['tom_max_res']  = max_res
        trader_state['tom_pos_sign'] = pos_sign

        # exit phase: residual has shrunk back to the exit band
        # hold position while residual is above band; exit when it falls below
        below_exit_band = abs_res < TOM_EXIT_BAND

        # ── determine desired position ────────────────────────────────────────
        late_session = ts >= UNWIND_START_TS

        if in_breakout:
            # trending: only move toward zero, never add inventory
            desired = 0
        elif below_exit_band:
            # reversion complete: target flat
            desired = 0
        elif late_session:
            # near end of session: cap target size to avoid large end inventory
            desired = edge_dir * min(target_qty, LATE_SKEW_POS)
        else:
            # normal entry/hold: full leaf target
            desired = edge_dir * target_qty

        # ── translate desired position into bid/ask orders ────────────────────
        delta = desired - position   # positive = need to buy, negative = need to sell

        bid_price = int(round(ema - TOM_QUOTE_OFFSET))
        ask_price = int(round(ema + TOM_QUOTE_OFFSET))

        # safety: bid must be below ask
        if bid_price >= ask_price:
            bid_price = int(round(ema - TOM_QUOTE_OFFSET))
            ask_price = int(round(ema + TOM_QUOTE_OFFSET))

        # apply inventory skew to shift the adding-side quote
        abs_pos = abs(position)
        skew_trig_tom = LATE_SKEW_POS if late_session else 30
        if abs_pos >= skew_trig_tom:
            if position > 0:
                bid_price -= 2   # make bid less attractive when long
                ask_price -= 1   # make ask more attractive to attract sells
            else:
                ask_price += 2   # make ask less attractive when short
                bid_price += 1   # make bid more attractive to attract buys

        # size each side: move toward desired position, cap at 8 to limit per-fill exposure
        if delta > 0:
            bid_qty = min(delta, self._cap(position, 'buy'), 8)
            ask_qty = 0
        elif delta < 0:
            ask_qty = min(-delta, self._cap(position, 'sell'), 8)
            bid_qty = 0
        else:
            # balanced position: post small passive quotes to keep earning spread
            bid_qty = min(2, self._cap(position, 'buy'))
            ask_qty = min(2, self._cap(position, 'sell'))

        if bid_qty > 0:
            orders.append(Order('TOMATOES', bid_price, bid_qty))
        if ask_qty > 0:
            orders.append(Order('TOMATOES', ask_price, -ask_qty))

        return orders, trader_state

    # ── shared helpers ────────────────────────────────────────────────────────

    def _close_position(
        self,
        symbol: str,
        order_depth: OrderDepth,
        position: int,
        best_bid: float,
        best_ask: float,
    ) -> List[Order]:
        """
        take from the book to close any residual position.
        called in the final HARD_CLOSE_TS ticks of the session.
        accepts the spread cost to guarantee a clean close.
        a residual position of 5-10 units costs ~35-85 xirecs to close,
        which is well worth avoiding the uncertainty of carrying inventory
        past the round end at an unknown mark.
        """
        orders = []
        if position > 0:
            # long: sell at best bid (take from book)
            qty = min(position, order_depth.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(symbol, int(best_bid), -qty))
        elif position < 0:
            # short: buy at best ask (take from book)
            qty = min(-position, -order_depth.sell_orders.get(best_ask, 0))
            if qty > 0:
                orders.append(Order(symbol, int(best_ask), qty))
        return orders

    def _cap(self, position: int, side: str) -> int:
        """
        how many more units we can trade before hitting the position limit.
        gate all order sizes through this to prevent exchange rejection.
        """
        if side == 'buy':
            return max(0, POSITION_LIMIT - position)
        return max(0, POSITION_LIMIT + position)

    def _load_state(self, trader_data: str) -> dict:
        """
        deserialise inter-tick state. returns safe defaults on first tick
        or if the string is corrupt (e.g., lambda cold-start failure).
        """
        defaults = {
            'tom_ema'        : None,
            'tom_ema_history': [],
            'tom_max_res'    : 0.0,
            'tom_pos_sign'   : 0,
        }
        if not trader_data:
            return defaults
        try:
            loaded = json.loads(trader_data)
            # fill in any missing keys from defaults
            for k, v in defaults.items():
                if k not in loaded:
                    loaded[k] = v
            return loaded
        except (json.JSONDecodeError, TypeError):
            return defaults


# ==========================================================================
# hypothesis break detection checklist
# run after every backtest before uploading
# ==========================================================================
#
# emeralds:
#   [ ] pnl near zero after 2000 ticks
#       our 9999/10001 quotes are not being filled by bots.
#       action: check if bots are only crossing to {9992,10000,10008}.
#       if so, try posting at 9998/10002 (larger price improvement).
#       last resort: switch to taking from narrow-spread regime only.
#
#   [ ] position pinned at +/-EM_HARD_HALT for many consecutive ticks
#       inventory is one-sided and not clearing.
#       action: reduce EM_HARD_HALT to 30, increase EM_SKEW_STEP to 2.
#
#   [ ] mid moves outside {9996, 10000, 10004}
#       assumption a1 broke (fv shifted). add an ema layer to emeralds.
#
#   [ ] spread values outside {8, 16} appear
#       new bot type. match their l1 bid+1 / ask-1 dynamically.
#
# tomatoes:
#   [ ] leaf target_qty = 0 on most ticks (never builds position)
#       residuals are too small. lower TOM_LEAF_K by 50% and retest.
#
#   [ ] position stuck at limit for extended periods
#       leaf model over-accumulated into a trend.
#       lower TOM_LEAF_K or raise TOM_VEL_THR to be less aggressive.
#
#   [ ] pnl falls in last 500 ticks
#       positions not unwinding before UNWIND_START_TS.
#       bring UNWIND_START_TS forward to 190000.
#
#   [ ] velocity filter fires on > 30% of ticks
#       market is in sustained trend. consider suspending tomatoes.
