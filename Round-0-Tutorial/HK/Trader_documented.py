"""
trader.py  -  IMC Prosperity 4, Round 0 Tutorial

HYPOTHESIS DOCUMENTATION

EMERALDS HYPOTHESIS

Claim:  EMERALDS is a stationary mean-reverting process anchored at
        fair value = 10,000 XIRECS.

Evidence (from EDA):
  - ADF test: p=0.000 on both days. Null of unit root is rejected.
  - Lag-1 autocorrelation: -0.490 (Day -1), -0.485 (Day -2).
  - Mid price std across 10,000 ticks: 0.72 XIRECS.
  - Deviations from 10,000: only ±4 XIRECS, on 3.2-3.3% of ticks.
  - Bot execution prices: ONLY {9992, 10000, 10008}. No exceptions
    across 399 trades on both days combined.

Strategy: Passive market-making at the L1 prices.
  - Normal regime (spread=16): post bid=9992, ask=10008.
  - Narrow regime (spread=8):  post at 10000 on the inside sub-state.
  - Round-trip capture: 16 XIRECS (normal) or 8 XIRECS (narrow).
  - No EMA needed. Fair value is a constant.

ASSUMPTIONS:
  A1. Fair value remains anchored at 10,000 across the scored day.
  A2. Bots continue to trade only at {9992, 10000, 10008}.
  A3. The spread remains bimodal: 16 (normal) or 8 (narrow).

IF ASSUMPTIONS HOLD:
  Bot crosses arrive at ~208/day. We capture 8 XIRECS per unit per
  fill. At mean fill size of 5.5 units, upper-bound PnL ≈ 9,000
  XIRECS/day if we capture all fills.

IF ASSUMPTIONS BREAK:
  Break A1 (FV shifts): Mid price starts trending away from 10,000.
  Detection: rolling 100-tick mean deviates > 2 XIRECS from 10,000.
  Response: widen quote offset to ±2, reduce size, add EMA layer.

  Break A2 (new execution prices): A bot starts posting at e.g. 9995.
  Detection: own_trades records fills at previously unseen prices.
  Response: switch to posting at best_bid+1 / best_ask-1 dynamically.

  Break A3 (spread widens or changes regime): Spread distribution
  gains new values beyond {8, 16}.
  Detection: spread values outside the known set appear.
  Response: do not post inside - match L1 quotes only.



TOMATOES HYPOTHESIS

Claim:  TOMATOES is a weakly mean-reverting process around a slowly
        drifting fair value best tracked by EMA-9.

Evidence (from EDA):
  - ADF test: p=0.259 (Day -1), p=0.143 (Day -2). NOT stationary.
    TOMATOES has a unit root -- it trends.
  - Lag-1 autocorrelation: -0.413 (Day -1), -0.428 (Day -2).
    Negative but weaker than EMERALDS -- tick-level mean reversion
    exists but is overwhelmed by session-level drift.
  - Day -1 drift: -49 XIRECS over 1,000 seconds (downtrend).
  - Day -2 drift: +28 XIRECS over 1,000 seconds (uptrend).
  - EMA-9 selected: MAD=0.72, RMSE=1.01, TrackCorr=0.996.
    EMA-9 wins over EMA-11 on every tracking metric.
    Cross-day MAD=0.543, same as in-sample -- no overfitting.
  - Spread: 13 XIRECS (48%) or 14 XIRECS (45%) on 93% of ticks.
  - EMA-9 ±5 quote: inside L1 bid on 95.0%, inside L1 ask on 95.1%.

Strategy: Passive market-making with EMA-9 as dynamic fair value,
          inventory skew, and a velocity filter for trend protection.

ASSUMPTIONS:
  A1. EMA-9 continues to track the local fair value with MAD < 1.5.
  A2. The spread remains in the 13-14 XIREC range.
  A3. The session has some drift but not a sustained runaway trend
      (as suggested by regime segmentation showing alternating up/flat/
      down windows of ~500 ticks each).
  A4. Bots trade at bid_price_1 or ask_price_1. Never at mid.

IF ASSUMPTIONS HOLD:
  EMA-9 ±5 posts inside L1 on 95% of ticks. Bots fill us passively.
  10 XIRECS round-trip. At 423 bot fills/day, mean qty 3.48 units,
  upper-bound PnL ≈ 14,700 XIRECS/day before inventory costs.

IF ASSUMPTIONS BREAK:
  Break A1 (EMA tracking fails): Price makes a sustained >10 XIREC
  move. EMA residual grows beyond our quote offset.
  Detection: abs(mid - ema) > QUOTE_OFFSET for 20+ consecutive ticks.
  Response: widen to EMA±7, increase skew sensitivity.

  Break A2 (spread narrows to <8): A new bot posts very aggressively.
  Detection: spread < 8 observed.
  Response: switch to EMA±3 to remain inside narrowed spread.

  Break A3 (runaway trend): Price drifts >25 XIRECS from session open
  in one direction without reverting.
  Detection: velocity filter triggers for >50 consecutive ticks.
  Response: suspend new orders on the losing side. Only quote one side
  to facilitate unwind. Resume when velocity normalises.

  Break A4 (fills at non-standard prices): Bots start crossing to
  arbitrary prices between L1 levels.
  Detection: own_trades shows fills away from bid1 or ask1.
  Response: EMA-based quotes already handle this -- no change needed.


RISK CONTROLS (applied to both products)

Position limit:  80 units (from spec). Hard rejection above this.
Skew trigger:    |position| >= 30. Shift losing-side quote past L1.
Hard halt:       |position| >= 60. Remove losing-side order entirely.
Position check:  All orders gated through _remaining_capacity() before
                 appending. If capacity <= 0 on a side, no order placed.
Velocity filter: If 10-tick EMA slope > VELOCITY_THRESHOLD (2.0),
                 new entries suspended. Only existing inventory unwind
                 quotes posted. Fires on 12% of TOMATOES ticks.
"""

import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


# Constants

POSITION_LIMIT   = 80        # hard limit from Round 0 spec
SKEW_TRIGGER     = 30        # begin skewing quotes at this position magnitude
HARD_HALT        = 60        # remove one-side orders entirely above this

# EMERALDS
EM_FAIR_VALUE    = 10_000    # stationary anchor, confirmed by ADF test
EM_QUOTE_NORMAL  = 8         # post at FV±8 = {9992, 10008} = exact L1 prices
EM_QUOTE_NARROW  = 0         # post at FV±0 = 10000 in narrow regime

# TOMATOES
TOM_EMA_SPAN     = 9         # selected: MAD=0.72, cross-day MAD=0.543
TOM_ALPHA        = 2 / (TOM_EMA_SPAN + 1)
TOM_QUOTE_OFFSET = 5         # EMA±5: inside L1 on 95% of ticks, 10 XIREC RT
TOM_SKEW_OFFSET  = 7         # EMA-7 pushes bid outside L1 on 66.5% of ticks
TOM_TIGHT_OFFSET = 3         # used on exit side during hard halt
TOM_VELOCITY_WIN = 10        # ticks over which EMA slope is measured
TOM_VELOCITY_THR = 2.0       # absolute EMA change over 10 ticks to suspend


class Trader:

    def bid(self):
        """Round 2 only -- ignored in all other rounds."""
        return 15

    # Entry point
  
    def run(self, state: TradingState):
        """
        Called every tick. Returns (orders, conversions, traderData).

        traderData carries all inter-tick state as a JSON string:
          {
            "tom_ema":        float,   # current EMA-9 value
            "tom_ema_history": list,   # last TOM_VELOCITY_WIN EMA values
          }
        """
        # Restore state
        trader_state = self._load_state(state.traderData)

        # Read live positions
        pos_em  = state.position.get('EMERALDS', 0)
        pos_tom = state.position.get('TOMATOES',  0)

        # Compute orders
        result: Dict[str, List[Order]] = {}

        if 'EMERALDS' in state.order_depths:
            result['EMERALDS'] = self._trade_emeralds(
                state.order_depths['EMERALDS'], pos_em
            )

        if 'TOMATOES' in state.order_depths:
            result['TOMATOES'], trader_state = self._trade_tomatoes(
                state.order_depths['TOMATOES'], pos_tom, trader_state
            )

        # Serialise state
        trader_data = json.dumps(trader_state)

        return result, 0, trader_data

    # EMERALDS strategy
  
    def _trade_emeralds(
        self, order_depth: OrderDepth, position: int
    ) -> List[Order]:
        """
        Passive market-making at the two provably fillable price points.

        Normal regime (spread=16):
          bid=9992, ask=10008  -- exact L1 prices, the only execution prices.

        Narrow regime (spread=8):
          Sub-state A (bot bid=10000, bot ask=10008): post ask=10000.
          Sub-state B (bot bid=9992,  bot ask=10000): post bid=10000.

        Inventory skew:
          position >= SKEW_TRIGGER:  withhold the bid.
          position <= -SKEW_TRIGGER: withhold the ask.
          |position| >= HARD_HALT:   withhold both; only post the exit side.
        """
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread   = best_ask - best_bid

        want_bid = True
        want_ask = True

        # Inventory skew: withhold the side that would deepen exposure
        if position >= SKEW_TRIGGER:
            want_bid = False
        if position <= -SKEW_TRIGGER:
            want_ask = False

        # Hard halt: if near the limit, only quote the exit side
        if position >= HARD_HALT:
            want_bid = False
            want_ask = True   # force-quote ask to encourage sells
        elif position <= -HARD_HALT:
            want_ask = False
            want_bid = True   # force-quote bid to encourage buys

        if spread == 8:
            # Narrow regime: post at the inside price (10000)
            # Sub-state A: bot ask = 10000 -> we post bid = 10000
            # Sub-state B: bot bid = 10000 -> we post ask = 10000
            if best_ask == EM_FAIR_VALUE and want_bid:
                cap = self._remaining_capacity(position, 'buy')
                if cap > 0:
                    qty = min(cap, -order_depth.sell_orders[best_ask])
                    orders.append(Order('EMERALDS', EM_FAIR_VALUE, qty))

            elif best_bid == EM_FAIR_VALUE and want_ask:
                cap = self._remaining_capacity(position, 'sell')
                if cap > 0:
                    qty = min(cap, order_depth.buy_orders[best_bid])
                    orders.append(Order('EMERALDS', EM_FAIR_VALUE, -qty))

        else:
            # Normal regime: post at L1 prices exactly
            bid_price = EM_FAIR_VALUE - EM_QUOTE_NORMAL   # 9992
            ask_price = EM_FAIR_VALUE + EM_QUOTE_NORMAL   # 10008

            if want_bid:
                cap = self._remaining_capacity(position, 'buy')
                if cap > 0:
                    # Match the volume at L1 bid, capped by remaining capacity
                    l1_vol = order_depth.buy_orders.get(bid_price, 0)
                    qty = min(cap, max(l1_vol, 5))  # post at least 5
                    orders.append(Order('EMERALDS', bid_price, qty))

            if want_ask:
                cap = self._remaining_capacity(position, 'sell')
                if cap > 0:
                    l1_vol = abs(order_depth.sell_orders.get(ask_price, 0))
                    qty = min(cap, max(l1_vol, 5))
                    orders.append(Order('EMERALDS', ask_price, -qty))

        return orders

    # TOMATOES strategy

    def _trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        trader_state: dict,
    ):
        """
        Passive market-making with EMA-9 as dynamic fair value.

        Quote placement:
          Normal:  bid = EMA-5,  ask = EMA+5  (inside L1 on 95% of ticks)
          Skewed:  bid = EMA-7,  ask = EMA+5  (long 30+; EMA-7 outside L1
                                               on 66.5% of ticks -> suppresses)
          Halt:    no bid,       ask = EMA+3  (long 60+; unwind only)

        Velocity filter:
          If |EMA_now - EMA_10_ticks_ago| > 2.0 XIRECS:
            Do not open new inventory on the trend side.
            Only post the anti-trend side to facilitate unwind.

        Drift overlay:
          Signed 10-tick slope signals trend direction.
          When trending down (slope < -1.5): suppress bid slightly.
          When trending up   (slope > +1.5): suppress ask slightly.
        """
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, trader_state

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid      = (best_bid + best_ask) / 2

        # Update EMA
        ema = trader_state.get('tom_ema', None)
        if ema is None:
            ema = mid   # cold start: seed EMA from current mid price
        ema = TOM_ALPHA * mid + (1 - TOM_ALPHA) * ema
        trader_state['tom_ema'] = ema

        # Maintain EMA history for velocity calculation
        ema_history: list = trader_state.get('tom_ema_history', [])
        ema_history.append(ema)
        if len(ema_history) > TOM_VELOCITY_WIN + 1:
            ema_history = ema_history[-(TOM_VELOCITY_WIN + 1):]
        trader_state['tom_ema_history'] = ema_history

        # Velocity filter
        in_breakout = False
        velocity    = 0.0
        if len(ema_history) >= TOM_VELOCITY_WIN:
            velocity    = ema_history[-1] - ema_history[-TOM_VELOCITY_WIN]
            in_breakout = abs(velocity) > TOM_VELOCITY_THR

        # Determine quote prices
        want_bid = True
        want_ask = True
        bid_offset = TOM_QUOTE_OFFSET  # EMA - bid_offset
        ask_offset = TOM_QUOTE_OFFSET  # EMA + ask_offset

        # Inventory skew level 1: soft (position >= SKEW_TRIGGER)
        if position >= SKEW_TRIGGER:
            bid_offset = TOM_SKEW_OFFSET   # EMA-7: outside L1 on 66.5% of ticks
        elif position <= -SKEW_TRIGGER:
            ask_offset = TOM_SKEW_OFFSET

        # Inventory skew level 2: hard halt (position >= HARD_HALT)
        if position >= HARD_HALT:
            want_bid   = False
            ask_offset = TOM_TIGHT_OFFSET  # EMA+3: tightest, most fills
        elif position <= -HARD_HALT:
            want_ask   = False
            bid_offset = TOM_TIGHT_OFFSET

        # Velocity filter: suppress side in direction of breakout
        if in_breakout:
            if velocity > 0:
                # Trending up: suppress ask (don't sell into uptrend inventory)
                # Only protect existing long by keeping bid suppressed
                if position > 0:
                    want_bid = False    # stop adding to long during uptrend
            else:
                # Trending down: suppress bid (don't buy into downtrend)
                if position < 0:
                    want_ask = False    # stop adding to short during downtrend
                want_bid = False        # always stop buying in a downtrend

        bid_price = int(round(ema - bid_offset))
        ask_price = int(round(ema + ask_offset))

        # Safety check: ensure bid < ask
        if bid_price >= ask_price:
            bid_price = int(round(ema - TOM_QUOTE_OFFSET))
            ask_price = int(round(ema + TOM_QUOTE_OFFSET))

        # Submit orders
        if want_bid:
            cap = self._remaining_capacity(position, 'buy')
            if cap > 0:
                orders.append(Order('TOMATOES', bid_price, min(cap, 5)))

        if want_ask:
            cap = self._remaining_capacity(position, 'sell')
            if cap > 0:
                orders.append(Order('TOMATOES', ask_price, -min(cap, 5)))

        return orders, trader_state

    # Helpers 

    def _remaining_capacity(self, position: int, side: str) -> int:
        """
        How many more units can we buy (side='buy') or sell (side='sell')
        before hitting the hard position limit?

        The exchange rejects ALL orders for a product if ANY single order
        would breach the limit if fully filled. This function ensures we
        never submit an order that risks that rejection.
        """
        if side == 'buy':
            return max(0, POSITION_LIMIT - position)
        else:
            return max(0, POSITION_LIMIT + position)

    def _load_state(self, trader_data: str) -> dict:
        """
        Deserialise inter-tick state. Returns empty defaults if the string
        is absent or corrupt (e.g. first tick of the session).
        """
        if not trader_data:
            return {
                'tom_ema'        : None,
                'tom_ema_history': [],
            }
        try:
            return json.loads(trader_data)
        except (json.JSONDecodeError, TypeError):
            return {
                'tom_ema'        : None,
                'tom_ema_history': [],
            }



# HYPOTHESIS BREAK DETECTION CHECKLIST
# (inspect the debug log after each backtest run)

# EMERALDS breaks:
#  [ ] own_trades contains fills at prices other than {9992, 10000, 10008}
#  [ ] Mid price mean drifts > 2 XIRECS from 10,000 over 500 ticks
#  [ ] Spread values outside {8, 16} appear in the log
#  [ ] Position repeatedly hits HARD_HALT (60) -- inventory not clearing

# TOMATOES breaks:
#  [ ] abs(mid - ema) > 6 for more than 20 consecutive ticks
#  [ ] Velocity filter fires on > 25% of ticks (persistent breakout)
#  [ ] Position oscillates between +60 and -60 without clearing
#  [ ] own_trades fills at prices far from bid1 or ask1

# If any box is ticked:
#  1. Print the full trader_state JSON at every tick that triggers it.
#  2. Identify the timestamp range where the break occurred.
#  3. Pull those rows from the price CSV and inspect the book manually.
#  4. Modify the relevant constant (offset, velocity threshold, skew
#     trigger) and re-run the backtest before uploading.

  
