# IMC Prosperity 4 — Kernel Wizards write up

> **Final Rank: 934 / 18,803** (top 5%)
> **Peak Rank: 643** (end of Round 3)

This repository documents my run through IMC Prosperity 4, the algorithmic trading
simulation hosted by IMC Trading. It contains the code, the EDA notebooks, and — more
importantly — a candid record of what I learned about quantitative finance at the
practical level: what actually matters when alpha has to survive contact with a live
order book.

---

## Table of Contents

- [Final Result](#final-result)
- [The Story](#the-story)
- [Round-by-Round Breakdown](#round-by-round-breakdown)
- [The R4 Reversal: A Lesson in Model Risk](#the-r4-reversal-a-lesson-in-model-risk)
- [The R2 Manual Win: Game Theory + Monte Carlo](#the-r2-manual-win-game-theory--monte-carlo)
- [Technical Capabilities Developed](#technical-capabilities-developed)
- [The Timo Diehm Pivot: First Principles](#the-timo-diehm-pivot-first-principles)
- [What I'd Do Differently](#what-id-do-differently)

---

## Final Result

| Metric | Value |
| --- | --- |
| Final rank | **934 / 18,803** (top 5%) |
| Peak rank | **643** (end of R3) |
| Number of rounds | 5 + manual trading |
| Best single round | R2 manual: 210k vs 230k modelled |
| Worst single round | R4: IV smile miscalibration, model risk realised |

The peak-to-final trajectory is the most important fact on this page. We climbed
into the top 4% by the end of R3 and gave a meaningful chunk of it back in R4. Most
of this README exists to explain *why*, and what I now do differently.

---

## The Story

I went into Prosperity treating it like a coding competition with a finance theme.
I came out of it understanding that it is, very precisely, a **model-risk**
competition with a coding component. Anyone can write a market-making loop; the
quants who win are the ones whose mental model of the data-generating process
matches the real one, and who know what to do when it doesn't.

Rounds 1 through 3 went well because the assets were well-behaved relative to my
priors: a stable asset around a known fair value, a noisy mean-reverting asset, and
an ETF-vs-components statistical arbitrage. By the end of R3 we were sitting at
**643rd**. Round 4 — options on a mean-reverting underlying — was where the wheels
came off, and it came off for a reason that is now burned into how I approach every
modelling problem: **I priced the options as though the underlying followed a
random walk when it actually followed a mean-reverting (Ornstein-Uhlenbeck)
process.** Black-Scholes assumes geometric Brownian motion. The underlying did not.
The implied vol smile I calibrated was therefore systematically miscalibrated, and
the strategy paid for that mismatch tick-by-tick until R4 was over.

Round 5 was a partial recovery — by then I had absorbed the lesson and rebuilt the
EDA pipeline from scratch on first principles before writing a single line of
trading logic. The notebook `R5_EDA.ipynb` is a record of that pivot.

We finished at **934**. Not the rank I was hoping for in R3, not the rank I feared
after R4. The competition's real value was in the asymmetry: every round of profit
felt like a confirmation of skill, and every round of loss was an unambiguous,
P&L-quantified piece of feedback about which of my assumptions were wrong.

---

## Round-by-Round Breakdown

### Round 1 — Trading Groundwork
**Stable asset (EMERALDS):** market-making around a known fair value of 10,000.
Captured spread with inside-quote logic, inventory skew, and a hard halt on
position limits. The strategy is in `Trader_v10_skew_optimised.py` —
parameters were chosen via cross-validated walk-forward analysis (120 combinations
across 8 folds) rather than hand-tuned, which became the template for every later
parameter choice.

**Noisy asset (TOMATOES):** a mean-reverting Ornstein-Uhlenbeck process. Z-score
entries, EMA fair-value tracking, regime-adaptive sizing (calm / active / volatile),
and a velocity-breakout filter to stand down when the price was clearly in a
directional move and we were on the wrong side.

### Round 2 — Statistical Arbitrage
ETF vs. components. Cointegration via OLS, traded the spread on z-score deviations.
Standard playbook — the real fight in R2 was the manual trading section, see below.

### Round 3 — Options
Black-Scholes pricing on the volatility-extract underlying. Delta hedging,
Greeks-aware position management, IV smile calibration. End-of-round rank: 643.
This was the high-water mark.

### Round 4 — Options, Continued (and the Reversal)
The same options framework applied to a new underlying. **The underlying was
mean-reverting, not log-normal.** Black-Scholes priced calls assuming the price
drifts under GBM; the actual process pulled prices back to a long-run mean. My IV
smile was calibrated to a model whose assumptions were violated, so every quote
embedded a systematic error. The PnL attribution post-mortem in
`PnL_Analysis_-_v5_HK.ipynb` shows where the bleed came from.

The fix, after the round closed, was a stationary-OU call-pricing formula:

```
C(K) = (θ - K) · Φ(d) + σ_stat · φ(d),   d = (θ - K) / σ_stat
```

This is implemented in `FinalTrader_v3.py` as `ou_call_fair_value()`. It would have
been the right model if I had run the stationarity and half-life diagnostics on the
underlying *before* writing the pricing code. I did not. That is the core lesson of
this competition for me.

### Round 5 — The Final Stretch
50 new products across 10 named categories, with a deliberately incomplete brief.
Position limit of 10 per product. Three days of price and trade data.

I rebuilt the EDA workflow from the ground up. `R5_EDA.ipynb` is a 7-section
empirical discovery exercise covering, in order:

1. **Foundation** — what is verifiable vs. hypothesised before any analysis runs
2. **Per-product DGP analysis** — stationarity (ADF), Hurst exponent, OU half-life,
   ACF of signed and squared returns (volatility clustering), jump detection
3. **Microstructure analysis** — spread distribution and regimes, order book depth
   and imbalance, the Frankfurt Hedgehogs *wall-mid* construction, one-sided book
   detection, order flow imbalance (OFI), aggressor inference from trade prices,
   trade size distribution
4. **Cross-product structure** — full 50×50 correlation matrix, within-category
   correlation, hierarchical clustering tested against the named categories,
   pairwise cointegration, per-category PCA
5. **Lead-lag analysis** — Granger causality on the most promising pairs
6. **Strategy hypothesis playbook** — per-cluster archetype mapping, FV estimator
   candidates, inventory rules, lead-lag monetisation, Bollinger/z-score specs,
   OFI monetisation, vol-regime gating, **falsification criteria** for each idea
7. **Risk and robustness review**

Every claim in that notebook is grounded in something measurable from the three
days of data. Every parameter range came with a confidence band. No assumption was
treated as fact until it had been tested.

This is the workflow I should have used in R3 going into R4.

---

## The R4 Reversal: A Lesson in Model Risk

R4 cost real rank. The autopsy is worth writing down explicitly because the failure
mode is not exotic — it is one of the canonical ways quant strategies die in
production.

**What I did wrong, in order:**

1. Took a working framework (Black-Scholes from R3) and ported it to a new
   underlying without re-validating the underlying's data-generating process
2. Calibrated an IV smile to fit observed option prices, treating the residual
   structure as smile shape rather than as evidence the pricing model itself was
   wrong
3. Did not run an ADF test, did not estimate a Hurst exponent, did not fit an
   Ornstein-Uhlenbeck half-life on the underlying before pricing options on it
4. Trusted the calibration because the in-sample fit looked good — classic
   overfitting masking a structural model error

**What was actually true:**

The underlying was an OU process with a long-run mean θ ≈ 5247.43 and a stationary
standard deviation σ_stat ≈ 17.0. Calls on a stationary underlying have a closed
form that is *not* Black-Scholes. A GBM-priced call on a stationary process will
systematically misprice deep ITM and deep OTM strikes in opposite directions, which
is exactly what the post-mortem showed.

**What I'd check first now, before any options work:**

- ADF p-value on the underlying (is it stationary?)
- Hurst exponent (is it mean-reverting, random-walk, or trending?)
- OU half-life via AR(1) regression (if mean-reverting, on what timescale?)
- Variance ratio test at multiple horizons (cross-check)

Any one of these would have flagged the problem in five minutes.

---

## The R2 Manual Win: Game Theory + Monte Carlo

The manual trading task in R2 had a game-theoretic component — payouts depended on
the distribution of choices made by other competitors. The naive approach is to
pick the choice that maximises payoff under a uniform prior over opponents. The
better approach is to model the opponents.

I did the latter. The pipeline:

1. **Segmentation.** Pulled the public rankings data from R1 to identify rough
   skill segments in the competitor population. Top performers behave differently
   from median performers, who behave differently from the long tail.
2. **Behavioural priors.** Assigned a game-theoretic decision rule to each segment
   — broadly, sophisticated players play closer to a Nash mixed strategy, median
   players anchor on focal points, the tail behaves close to uniform random.
3. **Monte Carlo.** Sampled opponent profiles from the segmented distribution,
   evaluated my candidate choices against each draw, and built a full distribution
   of PnL outcomes per choice.
4. **Decision.** Picked the choice with the best risk-adjusted expectation, with
   E[PnL] = **230k**.

Realised PnL was **210k**. That is a 9% miss on a forecast built from segmented
behavioural priors and a Monte Carlo over thousands of opponent draws — well inside
the simulation's standard error.

The reason this matters for the rest of the competition is that it was the one
moment where I followed the workflow I now believe in: I refused to assume, I
modelled the actual mechanism (opponent behaviour), I quantified uncertainty, and I
verified the prediction against reality. R4 is what happens when I don't.

---

## Technical Capabilities Developed

These are the skills the competition forced me to develop or sharpen:

**Market microstructure**
- Order book mechanics, matching priority, why "orders fill at *your* price, not the market price" is a strategy-defining constraint
- Spread distribution analysis, regime detection, one-sided book detection
- Order flow imbalance (OFI) as a short-horizon predictor
- Wall-mid construction (Frankfurt Hedgehogs technique) for robust fair-value estimation when the inside book is noisy
- Aggressor inference from trade prices

**Fair-value modelling**
- Stable assets (constant FV, spread capture)
- Mean-reverting assets (OU process, EMA trackers, z-score signals)
- Cointegrated baskets (ETF vs. components, OLS hedge ratios)
- Options on GBM (Black-Scholes) vs. options on OU (closed-form stationary call)

**EDA and statistical testing**
- ADF stationarity tests, Hurst exponent, OU half-life estimation
- Autocorrelation of signed and squared returns (vol clustering diagnostics)
- Jump detection (return z-scores, threshold tuning)
- Distribution moments, fat-tail diagnostics, Jarque-Bera
- Hierarchical clustering on correlation matrices, PCA per category
- Granger causality for lead-lag

**Algorithmic trading logic**
- Market-making with inventory skew and circuit breakers
- Mean-reversion entries with falsification criteria
- Volatility-regime-adaptive order sizing (calm / active / volatile)
- Velocity-breakout filters to stand down during directional moves
- Multi-component strategies (taker for cheap inventory, maker for spread capture)

**The full pipeline**
Signal generation → algo trading logic → backtesting →
PnL attribution → hyperparameter tuning → overfitting tests →
walk-forward validation. Every parameter in `Trader_v10_skew_optimised.py`
was chosen via cross-validated walk-forward analysis, not hand-tuned. Flat regions
of stability were preferred over sharp optima.

**Engineering**
- Stateless trader with JSON-serialised `traderData` and corruption fallback
- Bounded history buffers to prevent memory growth
- Defensive coding around malformed order books

---

## The Timo Diehm Pivot: First Principles

Mid-competition I read an article by Timo Diehm that genuinely changed how I think
about problems. The frame is:

1. **Verify what is true.** Competition objectives, mechanics, payoff functions —
   write these down explicitly. Anything not on the list is a hypothesis.
2. **Build only on the verified.** Strategies are constructed on top of the
   verified base, not on top of unexamined priors.
3. **Understand the problem deeply, then go further.** Most people stop at "deeply
   enough." The edge is in going further.
4. **Do whatever it takes to win.** Including, especially, throwing away work that
   was based on an assumption that turned out to be wrong.

Mapping this back to my performance: R2 manual was the one place I followed this
workflow end-to-end, and the prediction landed within 9% of reality. R4 was where I
violated step 1 — I did not verify the data-generating process before pricing
options on it — and the result was the worst round of the competition.

The R5 EDA notebook is structured around this discipline. Section 1 is literally
titled *"foundation, what is true and verifiable"* and lists what we know vs. what
we are still hypothesising. Every later section closes with a falsification
criterion: what would we need to observe to invalidate this idea?

---

## What I'd Do Differently

Two things, mostly.

**First: validate the DGP before building any pricing or strategy logic on top of
it.** The R4 failure was not a tuning problem, it was a model-selection problem,
and model selection should always come before tuning. ADF, Hurst, half-life,
variance ratio — these are five-minute checks. Run them first.

**Second: design every parameter choice with a falsification criterion built in.**
Not "this z-score threshold maxes out backtest PnL" but "this z-score threshold
sits in a flat region of stability across folds, and if we observe X regime change
the strategy must reduce size or stop." The Devil's Advocate role from the
multi-agent framework I now use exists precisely to enforce this — its job is to
assume the strategy is wrong and find the fastest path to invalidation.

A few smaller things: tighter PnL attribution between adverse selection vs.
inventory drag vs. spread capture, more aggressive use of cross-validated walk-
forward analysis from day one rather than as a late-round patch, and an earlier
commitment to closed-form fair values where they exist instead of empirical
fitters.

---

## Closing Note

Prosperity is not a coding contest. It is a model-risk contest with a coding
component, and the gap between rank 643 and rank 934 is the gap between a model
whose assumptions held and a model whose assumptions did not. Everything in this
repository — the EDA pipelines, the falsification criteria, the OU call pricer,
the segmented Monte Carlo for R2 manual — is an attempt to build the workflow that
catches that gap *before* it shows up in the PnL.

934 out of 18,803 is a good result. The lesson in how it could have been better is
worth more than the rank.

