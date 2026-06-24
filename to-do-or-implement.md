# To-Do / Improvements

Ordered roughly by impact. Top section = things that would meaningfully change results. Bottom = nice-to-have.

---

## 🔴 High priority — affects edge and risk directly

### ~~1. Single-bet stake cap~~ ✅
~~Kelly sizing on high-confidence short-odds bets produces dangerously large stakes.
Uruguay at 1.455 odds with 95.5% model confidence produced a 203 ETB stake — when it drew, one result wiped out two weeks of gains.~~

~~**Fix:** Hard cap any single bet at ~100 ETB (or ~4–5% of bankroll), regardless of Kelly output.
Particularly important for any bet with odds below 1.6.~~

Implemented in `kelly.py`: `SINGLE_BET_CAP_ETB = 100.0` applied via `.clip(upper=...)` after computing allocations.

---

### 2. Closing line value (CLV) tracking
The best proof of long-run edge isn't P&L — it's whether your bets consistently beat the closing line (the odds right before kick-off, after sharp money has moved them).

If your morning odds are consistently better than the closing price, you have real edge.
If they're consistently worse, sharps are beating you to the line and your edge is illusory.

**Implement:** Log the odds at placement time, then scrape closing odds from ESPN after the match and compare. Calculate CLV = (your odds / closing odds) - 1 per bet.

---

### 3. Proper odds source — replace ESPN
ESPN odds lag, differ from melbet, and disappear after matches finish. The current workflow requires manual correction every morning.

**Better sources:**
- [The Odds API](https://the-odds-api.com) — free tier covers major leagues, returns live odds from 40+ bookmakers including Pinnacle (the sharpest market)
- Use Pinnacle as the reference line (sharpest bookmaker = most efficient price)
- Use melbet for actual placement

This removes the ESPN → melbet discrepancy step and gives you a proper sharp line to measure your edge against.

---

### 4. Drawdown protection
No mechanism currently stops betting during a losing streak. A 5-day losing run at 15% bankroll per day destroys the account.

**Implement:** If bankroll drops more than 25% from its peak, automatically halve the Kelly fraction until it recovers.

```python
PEAK_BANKROLL = 2_543.0   # update this as new highs are hit
DRAWDOWN_THRESHOLD = 0.25  # 25% from peak
DRAWDOWN_KELLY_SCALE = 0.5 # halve sizing during drawdown

if (PEAK_BANKROLL - TOTAL_BANKROLL_ETB) / PEAK_BANKROLL > DRAWDOWN_THRESHOLD:
    KELLY_SCALE = 0.25  # half of the normal 0.5
```

---

### 5. Automatic bet result logging
Currently P&L is tracked manually in README and kelly.py. There's no machine-readable record of individual bets for analysis.

**Implement:** Append each bet to `bet_log.csv` when placed, with columns:
`date, match, choice, model_prob, odds, stake, result, pnl, bankroll_after`

This enables proper edge decay analysis, CLV tracking, and calibration checks without doing it manually.

---

## 🟡 Medium priority — improves model accuracy

### 6. Tournament stage weighting
Dixon-Coles treats a group stage match the same as a knockout match. Knockout football is tactically different — teams play more cautiously, draws are eliminated as a strategic option, and underdogs defend deeper.

**Fix:** Add a `KNOCKOUT` importance tier (weight ~3.5) and adjust the τ (low-score correction) parameter for knockout matches. Knockout games have a lower draw rate at 90 minutes than group games.

---

### 7. Injury and squad data
The model knows nothing about team news. A 95.5% Uruguay call doesn't account for whether Nuñez is suspended or whether this is a dead-rubber group game with rotation.

**Basic fix:** Before placing any bet, check a quick news source for the starting lineup. Don't bet if a key player is missing and the model is pricing a >90% probability — those are the bets most vulnerable to lineup shocks.

**Longer term:** Integrate player availability data from a football API (API-Football has this on the free tier).

---

### 8. Expected goals (xG) integration
Dixon-Coles uses actual goals, which are noisy. xG (expected goals) is a better measure of true team quality — a team that wins 3–0 on 0.8 xG got lucky; one that loses 0–1 on 2.5 xG was unlucky.

**Implement:** Add an xG-weighted alternative alongside the goals-based model and compare Brier scores. If xG model beats goals model consistently, switch primary.

Data source: Understat (free, has xG for EPL/La Liga/Bundesliga/Serie A/Ligue 1 going back to 2014).

---

### 9. Separate model per competition stage
Currently one model handles WC group stage, WC knockouts, EPL, La Liga, etc.
These have meaningfully different dynamics:
- Home advantage varies significantly by league
- Draw rates differ (EPL ~25%, WC knockouts ~18%, La Liga ~22%)
- Goal rates differ

**Implement:** Fit a model per league for club football. For WC, fit separate calibration (T) for group stage vs knockout.

---

### 10. Kelly with uncertainty (fractional based on model confidence)
Current Kelly assumes the model probability is correct. But model confidence should itself be penalised when:
- The team has < 10 matches in training data
- The match is a major upset scenario (odds > 15.0)
- The match is a neutral venue the model hasn't seen before

**Fix:** Reduce Kelly fraction by a further 50% on any bet where the model has sparse data for either team.

---

## 🟢 Low priority — workflow and monitoring

### 11. Re-fetch odds immediately before placement
Currently odds are fetched at script start, but you might place bets 20–30 minutes later after going through the verification step. Odds can move materially in that window.

**Fix:** Re-fetch and display current odds one more time immediately before the final bet sheet is printed.

---

### 12. Automated weekly retrain trigger
Currently the weekly retrain (`club_main.py --fetch`) is manual. It's easy to forget.

**Fix:** Add a check at the top of `morning.py` — if `club_fixtures.csv` is more than 7 days old, warn and prompt to retrain before continuing.

---

### 13. P&L chart in README
The running total table is useful but a visual P&L curve would show variance vs trend more clearly. A simple ASCII chart or a generated PNG via matplotlib committed weekly.

---

### 14. Multiple bookmaker comparison
If you can access more than one bookmaker (e.g. melbet + 1xBet + SportyBet), always bet with whoever has the best odds. Even a 0.05 difference in odds on a regular bet compounds meaningfully over a season.

**Implement:** In `morning.py`, during the odds verification step, prompt for odds from 2–3 bookmakers and auto-select the best one.

---

### 15. Model ensemble
Run two versions of the model — one with 2-year half-life (current) and one with 1-year half-life (more recent form) — and average their probabilities. Ensembling reduces variance in the probability estimates, which leads to more stable Kelly fractions.

---

## Done ✅
- [x] Dixon-Coles model with time-decay and importance weighting
- [x] Temperature calibration to fix systematic underconfidence
- [x] Half-Kelly with 15% daily cap
- [x] ESPN odds auto-fetch with melbet verification step
- [x] Weekly alpha decay checklist (weekly-check.md)
- [x] Club football model (EPL, La Liga, Bundesliga, Serie A, Ligue 1)
- [x] WC group stage results retraining (June 12–21)
- [x] Single-bet stake cap: 100 ETB hard cap in kelly.py (June 22)
