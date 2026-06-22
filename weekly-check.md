# Weekly Alpha Decay Checklist

Run every Sunday. Takes ~10 minutes. Protects against the model silently dying.

---

## 1. Retrain on new results (always do this)

Append the week's completed matches to the training data, then refit.

**World Cup:**
```bash
# Results are already in real_fixtures.csv from daily updates
# Just re-run the model — it retrains automatically on every morning.py run
python morning.py wc
```

**Club football:**
```bash
# Re-fetch the latest completed matches from football-data.co.uk
python club_main.py --fetch
# This overwrites club_fixtures.csv with all seasons up to today
```

---

## 2. Check calibration temperature T

Run the model and note the T value printed on startup.

```bash
python morning.py wc
# Look for: γ=X.XXXX  ρ=X.XXXX  T=X.XXXX
```

| T value | Meaning | Action |
|---|---|---|
| 0.55 – 0.75 | Normal — model is sharpening probabilities correctly | Nothing |
| 0.75 – 0.90 | Softening — model less confident than it used to be | Monitor |
| > 0.90 | Significant drift — raw probabilities losing accuracy | Investigate |
| < 0.55 | Over-sharpening — model is overconfident | Investigate |

**What causes T to drift:** team quality shifts the model wasn't trained on (new manager, squad change, injuries), or the game style in this tournament/season differs from historical data.

**If T drifts above 0.90:** retrain with a longer calibration window.
```bash
# In main.py, extend the calibration window from 365 to 730 days:
# cal_df = df[df["date"] >= ref - pd.Timedelta(days=730)]
```

---

## 3. Check Brier score

Brier score measures raw prediction accuracy, independent of P&L. This is the most reliable decay signal.

```bash
python backtest.py --fixtures real_fixtures.csv --bankroll 2000
# Note the DC Brier score in the output
```

| Brier score | Meaning | Action |
|---|---|---|
| < 0.550 | Model performing at baseline | Nothing |
| 0.550 – 0.560 | Slight degradation | Monitor weekly |
| > 0.560 | Real decay — approaching Elo baseline | Retrain, check data |
| Rising 3 weeks in a row | Structural decay, not noise | Stop and investigate |

**Baseline to beat:** Elo Brier = 0.555. If DC Brier exceeds this, you have no edge.

---

## 4. Check average Kelly fraction per bet

Open your bet log and calculate the average `Half-Kelly %` across recent bets. If it's shrinking, the model is finding less edge — either because your edge is real and decaying, or the bookmaker is pricing more accurately.

```bash
python -c "
import pandas as pd
# Load your recent bet sheet output (or track manually)
# Check: is the average Half-Kelly % this week lower than last week?
print('Track this manually from morning.py output each day')
"
```

| Trend | Meaning | Action |
|---|---|---|
| Stable (2–8% per bet) | Edge holding | Nothing |
| Slowly shrinking | Bookmaker sharpening, or edge eroding | Monitor |
| Consistently < 1% | Nearly no edge left | Reduce stakes or pause |
| Spikes > 15% | Model may be overconfident on a match | Double-check manually |

---

## 5. Check P&L vs Brier divergence

This is the key test for noise vs real decay.

| P&L | Brier score | Diagnosis |
|---|---|---|
| Positive | Stable | Model working, keep going |
| Negative | Stable | Bad variance, keep going — do NOT stop |
| Positive | Rising | Lucky run masking decay — watch closely |
| Negative | Rising | **Real decay** — retrain or pause |
| Negative | Falling | Unlucky run, model actually improving — definitely keep going |

**Rule:** Never stop the model because of P&L alone. Only stop when Brier score degrades AND stays degraded for 3+ weeks.

---

## 6. Check for data staleness

Make sure the training data reflects recent team form.

```bash
python -c "
import pandas as pd
df = pd.read_csv('real_fixtures.csv')
print('WC data up to:', df['date'].max())
df2 = pd.read_csv('club_fixtures.csv')
print('Club data up to:', df2['date'].max())
"
```

- WC data should be current to within 2 days
- Club data should be within 7 days (run `--fetch` if not)
- If a major team had a managerial change this week, note it — the model won't know for ~5 matches

---

## 7. Check bookmaker sharpness

Are the odds on melbet getting tighter over time? If the gap between ESPN odds and melbet odds is shrinking week-over-week, the bookmaker is getting sharper and your edge is being compressed.

Track this informally: note how much you're adjusting odds downward during the `verify_melbet_odds` step each morning. If you're consistently seeing 0.05–0.10 gaps (e.g. ESPN 1.714 → melbet 1.635), that's normal. If gaps start reaching 0.15+, the bookmaker is pricing more aggressively.

---

## 8. Weekly decision tree

```
Did Brier score rise this week?
├── No  → Continue as normal
└── Yes → Is this the 2nd or 3rd week in a row?
           ├── No  → Monitor, check again next week
           └── Yes → Retrain with extended window (730 days)
                     └── Did Brier recover after retrain?
                          ├── Yes → Back to normal
                          └── No  → Architecture may need updating
                                    Consider: longer half-life, new tiers,
                                    different importance weights
```

---

## 9. When to seriously reconsider the model

Stop and reassess if **two or more** of these are true simultaneously:

- [ ] Brier score > 0.560 for 3 consecutive weeks
- [ ] Average Kelly fraction < 1% (barely finding any edge)
- [ ] P&L negative AND Brier rising (not just variance)
- [ ] T > 0.95 and not recovering after retrain
- [ ] Melbet odds consistently 0.15+ below ESPN (bookmaker sharpening fast)

If this happens: pause betting, do a full retrain with a longer data window, and only resume when Brier score returns below 0.555.

---

## Quick reference — what to record each week

| Date | T value | Brier score | Edge over Elo | Avg Kelly % | P&L this week | Status |
|---|---|---|---|---|---|---|
| 2026-06-22 | 0.610 | **0.5252** | **+2.98%** | ~6% | +2 ETB (net flat) | ✅ Healthy — Brier well below 0.555 baseline |

### Week 1 notes (June 22)
- **T = 0.610** — healthy, within normal range (0.55–0.75). Model is sharpening probabilities correctly.
- **Brier = 0.5252** — significantly better than the 0.549 historical figure and well below the 0.555 Elo baseline. Edge over Elo has widened to **+2.98%** from +1.1% — likely because WC group stage data (high-quality, high-importance matches) improved team ratings for the teams actually in the tournament.
- **Data freshness** — WC data current to June 21 ✅. Club data current to May 24 (off-season, acceptable) ✅.
- **P&L** — flat this week (+2 ETB net). Germany/IC bets lost, Ecuador draw saved the day. No decay signal.
- **Action** — nothing to change. Continue as normal. Retrain with WC results weekly.
