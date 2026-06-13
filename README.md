# World Cup 2026 Prediction Model

A statistical soccer prediction engine built around the **2026 FIFA World Cup**. Uses the Dixon-Coles Poisson model to forecast match outcomes, calibrates probabilities against historical data, and sizes bets using the Kelly criterion.

Built as a learning project to explore sports modelling and quantitative betting theory.

---

## How It Works

### 1. Dixon-Coles Model (`dixon_coles.py`)
Fits per-team **attack** (α) and **defense** (β) ratings from 15 years of international match data using maximum likelihood estimation. Key features:
- **Time decay** — recent matches weighted more heavily (2-year half-life)
- **Match importance tiers** — World Cup finals weighted 4× more than friendlies
- **Low-score correction** (τ) — fixes Poisson's tendency to underestimate 0-0 and 1-1 results
- **Home advantage** (γ) — applied only to the host nation (Canada)
- **L2 regularisation** — prevents overfitting on teams with few matches
- **Priors** — FIFA-ranked fallbacks for teams with sparse data

### 2. Probability Calibration (`dixon_coles.py → fit_calibration`)
Raw model probabilities are **temperature-scaled** (T ≈ 0.65) to correct a systematic bias discovered during testing: the model underestimates strong favourites by ~6–8 percentage points.

| Probability range | Raw model | After calibration | Actual hit rate |
|---|---|---|---|
| 60–70% | 64.6% | ~70% | 70.4% |
| 70–80% | 74.4% | ~79% | 80.8% |
| 80–90% | 84.4% | ~89% | 89.6% |

### 3. Kelly Criterion (`kelly.py`)
Sizes bets to maximise long-run bankroll growth:
- **Half-Kelly** (50% of full Kelly) to reduce variance
- **2% slippage** buffer on market odds
- **2.5% edge floor** — ignores bets with thin or negative expected value
- **15% daily cap** — total exposure capped at 15% of bankroll per matchday

### 4. Tournament Simulator (`simulator.py`)
Runs 100,000 Monte Carlo simulations of the full WC bracket (group stage → R16 → QF → SF → Final), outputting win probabilities for all 32 teams.

---

## Setup

```bash
pip install -r requirements.txt   # numpy, pandas, scipy
python data_fetcher.py            # download 15k real international results
```

---

## Daily Usage

**Fetch live odds and run bet sheet**
```bash
python -c "
import json, urllib.request, pandas as pd
req = urllib.request.Request(
    'https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard',
    headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as r:
    data = json.loads(r.read())
rows = []
for event in data.get('events', []):
    for comp in event.get('competitions', []):
        comps = comp.get('competitors', [])
        if len(comps) < 2: continue
        home = comps[0].get('team',{}).get('displayName','')
        away = comps[1].get('team',{}).get('displayName','')
        od = (comp.get('odds') or [{}])[0] or {}
        ml = od.get('moneyline') or {}
        def dec(node, key='close'):
            v = ((node.get(key) or {}).get('odds') or (node.get('open') or {}).get('odds'))
            try:
                v = int(str(v).replace('+',''))
                return round(v/100+1,3) if v>0 else round(100/abs(v)+1,3)
            except: return None
        dn = od.get('drawOdds') or {}
        h = dec(ml.get('home') or {}); d = dec(dn,'close') or (lambda v: round(int(str(v).replace('+',''))/100+1,3) if v else None)(dn.get('moneyLine')); a = dec(ml.get('away') or {})
        if h and d and a:
            rows.append({'home_team':home,'away_team':away,'home_odds':h,'draw_odds':d,'away_odds':a})
pd.DataFrame(rows).to_csv('espn_wc2026_odds.csv', index=False)
print(f'{len(rows)} fixtures saved')
"

python main.py --fixtures real_fixtures.csv --ref-date $(date +%Y-%m-%d) --host Canada --no-sim --odds espn_wc2026_odds.csv
```

**Update bankroll after results**
```bash
python -c "from kelly import update_bankroll; update_bankroll(-300)"  # net P&L in ETB
```

**Run full tournament simulation**
```bash
python main.py --fixtures real_fixtures.csv --host Canada
```

---

## Backtest Results

Walk-forward evaluation across **11,849 matches** from 2014–2026. Elo is used as the market proxy.

| Metric | Value |
|---|---|
| Matches tested | 11,849 |
| Total bets placed | 14,713 |
| DC Brier score | 0.549 |
| Elo (market proxy) Brier | 0.555 |
| Edge over market | **+1.1%** |
| Starting bankroll | 2,000 ETB |
| Final bankroll | 108,841 ETB |
| Win rate | 32.9% |

> Win rate is low because the model primarily bets draws and underdogs at high odds — it doesn't need to win often, it needs the wins to pay enough. Backtest is against Elo prices, not a real sharp bookmaker, so real-world performance will be lower.

---

## Live Results — WC 2026

Starting bankroll: **2,000 ETB**

### Matchday 1 — June 12, 2026

| Match | Bet | Odds | Stake | Result | P&L |
|---|---|---|---|---|---|
| Canada vs Bosnia and Herzegovina | Canada | 1.87 | 196 ETB | **LOST** (0–1) | −196 ETB |
| USA vs Paraguay | Paraguay | 3.85 | 104 ETB | **LOST** (0–0 draw) | −104 ETB |

**Day 1 P&L: −300 ETB**
**Bankroll: 1,700 ETB**

The model was highly confident on Canada (85% win probability as host nation) and saw Paraguay as undervalued at 3.85 odds. Both calls were wrong. Canada lost 0–1 to Bosnia, and Paraguay drew with USA.

This is expected variance — statistically, even an 85% favourite loses 1 in 6 times. Two wrong calls on Day 1 says nothing about the model's long-term edge.

---

## Why the Model Bets Underdogs and Draws

The model only bets when **Expected Value > 0**:

```
EV = (model_probability × payout) − stake
```

A heavy favourite at short odds (e.g. Switzerland at 1.211) requires winning **83%+** just to break even. If the model says 77%, the EV is negative — you lose money long-term even though Switzerland probably wins.

Draws and underdogs are where bookmakers consistently misprice:
- **Draws**: bettors dislike them, so bookmakers price them slightly worse than true probability
- **Underdogs**: public bias toward recognisable teams inflates their implied probability, leaving value on the other side

The core rule: **it's not about who wins, it's about whether the odds pay you enough for the risk.**

---

## Limitations

- **50 games is too few**: The model's 1.1% edge over the market only becomes statistically meaningful after 500+ bets. A single tournament is dominated by variance.
- **Soft market proxy**: The backtest beats Elo, not a real bookmaker like Pinnacle. Real-world edge is likely smaller.
- **Training data skew**: Model is trained on all international matches — WC results may differ from qualifiers and friendlies.
- **Not financial advice**: This is a demo project. Bet responsibly.

---

## Project Structure

```
├── main.py              # Entry point — fit model, simulate, generate bet sheet
├── dixon_coles.py       # Dixon-Coles Poisson model + temperature calibration
├── simulator.py         # Monte Carlo tournament simulator (100k runs)
├── elo.py               # Elo model (market proxy for backtesting)
├── kelly.py             # Kelly criterion bet sizing
├── backtest.py          # Walk-forward backtest (DC vs Elo)
├── calibrate.py         # Sweep time-decay parameter PHI
├── data_loader.py       # CSV validation and loading
├── data_fetcher.py      # Download real match data from GitHub
├── odds_scraper.py      # Scrape historical odds from ESPN
├── generate_fixtures.py # Generate synthetic test fixtures
├── tests.py             # Unit tests
└── requirements.txt
```

---

## CLI Reference

```bash
# Download real match data
python main.py --fetch

# Fit model + simulate tournament
python main.py --fixtures real_fixtures.csv --host Canada --n-sims 100000

# Bet sheet from odds file (no simulation)
python main.py --fixtures real_fixtures.csv --no-sim --odds my_odds.csv

# Calibrate time-decay half-life
python calibrate.py --fixtures real_fixtures.csv

# Walk-forward backtest
python backtest.py --fixtures real_fixtures.csv --bankroll 2000
```
