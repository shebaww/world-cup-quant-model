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

> 🟢 **LIVE** — Real money bets placed on [melbet-et.com](https://melbet-et.com) using automated execution via `morning.py` + `bet_executor.py`.

**Starting bankroll: 2,000 ETB** | **Current bankroll: 2,513 ETB** | **Overall: +513 ETB (+25.7%)**

> **June 12–19 results are real tracked bets.** From June 20 onwards, bets are placed live on melbet-et.com via the automated workflow. Bankroll reflects actual account balance.

> **Note on sample size:** ~31 settled bets is still noise. The model's 1.1% edge only becomes statistically detectable after 200–500 bets. The EPL 2026–27 season (starting August) is where real validation happens — 380 matches per season.

### Matchday 1 — June 12, 2026

| Match | Bet | Odds | Stake | Result | P&L |
|---|---|---|---|---|---|
| Canada vs Bosnia and Herzegovina | Canada | 1.87 | 196 ETB | **LOST** (1–1 draw) | −196 ETB |
| USA vs Paraguay | Paraguay | 3.85 | 104 ETB | **LOST** (USA 4–1) | −104 ETB |

**Day P&L: −300 ETB** | **Bankroll: 1,700 ETB**

### Matchday 2 — June 13, 2026

| Match | Bet | Odds | Stake | Result | P&L |
|---|---|---|---|---|---|
| Qatar vs Switzerland | Qatar | 14.0 | 25 ETB | **LOST** (1–1 draw) | −25 ETB |
| Brazil vs Morocco | Draw | 3.7 | 85 ETB | **WON** (1–1) | +230 ETB |
| Haiti vs Scotland | Haiti | 6.0 | 48 ETB | **LOST** (0–1) | −48 ETB |

**Day P&L: +157 ETB** | **Bankroll: 1,857 ETB**

### Matchday 3 — June 14, 2026

| Match | Bet | Odds | Stake | Result | P&L |
|---|---|---|---|---|---|
| Australia vs Türkiye | Australia | 5.0 | 123 ETB | **WON** (2–0) | +492 ETB |
| Ivory Coast vs Ecuador | Draw | 2.85 | 23 ETB | **LOST** (1–0) | −23 ETB |
| Ivory Coast vs Ecuador | Ecuador | 2.45 | 8 ETB | **LOST** (1–0) | −8 ETB |
| Netherlands vs Japan | Japan | 3.65 | 49 ETB | **LOST** (2–2 draw) | −49 ETB |
| Sweden vs Tunisia | Tunisia | 4.30 | 50 ETB | **LOST** (5–1) | −50 ETB |
| Germany vs Curaçao | Draw | 16.0 | 18 ETB | **LOST** (7–1) | −18 ETB |
| Germany vs Curaçao | Curaçao | 31.0 | 8 ETB | **LOST** (7–1) | −8 ETB |

**Day P&L: +336 ETB** | **Bankroll: 2,193 ETB**

### Matchday 4 — June 15, 2026

| Match | Bet | Odds | Stake | Result | P&L |
|---|---|---|---|---|---|
| Spain vs Cape Verde | — | — | — | 0–0 draw (no bet) | — |
| Belgium vs Egypt | Draw | 3.95 | 74 ETB | **WON** (1–1) | +218 ETB |
| Belgium vs Egypt | Egypt | 5.75 | 104 ETB | **LOST** (1–1) | −104 ETB |
| Saudi Arabia vs Uruguay | Draw | 4.50 | 75 ETB | **WON** (1–1) | +263 ETB |
| Saudi Arabia vs Uruguay | Saudi Arabia | 8.00 | 30 ETB | **LOST** (1–1) | −30 ETB |
| Iran vs New Zealand | Draw | 3.50 | 16 ETB | **WON** (2–2) | +40 ETB |
| Iran vs New Zealand | New Zealand | 4.70 | 37 ETB | **LOST** (2–2) | −37 ETB |

**Day P&L: +350 ETB** | **Bankroll: 2,543 ETB**

### Matchday 5 — June 16, 2026 *(model not run — no bets)*

| Match | Result |
|---|---|
| France vs Senegal | France 3–1 |
| Norway vs Iraq | Norway 4–1 |
| Argentina vs Algeria | Argentina 3–0 |

### Matchday 6 — June 17, 2026 *(model not run — no bets)*

| Match | Result |
|---|---|
| Austria vs Jordan | Austria 3–1 |
| Portugal vs Congo DR | 1–1 draw |
| England vs Croatia | England 4–2 |
| Ghana vs Panama | Ghana 1–0 |
| Colombia vs Uzbekistan | Colombia 3–1 |

### Matchday 7 — June 18, 2026

| Match | Bet | Odds | Stake | Result | P&L |
|---|---|---|---|---|---|
| Czechia vs South Africa | South Africa | 4.80 | 130 ETB | **LOST** (1–1 draw) | −130 ETB |
| Switzerland vs Bosnia-Herzegovina | Switzerland | 1.556 | 142 ETB | **WON** (4–1) | +79 ETB |
| Canada vs Qatar | Canada | 1.294 | 74 ETB | **WON** (6–0) | +22 ETB |

**Day P&L: −29 ETB** | **Bankroll: 2,513 ETB**

### Matchday 8 — June 19, 2026 *(model not run — no bets)*

| Match | Result |
|---|---|
| USA vs Australia | USA 2–0 |
| Morocco vs Scotland | Morocco 1–0 |
| Brazil vs Haiti | Brazil 3–0 |
| Türkiye vs Paraguay | Paraguay 1–0 |

### Matchday 9 — June 20, 2026

| Match | Bet | Odds | Stake | Result | P&L |
|---|---|---|---|---|---|
| Netherlands vs Sweden | Netherlands | 1.714 | — | **MISSED** (5–1) | — |
| Germany vs Ivory Coast | Draw | 4.60 | 119 ETB | **LOST** (2–1) | −119 ETB |
| Germany vs Ivory Coast | Ivory Coast | 6.50 | 59 ETB | **LOST** (2–1) | −59 ETB |
| Ecuador vs Curaçao | Draw | 8.50 | 24 ETB | **WON** (0–0) | +180 ETB |

**Day P&L: +2 ETB** | **Bankroll: 2,515 ETB**

### Matchday 10 — June 21, 2026 *(model not run — no bets)*

| Match | Result |
|---|---|
| Tunisia vs Japan | Japan 4–0 |
| Spain vs Saudi Arabia | Spain 4–0 |
| Belgium vs Iran | 0–0 draw |
| Uruguay vs Cape Verde | 2–2 draw |
| New Zealand vs Egypt | Egypt 3–1 |

### Running Total

| Day | P&L | Bankroll | Note |
|---|---|---|---|
| Start | — | 2,000 ETB | — |
| June 12 | −300 ETB | 1,700 ETB | Canada draw & Paraguay wrong |
| June 13 | +157 ETB | 1,857 ETB | Brazil/Morocco draw hit |
| June 14 | +336 ETB | 2,193 ETB | Australia 5.0 win; 5 losers |
| June 15 | +350 ETB | 2,543 ETB | Three draws hit (Belgium, Saudi/Uruguay, Iran/NZ) |
| June 16 | — | 2,543 ETB | Model not run |
| June 17 | — | 2,543 ETB | Model not run |
| June 18 | −29 ETB | 2,513 ETB | South Africa drew; Switzerland & Canada won |
| June 19 | — | 2,513 ETB | Model not run |
| June 20 | +2 ETB | 2,515 ETB | Ecuador 0–0 draw saved the day; Netherlands missed |
| June 21 | — | 2,515 ETB | Model not run |

**Overall (settled): +515 ETB (+25.8%) across ~33 settled bets**

⚠️ Ecuador 0–0 Curaçao at 8.50 odds (+180 ETB) rescued a day that was otherwise −178 ETB. One high-odds draw covering two straight losses is a textbook variance day — not signal.

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

## Club Football Model

The same Dixon-Coles engine extended to Premier League, La Liga, Bundesliga, Serie A, and Ligue 1 — where 500+ bets per season make edge statistically measurable.

Key difference from the WC model: home advantage (γ) applies to **all** home teams, not just the tournament host.

```bash
# Download 11 seasons of top-5 league data (~19,000 matches, run once):
python club_main.py --fetch

# Generate today's bet sheet for any league:
python club_main.py --league epl
python club_main.py --league laliga
python club_main.py --league bundesliga
python club_main.py --league seriea
python club_main.py --league ligue1

# Provide your own odds CSV:
python club_main.py --league epl --odds my_odds.csv
```

---

## Project Structure

```
├── main.py              # WC entry point — fit model, simulate, generate bet sheet
├── club_main.py         # Club football entry point (EPL, La Liga, Bundesliga, etc.)
├── dixon_coles.py       # Dixon-Coles Poisson model + temperature calibration
├── simulator.py         # Monte Carlo tournament simulator (100k runs)
├── elo.py               # Elo model (market proxy for backtesting)
├── kelly.py             # Kelly criterion bet sizing
├── backtest.py          # Walk-forward backtest (DC vs Elo)
├── calibrate.py         # Sweep time-decay parameter PHI
├── data_loader.py       # CSV validation and loading
├── data_fetcher.py      # Download real international match data from GitHub
├── club_data_fetcher.py # Download club football data from football-data.co.uk
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
