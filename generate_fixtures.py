"""Generate a synthetic fixtures.csv for testing."""
import pandas as pd
import numpy as np
import random, datetime

random.seed(0); np.random.seed(0)

TEAMS = [
    "Canada","Morocco","Croatia","Belgium","USA","Portugal","Brazil","South Korea",
    "Mexico","Argentina","Poland","Saudi Arabia","France","England","Germany","Japan",
    "Spain","Netherlands","Senegal","Ecuador","Uruguay","Switzerland","Cameroon","Serbia",
    "Australia","Denmark","Tunisia","Colombia","Ghana","Iran","Costa Rica","Qatar",
]
TIERS = ["WC_FINAL","CONFED_CAMP","QUALIFIER","FRIENDLY"]
TIER_W = [0.05, 0.15, 0.50, 0.30]

rows = []
start = datetime.date(2018, 1, 1)
end   = datetime.date(2026, 5, 31)
delta = (end - start).days

for _ in range(3000):
    t = random.randint(0, delta)
    date = start + datetime.timedelta(days=t)
    home, away = random.sample(TEAMS, 2)
    tier = random.choices(TIERS, weights=TIER_W)[0]
    lam = np.random.exponential(1.4)
    mu  = np.random.exponential(1.1)
    hg  = int(np.random.poisson(lam))
    ag  = int(np.random.poisson(mu))
    rows.append({"date": date.isoformat(), "home_team": home, "away_team": away,
                 "home_goals": hg, "away_goals": ag, "match_importance_tier": tier})

pd.DataFrame(rows).to_csv("fixtures.csv", index=False)
print("fixtures.csv written with", len(rows), "rows")
