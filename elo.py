"""
Elo rating model used as the naive market proxy in backtest.py.

Elo probabilities represent what a simple market participant would price —
beating Elo consistently with Dixon-Coles demonstrates real edge.
"""
import math
import numpy as np
import pandas as pd
from data_loader import IMPORTANCE

# K-factor scaled by match importance (same tiers as DC model)
_K = {"WC_FINAL": 50, "CONFED_CAMP": 35, "QUALIFIER": 25, "FRIENDLY": 10}

HOME_ADV = 50.0   # Elo points added to home team in non-neutral games
DRAW_BASE = 0.265  # draw frequency in international football at equal strength
DRAW_DECAY = 450.0  # Elo diff at which draw probability halves


class EloModel:
    def __init__(self):
        self.ratings: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "EloModel":
        self.ratings = {}
        for _, row in df.sort_values("date").iterrows():
            home, away = row["home_team"], row["away_team"]
            neutral = bool(row.get("neutral", False))
            k = _K.get(row["match_importance_tier"], 15)

            rh = self.ratings.get(home, 1500.0)
            ra = self.ratings.get(away, 1500.0)
            adv = 0.0 if neutral else HOME_ADV

            exp_h = 1.0 / (1.0 + 10.0 ** (-(rh + adv - ra) / 400.0))

            hg, ag = row["home_goals"], row["away_goals"]
            act_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

            self.ratings[home] = rh + k * (act_h - exp_h)
            self.ratings[away] = ra + k * ((1 - act_h) - (1 - exp_h))
        return self

    def predict(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        """Return (P_home_win, P_draw, P_away_win)."""
        rh = self.ratings.get(home, 1500.0)
        ra = self.ratings.get(away, 1500.0)
        adv = 0.0 if neutral else HOME_ADV
        exp_h = 1.0 / (1.0 + 10.0 ** (-(rh + adv - ra) / 400.0))

        elo_diff = abs(rh + adv - ra)
        p_draw = DRAW_BASE * math.exp(-elo_diff / DRAW_DECAY)
        p_draw = max(p_draw, 0.04)

        p_home = exp_h * (1.0 - p_draw)
        p_away = (1.0 - exp_h) * (1.0 - p_draw)
        return p_home, p_draw, p_away
