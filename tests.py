"""Property-based tests for probability distributions."""
import numpy as np
import pytest
from unittest.mock import MagicMock


def make_mock_model(p_home=0.45, p_draw=0.25, p_away=0.30):
    m = MagicMock()
    m.match_outcome_probs.return_value = (p_home, p_draw, p_away)
    m.predict_lambda_mu.return_value = (1.4, 1.1)
    m.teams_ = []
    return m


# ── Outcome probability sum ────────────────────────────────────────────────
def test_outcome_probs_sum_to_one():
    """P(home win) + P(draw) + P(away win) must equal 1.0000 for any fixture."""
    from dixon_coles import DixonColesModel
    import pandas as pd, math

    # Minimal synthetic dataset
    rows = [
        {"date": "2023-01-01", "home_team": "A", "away_team": "B",
         "home_goals": 2, "away_goals": 1, "match_importance_tier": "QUALIFIER"},
        {"date": "2023-06-01", "home_team": "B", "away_team": "A",
         "home_goals": 0, "away_goals": 0, "match_importance_tier": "FRIENDLY"},
        {"date": "2024-01-01", "home_team": "A", "away_team": "B",
         "home_goals": 1, "away_goals": 2, "match_importance_tier": "QUALIFIER"},
        {"date": "2024-06-01", "home_team": "B", "away_team": "A",
         "home_goals": 3, "away_goals": 1, "match_importance_tier": "CONFED_CAMP"},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    model = DixonColesModel().fit(df, ref_date=pd.Timestamp("2026-06-12"))

    for home, away in [("A", "B"), ("B", "A")]:
        ph, pd_, pa = model.match_outcome_probs(home, away)
        total = ph + pd_ + pa
        assert abs(total - 1.0) < 1e-9, f"Probabilities sum to {total} not 1.0"


def test_half_kelly_negative_edge_returns_zero():
    from kelly import half_kelly
    # p=0.30, raw decimal odds=2.0 -> adj=1.98 -> edge=0.30*1.98-1=-0.406 -> 0
    assert half_kelly(0.30, 2.0) == 0.0


def test_half_kelly_tiny_edge_returns_zero():
    from kelly import half_kelly
    # p=0.501, raw decimal odds=2.0 -> adj=1.98 -> edge=0.501*1.98-1=-0.0802 < 2.5% floor -> 0
    assert half_kelly(0.501, 2.0) == 0.0


def test_half_kelly_valid_edge():
    from kelly import half_kelly
    # p=0.60, raw decimal odds=2.0 -> adj=1.98
    # edge = 0.60*1.98-1 = 0.188
    # f* = 0.188/(1.98-1) * 0.5 = 0.188/0.98 * 0.5
    expected = (0.60 * 1.98 - 1) / (1.98 - 1) * 0.5
    assert abs(half_kelly(0.60, 2.0) - expected) < 1e-9


def test_matchday_cap_scales_bets():
    from kelly import build_bet_sheet
    # Two bets each with 10% Kelly -> total 20% > 15% cap -> should be scaled to 15%
    opps = [
        {"match": "A vs B", "choice": "A", "model_prob": 0.60, "decimal_odds": 2.0},
        {"match": "C vs D", "choice": "C", "model_prob": 0.60, "decimal_odds": 2.0},
    ]
    sheet = build_bet_sheet(opps, bankroll=100_000)
    total_alloc = sheet["Allocation (Br)"].sum()
    assert total_alloc <= 15_000 + 0.01, f"Cap breached: {total_alloc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
