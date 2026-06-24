"""Fractional Kelly position sizing layer in ETB."""
import pandas as pd

TOTAL_BANKROLL_ETB: float = 2_133.99
MAX_MATCHDAY_FRACTION  = 0.15   # total daily exposure cap
MAX_SINGLE_BET_FRACTION = 0.10  # no single bet exceeds 10% of bankroll


SLIPPAGE      = 0.02   # market slippage buffer subtracted from decimal odds
EDGE_FLOOR    = 0.025  # minimum edge to qualify (2.5%)
KELLY_SCALE   = 0.5    # base fractional scaling applied to f*

# Short-odds tiers: (upper_bound, multiplier applied on top of KELLY_SCALE).
# Heavy favourites are penalised because a small model miscalibration at short
# odds is disproportionately costly — losing 1-in-20 at 1.45 wipes many wins.
SHORT_ODDS_TIERS = [
    (1.50, 0.25),   # odds < 1.50  → 0.5 × 0.25 = ⅛ Kelly
    (1.75, 0.50),   # odds 1.50–1.75 → 0.5 × 0.50 = ¼ Kelly
]
# odds ≥ 1.75: multiplier = 1.0 → normal ½ Kelly


def _odds_multiplier(b_raw: float) -> float:
    for threshold, scale in SHORT_ODDS_TIERS:
        if b_raw < threshold:
            return scale
    return 1.0


def half_kelly(p: float, b_raw: float) -> float:
    """
    b_raw = raw decimal odds (before slippage).
    Returns Kelly fraction (0.0 if edge < EDGE_FLOOR).
    Applies SHORT_ODDS_TIERS multiplier to de-risk heavy-favourite bets
    while preserving proportional Kelly sizing across the full bet sheet.
    """
    b_adj = b_raw - SLIPPAGE
    if b_adj <= 1:
        return 0.0
    edge = (p * b_adj) - 1
    if edge < EDGE_FLOOR:
        return 0.0
    f_star = edge / (b_adj - 1)
    return f_star * KELLY_SCALE * _odds_multiplier(b_raw)


def build_bet_sheet(
    opportunities: list[dict],
    bankroll: float = None,
) -> pd.DataFrame:
    """
    opportunities: list of dicts with keys:
        match, choice, model_prob, decimal_odds
    Returns actionable transaction DataFrame.
    """
    global TOTAL_BANKROLL_ETB
    br = bankroll if bankroll is not None else TOTAL_BANKROLL_ETB

    rows = []
    for opp in opportunities:
        p   = opp["model_prob"]
        fk  = half_kelly(p, opp["decimal_odds"])   # pass raw decimal odds
        rows.append({
            "Match":            opp["match"],
            "Recommended Choice": opp["choice"],
            "Model Probability":  round(p, 4),
            "Market Odds":        opp["decimal_odds"],
            "Half-Kelly %":       round(fk * 100, 4),
            "_raw_f":             fk,
        })

    df = pd.DataFrame(rows)
    total_f = df["_raw_f"].sum()
    cap = MAX_MATCHDAY_FRACTION

    # Scale down if total exceeds 15% daily cap
    if total_f > cap:
        df["_raw_f"] *= cap / total_f

    # Per-bet fraction cap — no single bet exceeds 10% of bankroll
    df["_raw_f"] = df["_raw_f"].clip(upper=MAX_SINGLE_BET_FRACTION)

    df["Allocation (Br)"] = (df["_raw_f"] * br).round(2)
    df["Half-Kelly %"] = (df["_raw_f"] * 100).round(4)
    return df.drop(columns=["_raw_f"])


def update_bankroll(result_pnl_etb: float):
    """Call after each matchday settlement with net P&L."""
    global TOTAL_BANKROLL_ETB
    TOTAL_BANKROLL_ETB += result_pnl_etb
    print(f"Bankroll updated → {TOTAL_BANKROLL_ETB:,.2f} Br")
    return TOTAL_BANKROLL_ETB
