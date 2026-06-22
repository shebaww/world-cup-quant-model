"""Fractional Kelly position sizing layer in ETB."""
import pandas as pd

TOTAL_BANKROLL_ETB: float = 2_197.27
MAX_MATCHDAY_FRACTION = 0.15


SLIPPAGE      = 0.02   # market slippage buffer subtracted from decimal odds
EDGE_FLOOR    = 0.025  # minimum edge to qualify (2.5%)
KELLY_SCALE   = 0.5    # fractional scaling applied to f*


def half_kelly(p: float, b_raw: float) -> float:
    """
    Live risk arbitrage pipeline per spec.
    b_raw = raw decimal odds (before slippage).
    Returns Kelly fraction (0.0 if edge < EDGE_FLOOR).
    """
    b_adj = b_raw - SLIPPAGE          # Odds_Adjusted = Odds - 0.02
    if b_adj <= 1:                    # adjusted net odds must be positive
        return 0.0
    edge = (p * b_adj) - 1            # Edge = (Prob * Odds_Adjusted) - 1
    if edge < EDGE_FLOOR:
        return 0.0
    f_star = edge / (b_adj - 1)       # f* = Edge / (Odds_Adjusted - 1)
    return f_star * KELLY_SCALE       # Apply fractional scaling (0.5)


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

    # Scale down if total exceeds 15% cap
    if total_f > cap:
        df["_raw_f"] *= cap / total_f

    df["Allocation (Br)"] = (df["_raw_f"] * br).round(2)
    df["Half-Kelly %"] = (df["_raw_f"] * 100).round(4)
    return df.drop(columns=["_raw_f"])


def update_bankroll(result_pnl_etb: float):
    """Call after each matchday settlement with net P&L."""
    global TOTAL_BANKROLL_ETB
    TOTAL_BANKROLL_ETB += result_pnl_etb
    print(f"Bankroll updated → {TOTAL_BANKROLL_ETB:,.2f} Br")
    return TOTAL_BANKROLL_ETB
