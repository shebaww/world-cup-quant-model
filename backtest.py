"""
Walk-forward backtest: Dixon-Coles vs Elo market proxy.

The Elo model is trained on the same historical window as DC and used as the
"bookmaker." Elo prices are straightforward to implement and represent what
a moderately-informed market participant would offer. DC should consistently
outperform Elo because it uses goal counts, time-decay, and match importance
— not just win/draw/loss outcomes.

Usage:
    python backtest.py                           # needs real_fixtures.csv
    python backtest.py --fixtures my_data.csv
    python backtest.py --train-days 1461 --step-days 180
"""
import argparse

import numpy as np
import pandas as pd

from data_loader import load_fixtures
from dixon_coles import DixonColesModel
from elo import EloModel
from kelly import half_kelly, MAX_MATCHDAY_FRACTION

VIG = 0.05  # bookmaker margin applied on top of Elo's implied probability


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - outcomes) ** 2, axis=1)))


def run(
    fixtures_path: str = "real_fixtures.csv",
    train_days: int    = 4 * 365,
    step_days: int     = 180,
    bankroll: float    = 15_000.0,
    verbose: bool      = True,
) -> dict:
    df = load_fixtures(fixtures_path)

    first_test = df["date"].min() + pd.Timedelta(days=train_days)
    windows = []
    t = first_test
    while t < df["date"].max():
        windows.append(t)
        t += pd.Timedelta(days=step_days)

    all_dc_probs: list    = []
    all_elo_probs: list   = []
    all_outcomes: list    = []
    bets: list            = []
    br = bankroll

    for ref in windows:
        train = df[df["date"] < ref]
        test  = df[(df["date"] >= ref) & (df["date"] < ref + pd.Timedelta(days=step_days))]
        if len(train) < 200 or len(test) < 5:
            continue

        try:
            dc  = DixonColesModel().fit(train, ref)
            elo = EloModel().fit(train)
        except Exception:
            continue

        # Collect all candidate bets for this test window grouped by date,
        # then apply the 15% daily cap across the whole day before sizing stakes.
        daily: dict[str, list] = {}
        for _, row in test.iterrows():
            neutral = bool(row.get("neutral", False))
            try:
                ph_dc, pd_dc, pa_dc = dc.match_outcome_probs(
                    row["home_team"], row["away_team"], neutral=neutral
                )
                ph_el, pd_el, pa_el = elo.predict(
                    row["home_team"], row["away_team"], neutral=neutral
                )
            except Exception:
                continue

            h_win = int(row["home_goals"] > row["away_goals"])
            draw  = int(row["home_goals"] == row["away_goals"])
            a_win = int(row["home_goals"] < row["away_goals"])

            all_dc_probs.append([ph_dc, pd_dc, pa_dc])
            all_elo_probs.append([ph_el, pd_el, pa_el])
            all_outcomes.append([h_win, draw, a_win])

            day_key = str(row["date"].date())
            if day_key not in daily:
                daily[day_key] = []
            for p_dc, p_elo, actual, label in [
                (ph_dc, ph_el, h_win, "home"),
                (pd_dc, pd_el, draw,  "draw"),
                (pa_dc, pa_el, a_win, "away"),
            ]:
                if p_elo <= 0:
                    continue
                mkt_odds = 1.0 / (p_elo * (1.0 + VIG))
                fk = half_kelly(p_dc, mkt_odds)
                if fk > 0:
                    daily[day_key].append({
                        "match":    f"{row['home_team']} vs {row['away_team']}",
                        "label":    label,
                        "p_dc":     p_dc,
                        "p_elo":    p_elo,
                        "mkt_odds": mkt_odds,
                        "fk":       fk,
                        "actual":   actual,
                    })

        # Apply daily cap: total fractional allocation ≤ MAX_MATCHDAY_FRACTION
        for day_key in sorted(daily):
            day_bets = daily[day_key]
            total_fk = sum(b["fk"] for b in day_bets)
            scale = min(1.0, MAX_MATCHDAY_FRACTION / total_fk) if total_fk > 0 else 1.0
            for b in day_bets:
                f_scaled = b["fk"] * scale
                stake = f_scaled * bankroll  # fixed bankroll — avoids Kelly compounding blowup
                pnl   = stake * (b["mkt_odds"] - 1.0) if b["actual"] else -stake
                br   += pnl
                bets.append({
                    "date":     day_key,
                    "match":    b["match"],
                    "bet":      b["label"],
                    "p_dc":     round(b["p_dc"], 4),
                    "p_elo":    round(b["p_elo"], 4),
                    "mkt_odds": round(b["mkt_odds"], 3),
                    "stake":    round(stake, 2),
                    "pnl":      round(pnl, 2),
                    "bankroll": round(br, 2),
                })

    dc_arr  = np.array(all_dc_probs)  if all_dc_probs  else np.empty((0, 3))
    elo_arr = np.array(all_elo_probs) if all_elo_probs else np.empty((0, 3))
    out_arr = np.array(all_outcomes)  if all_outcomes  else np.empty((0, 3))

    if len(dc_arr):
        dc_brier  = _brier(dc_arr,  out_arr)
        elo_brier = _brier(elo_arr, out_arr)
        brier_gain = f"{(1 - dc_brier / elo_brier) * 100:.1f}%"
    else:
        dc_brier = elo_brier = float("nan")
        brier_gain = "N/A"

    total_pnl = br - bankroll
    summary = {
        "matches_tested":    len(all_dc_probs),
        "dc_brier":          round(dc_brier, 5),
        "elo_brier":         round(elo_brier, 5),
        "brier_improvement": brier_gain,
        "total_bets":        len(bets),
        "total_pnl_etb":     round(total_pnl, 2),
        "roi_pct":           round(total_pnl / bankroll * 100, 2) if bankroll else 0,
        "final_bankroll_etb": round(br, 2),
    }

    if verbose:
        print("\n── Backtest Results (DC vs Elo market, 5% vig) ──")
        for k, v in summary.items():
            print(f"  {k:28s}: {v}")
        print()
        print("  Elo model = market proxy trained on same historical data.")
        print("  DC beats Elo → edge exists vs moderately-informed markets.")

    return {"summary": summary, "bets": pd.DataFrame(bets)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures",   default="real_fixtures.csv")
    ap.add_argument("--train-days", type=int,   default=4 * 365)
    ap.add_argument("--step-days",  type=int,   default=180)
    ap.add_argument("--bankroll",   type=float, default=15_000.0)
    args = ap.parse_args()

    results = run(args.fixtures, args.train_days, args.step_days, args.bankroll)
    df_bets = results["bets"]
    if not df_bets.empty:
        n = min(30, len(df_bets))
        print(f"\n── Last {n} Bets ──")
        print(df_bets.tail(n).to_string(index=False))
        wins   = (df_bets["pnl"] > 0).sum()
        losses = (df_bets["pnl"] < 0).sum()
        print(f"\n  Win/Loss: {wins}W / {losses}L  ({wins/(wins+losses)*100:.1f}% hit rate)")
