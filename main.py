"""
World Cup Prediction Engine — main entry point.

Quickstart:
    # 1. Download real match data (run once):
    python main.py --fetch

    # 2. Fit model and simulate tournament:
    python main.py --fixtures real_fixtures.csv --host Canada

    # 3. Generate Kelly bet sheet from live odds:
    #    Odds CSV columns: home_team, away_team, home_odds, draw_odds, away_odds
    python main.py --fixtures real_fixtures.csv --odds upcoming_odds.csv

    # 4. Validate the model (walk-forward backtest):
    python backtest.py
"""
import argparse
import pandas as pd
from data_loader import load_fixtures
from dixon_coles import DixonColesModel
from simulator import TournamentSimulator
from kelly import build_bet_sheet, TOTAL_BANKROLL_ETB

DEFAULT_GROUPS = {
    "A": ["Canada",    "Morocco",   "Croatia",    "Belgium"],
    "B": ["USA",       "Portugal",  "Brazil",     "South Korea"],
    "C": ["Mexico",    "Argentina", "Poland",     "Saudi Arabia"],
    "D": ["France",    "England",   "Germany",    "Japan"],
    "E": ["Spain",     "Netherlands","Senegal",   "Ecuador"],
    "F": ["Uruguay",   "Switzerland","Cameroon",  "Serbia"],
    "G": ["Australia", "Denmark",   "Tunisia",    "Colombia"],
    "H": ["Ghana",     "Iran",      "Costa Rica", "Qatar"],
}


def _kelly_from_odds(odds_path: str, model: DixonColesModel, bankroll: float) -> pd.DataFrame:
    """Load a CSV of upcoming fixtures with market odds; return Kelly bet sheet."""
    odds_df = pd.read_csv(odds_path)
    required = {"home_team", "away_team", "home_odds", "draw_odds", "away_odds"}
    missing = required - set(odds_df.columns)
    if missing:
        raise ValueError(f"Odds CSV is missing columns: {missing}")

    bets = []
    for _, row in odds_df.iterrows():
        ph, pd_, pa = model.match_outcome_probs(row["home_team"], row["away_team"])
        label = f"{row['home_team']} vs {row['away_team']}"
        for prob, odds, choice in [
            (ph, row["home_odds"], row["home_team"]),
            (pd_, row["draw_odds"], "Draw"),
            (pa, row["away_odds"], row["away_team"]),
        ]:
            bets.append({
                "match":        label,
                "choice":       choice,
                "model_prob":   round(prob, 4),
                "decimal_odds": odds,
            })
    return build_bet_sheet(bets, bankroll=bankroll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures",  default="fixtures.csv")
    parser.add_argument("--ref-date",  default="2026-06-11")
    parser.add_argument("--host",      default="Canada")
    parser.add_argument("--n-sims",    type=int, default=100_000)
    parser.add_argument("--no-sim",    action="store_true",
                        help="Skip tournament simulation")
    parser.add_argument("--fetch",     action="store_true",
                        help="Download real fixtures from GitHub before fitting")
    parser.add_argument("--odds",      default=None, metavar="PATH",
                        help="CSV with home_team,away_team,home_odds,draw_odds,away_odds")
    args = parser.parse_args()

    if args.fetch:
        from data_fetcher import fetch
        args.fixtures = "real_fixtures.csv"
        fetch(save_path=args.fixtures)

    ref_date = pd.Timestamp(args.ref_date)

    print(f"Loading fixtures from '{args.fixtures}' ...")
    df = load_fixtures(args.fixtures)
    print(f"  {len(df):,} matches loaded spanning "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    print("Fitting Dixon-Coles model ...")
    model = DixonColesModel(host_team=args.host).fit(df, ref_date)
    cal_df = df[df["date"] >= ref_date - pd.Timedelta(days=365)]
    model.fit_calibration(cal_df)
    print(f"  Teams: {len(model.teams_)} | γ={model.gamma_:.4f} | ρ={model.rho_:.4f} | T={model.temperature_:.4f}")

    if not args.no_sim:
        print(f"Running {args.n_sims:,} Monte Carlo simulations ...")
        sim = TournamentSimulator(model, DEFAULT_GROUPS)
        results = sim.run(n=args.n_sims)
        print("\n── Tournament Probabilities ──")
        print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.odds:
        sheet = _kelly_from_odds(args.odds, model, TOTAL_BANKROLL_ETB)
        print("\n── Kelly Bet Sheet (ETB) ──")
        print(sheet.to_string(index=False))
    else:
        print("\n  Tip: pass --odds <file.csv> to generate a Kelly bet sheet from live market odds.")
        print("       CSV must have: home_team,away_team,home_odds,draw_odds,away_odds")


if __name__ == "__main__":
    main()
