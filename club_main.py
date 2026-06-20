"""
Club football prediction engine — Premier League, La Liga, Bundesliga, etc.

Quickstart:
    # 1. Download historical data (run once, takes ~30s):
    python club_main.py --fetch

    # 2. Generate Kelly bet sheet for today's EPL matches:
    python club_main.py --league epl

    # 3. Other leagues:
    python club_main.py --league laliga
    python club_main.py --league bundesliga
    python club_main.py --league seriea
    python club_main.py --league ligue1

    # 4. Provide your own odds CSV (columns: home_team,away_team,home_odds,draw_odds,away_odds):
    python club_main.py --league epl --odds my_odds.csv
"""
import argparse
import json
import urllib.request
import pandas as pd
from data_loader import load_fixtures
from dixon_coles import DixonColesModel
from kelly import build_bet_sheet, TOTAL_BANKROLL_ETB

# Maps ESPN displayName → football-data.co.uk team name
ESPN_TO_FDC: dict[str, str] = {
    # Premier League
    "Manchester City":            "Man City",
    "Manchester United":          "Man United",
    "Tottenham Hotspur":          "Tottenham",
    "Wolverhampton Wanderers":    "Wolves",
    "Newcastle United":           "Newcastle",
    "West Ham United":            "West Ham",
    "Leicester City":             "Leicester",
    "Leeds United":               "Leeds",
    "Brighton & Hove Albion":     "Brighton",
    "Nottingham Forest":          "Nott'm Forest",
    "Sheffield United":           "Sheffield United",
    "Luton Town":                 "Luton",
    "Brentford":                  "Brentford",
    "AFC Bournemouth":            "Bournemouth",
    "Ipswich Town":               "Ipswich",
    "Sunderland AFC":             "Sunderland",
    # La Liga
    "Atlético de Madrid":         "Ath Madrid",
    "Athletic Club":              "Ath Bilbao",
    "Real Betis":                 "Betis",
    "Real Sociedad":              "Sociedad",
    "Deportivo Alavés":           "Alaves",
    "Rayo Vallecano":             "Vallecano",
    "Celta Vigo":                 "Celta",
    "Getafe CF":                  "Getafe",
    "RCD Mallorca":               "Mallorca",
    "Girona FC":                  "Girona",
    "UD Las Palmas":              "Las Palmas",
    "CD Leganés":                 "Leganes",
    "RCD Espanyol":               "Espanol",
    "Real Valladolid":            "Valladolid",
    "CA Osasuna":                 "Osasuna",
    # Bundesliga
    "Bayer Leverkusen":           "Leverkusen",
    "Borussia Dortmund":          "Dortmund",
    "Borussia Mönchengladbach":   "Ein Frankfurt",  # placeholder
    "Eintracht Frankfurt":        "Ein Frankfurt",
    "VfB Stuttgart":              "Stuttgart",
    "SC Freiburg":                "Freiburg",
    "RB Leipzig":                 "RB Leipzig",
    "FC Augsburg":                "Augsburg",
    "VfL Wolfsburg":              "Wolfsburg",
    "TSG Hoffenheim":             "Hoffenheim",
    "Werder Bremen":              "Werder Bremen",
    "FC Union Berlin":            "Union Berlin",
    "VfL Bochum":                 "Bochum",
    "FC Heidenheim 1846":         "Heidenheim",
    "1. FC Köln":                 "FC Koln",
    "Darmstadt 98":               "Darmstadt",
    "Mainz 05":                   "Mainz",
    "FC St. Pauli":               "St Pauli",
    "Holstein Kiel":              "Holstein Kiel",
    # Serie A
    "Inter Milan":                "Inter",
    "AC Milan":                   "Milan",
    "Juventus":                   "Juventus",
    "AS Roma":                    "Roma",
    "SS Lazio":                   "Lazio",
    "ACF Fiorentina":             "Fiorentina",
    "SSC Napoli":                 "Napoli",
    "Atalanta BC":                "Atalanta",
    "Torino FC":                  "Torino",
    "Bologna FC":                 "Bologna",
    "Udinese Calcio":             "Udinese",
    "Genoa CFC":                  "Genoa",
    "Cagliari Calcio":            "Cagliari",
    "Hellas Verona":              "Verona",
    "Empoli FC":                  "Empoli",
    "Frosinone Calcio":           "Frosinone",
    "US Sassuolo":                "Sassuolo",
    "Venezia FC":                 "Venezia",
    "Como 1907":                  "Como",
    "Parma Calcio":               "Parma",
    "Lecce":                      "Lecce",
    "Monza":                      "Monza",
    # Ligue 1
    "Paris Saint-Germain":        "Paris SG",
    "Olympique de Marseille":     "Marseille",
    "Olympique Lyonnais":         "Lyon",
    "AS Monaco":                  "Monaco",
    "OGC Nice":                   "Nice",
    "RC Lens":                    "Lens",
    "Stade Rennais FC":           "Rennes",
    "Stade Brestois 29":          "Brest",
    "Montpellier HSC":            "Montpellier",
    "RC Strasbourg Alsace":       "Strasbourg",
    "Nantes":                     "Nantes",
    "Toulouse FC":                "Toulouse",
    "Le Havre AC":                "Le Havre",
    "FC Metz":                    "Metz",
    "Clermont Foot":              "Clermont",
    "Stade de Reims":             "Reims",
    "Angers SCO":                 "Angers",
    "Saint-Étienne":              "St Etienne",
    "Auxerre":                    "Auxerre",
    "Lille OSC":                  "Lille",
}

ESPN_LEAGUE_CODES: dict[str, str] = {
    "epl":        "eng.1",
    "championship": "eng.2",
    "laliga":     "esp.1",
    "bundesliga": "ger.1",
    "seriea":     "ita.1",
    "ligue1":     "fra.1",
    "cl":         "uefa.champions",
    "el":         "uefa.europa",
}


def _espn_name(name: str) -> str:
    return ESPN_TO_FDC.get(name, name)


def fetch_odds(league: str) -> pd.DataFrame:
    """Fetch today's fixtures and odds from ESPN for the given league."""
    espn_code = ESPN_LEAGUE_CODES.get(league)
    if not espn_code:
        raise ValueError(f"Unknown league '{league}'. Options: {list(ESPN_LEAGUE_CODES)}")

    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())

    rows = []
    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_SCHEDULED":
            continue
        for comp in event.get("competitions", []):
            comps = comp.get("competitors", [])
            if len(comps) < 2:
                continue
            home = _espn_name(comps[0].get("team", {}).get("displayName", ""))
            away = _espn_name(comps[1].get("team", {}).get("displayName", ""))
            od = (comp.get("odds") or [{}])[0] or {}
            ml = od.get("moneyline") or {}

            def dec(node, key="close"):
                v = (node.get(key) or {}).get("odds") or (node.get("open") or {}).get("odds")
                try:
                    v = int(str(v).replace("+", ""))
                    return round(v / 100 + 1, 3) if v > 0 else round(100 / abs(v) + 1, 3)
                except Exception:
                    return None

            dn = od.get("drawOdds") or {}
            h = dec(ml.get("home") or {})
            d_v = dn.get("close", {}).get("odds") or dn.get("open", {}).get("odds") or dn.get("moneyLine")
            try:
                d_v = int(str(d_v).replace("+", ""))
                d = round(d_v / 100 + 1, 3) if d_v > 0 else round(100 / abs(d_v) + 1, 3)
            except Exception:
                d = None
            a = dec(ml.get("away") or {})

            if h and d and a:
                rows.append({"home_team": home, "away_team": away,
                             "home_odds": h, "draw_odds": d, "away_odds": a})

    return pd.DataFrame(rows)


def kelly_from_odds(odds_df: pd.DataFrame, model: DixonColesModel, bankroll: float) -> pd.DataFrame:
    bets = []
    for _, row in odds_df.iterrows():
        ph, pd_, pa = model.match_outcome_probs(row["home_team"], row["away_team"])
        label = f"{row['home_team']} vs {row['away_team']}"
        for prob, odds, choice in [
            (ph, row["home_odds"], row["home_team"]),
            (pd_, row["draw_odds"], "Draw"),
            (pa, row["away_odds"], row["away_team"]),
        ]:
            bets.append({"match": label, "choice": choice,
                         "model_prob": round(prob, 4), "decimal_odds": odds})
    return build_bet_sheet(bets, bankroll=bankroll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch",     action="store_true",
                        help="Download historical club data from football-data.co.uk")
    parser.add_argument("--fixtures",  default="club_fixtures.csv")
    parser.add_argument("--league",    default="epl",
                        choices=list(ESPN_LEAGUE_CODES),
                        help="League to fetch odds for")
    parser.add_argument("--leagues",   nargs="+",
                        default=["epl","laliga","bundesliga","seriea","ligue1"],
                        help="Leagues to include in training data (with --fetch)")
    parser.add_argument("--ref-date",  default=None,
                        help="Reference date for time-decay (default: today)")
    parser.add_argument("--odds",      default=None, metavar="PATH",
                        help="CSV with home_team,away_team,home_odds,draw_odds,away_odds")
    args = parser.parse_args()

    if args.fetch:
        from club_data_fetcher import fetch
        fetch(leagues=args.leagues, save_path=args.fixtures)

    ref_date = pd.Timestamp(args.ref_date) if args.ref_date else pd.Timestamp.today().normalize()

    print(f"\nLoading fixtures from '{args.fixtures}' ...")
    df = load_fixtures(args.fixtures)
    print(f"  {len(df):,} matches loaded spanning "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    print("Fitting Dixon-Coles model (host_team=None → universal home advantage) ...")
    model = DixonColesModel(host_team=None).fit(df, ref_date)
    cal_df = df[df["date"] >= ref_date - pd.Timedelta(days=365)]
    model.fit_calibration(cal_df)
    print(f"  Teams: {len(model.teams_)} | γ={model.gamma_:.4f} | ρ={model.rho_:.4f} | T={model.temperature_:.4f}")

    if args.odds:
        odds_df = pd.read_csv(args.odds)
    else:
        print(f"\nFetching live odds for {args.league.upper()} from ESPN ...")
        odds_df = fetch_odds(args.league)
        if odds_df.empty:
            print("  No scheduled fixtures with odds found for today.")
            print("  Pass --odds <file.csv> to provide odds manually.")
            return

    sheet = kelly_from_odds(odds_df, model, TOTAL_BANKROLL_ETB)
    actionable = sheet[sheet["Allocation (Br)"] > 0]

    print(f"\n── Kelly Bet Sheet — {args.league.upper()} ──")
    print(sheet.to_string(index=False))

    if actionable.empty:
        print("\nNo bets qualify today (edge below 2.5% floor after slippage).")
    else:
        total = actionable["Allocation (Br)"].sum()
        print(f"\n  {len(actionable)} bets | Total stake: {total:.0f} ETB "
              f"({100*total/TOTAL_BANKROLL_ETB:.1f}% of bankroll)")


if __name__ == "__main__":
    main()
