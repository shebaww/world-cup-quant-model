"""Download and normalise historical club football data from football-data.co.uk."""
import io
import urllib.request
import pandas as pd

# League codes → (fdc_code, importance_tier, display_name)
LEAGUES: dict[str, tuple[str, str, str]] = {
    "epl":        ("E0",  "CONFED_CAMP", "Premier League"),
    "championship":("E1", "QUALIFIER",   "Championship"),
    "laliga":     ("SP1", "CONFED_CAMP", "La Liga"),
    "bundesliga": ("D1",  "CONFED_CAMP", "Bundesliga"),
    "seriea":     ("I1",  "CONFED_CAMP", "Serie A"),
    "ligue1":     ("F1",  "CONFED_CAMP", "Ligue 1"),
}

# Seasons available on football-data.co.uk
SEASONS = [
    "1516","1617","1718","1819","1920",
    "2021","2122","2223","2324","2425","2526",
]


def _fetch_season(fdc_code: str, season: str) -> pd.DataFrame | None:
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{fdc_code}.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            content = r.read().decode("latin-1")
        df = pd.read_csv(io.StringIO(content))
        # Strip BOM from first column name if present
        df.columns = [c.lstrip("﻿").lstrip("ï»¿") for c in df.columns]
        needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not needed.issubset(df.columns):
            print(f"    {season}: missing columns, skipping")
            return None
        df = df[list(needed)].dropna(subset=["FTHG", "FTAG"])
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        return df
    except Exception as e:
        print(f"    {season}: {e}")
        return None


def fetch(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    save_path: str = "club_fixtures.csv",
) -> pd.DataFrame:
    """
    Download historical club data and save to CSV.

    leagues: list of keys from LEAGUES (default: all top-5 + EPL)
    seasons: list of season codes (default: all available)
    """
    if leagues is None:
        leagues = ["epl", "laliga", "bundesliga", "seriea", "ligue1"]
    if seasons is None:
        seasons = SEASONS

    all_rows = []
    for league_key in leagues:
        if league_key not in LEAGUES:
            print(f"Unknown league '{league_key}', skipping")
            continue
        fdc_code, tier, display = LEAGUES[league_key]
        print(f"Fetching {display} ({fdc_code}) ...")
        for season in seasons:
            df = _fetch_season(fdc_code, season)
            if df is None:
                continue
            completed = df[["FTHG", "FTAG"]].notna().all(axis=1).sum()
            print(f"  {season}: {completed} completed matches")
            df["match_importance_tier"] = tier
            df["neutral"] = False
            df = df.rename(columns={
                "Date": "date",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
            })
            df["home_goals"] = df["home_goals"].astype(int)
            df["away_goals"] = df["away_goals"].astype(int)
            all_rows.append(df[["date","home_team","away_team",
                                 "home_goals","away_goals",
                                 "match_importance_tier","neutral"]])

    combined = (
        pd.concat(all_rows)
        .drop_duplicates(subset=["date","home_team","away_team"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    combined.to_csv(save_path, index=False)
    print(f"\nSaved {len(combined):,} matches to '{save_path}'")
    print(f"Spanning {combined['date'].min().date()} → {combined['date'].max().date()}")
    return combined


if __name__ == "__main__":
    fetch()
