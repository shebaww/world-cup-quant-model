"""Download real international match results from the public dataset."""
import io
import urllib.request
import pandas as pd

_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

_NAME_MAP = {
    "United States":      "USA",
    "Korea Republic":     "South Korea",
    "IR Iran":            "Iran",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Côte d'Ivoire":      "Ivory Coast",
    "Cape Verde":         "Cape Verde Islands",
}

_TIER_RULES = [
    (["fifa world cup"],                                                    "WC_FINAL"),
    (["euro", "copa america", "africa cup", "asian cup", "gold cup",
      "nations cup", "nations league"],                                    "CONFED_CAMP"),
    (["qualif"],                                                            "QUALIFIER"),
]


def _map_tier(tournament: str) -> str:
    t = tournament.lower()
    for keywords, tier in _TIER_RULES:
        if any(k in t for k in keywords):
            return tier
    return "FRIENDLY"


def fetch(since_year: int = 2010, save_path: str = "real_fixtures.csv") -> pd.DataFrame:
    """Download international results and save in load_fixtures-compatible format."""
    print(f"Fetching international results (since {since_year}) ...")
    with urllib.request.urlopen(_URL, timeout=30) as resp:
        raw = pd.read_csv(io.BytesIO(resp.read()))

    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[
        (raw["date"].dt.year >= since_year) &
        raw["home_score"].notna() &
        raw["away_score"].notna()
    ].copy()

    for col in ("home_team", "away_team"):
        raw[col] = raw[col].replace(_NAME_MAP)

    df = pd.DataFrame({
        "date":                  raw["date"].dt.strftime("%Y-%m-%d"),
        "home_team":             raw["home_team"],
        "away_team":             raw["away_team"],
        "home_goals":            raw["home_score"].astype(int),
        "away_goals":            raw["away_score"].astype(int),
        "match_importance_tier": raw["tournament"].apply(_map_tier),
        "neutral":               raw["neutral"].astype(bool),
    })

    df.to_csv(save_path, index=False)
    print(f"  {len(df):,} matches saved → {save_path}")
    return df


if __name__ == "__main__":
    fetch()
