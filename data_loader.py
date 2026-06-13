"""Data ingestion and validation module."""
import pandas as pd

REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_goals", "away_goals", "match_importance_tier"}
VALID_TIERS = {"WC_FINAL", "CONFED_CAMP", "QUALIFIER", "FRIENDLY"}
IMPORTANCE = {"WC_FINAL": 4.0, "CONFED_CAMP": 3.0, "QUALIFIER": 2.5, "FRIENDLY": 1.0}


def load_fixtures(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    assert df["home_goals"].ge(0).all() and df["away_goals"].ge(0).all(), "Goals must be >= 0"
    invalid = set(df["match_importance_tier"]) - VALID_TIERS
    if invalid:
        raise ValueError(f"Invalid tiers: {invalid}")
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].astype(bool)
    return df
