"""
Scrape historical and upcoming match odds from ESPN (DraftKings closing lines).
Covers major international tournaments 2014–2026 where ESPN has data.

Usage:
    python odds_scraper.py                  # saves → historical_odds.csv
    python odds_scraper.py --out my.csv
"""
import argparse
import json
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd

# ESPN competition slugs
COMPETITIONS = {
    "fifa.world":             "FIFA World Cup",
    "uefa.euro":              "UEFA Euro",
    "conmebol.america":       "Copa America",
    "concacaf.gold":          "CONCACAF Gold Cup",
    "caf.nations":            "AFCON",
    "afc.asian.cup":          "AFC Asian Cup",
    "uefa.nations":           "UEFA Nations League",
    "concacaf.nations.league":"CONCACAF Nations League",
}

# Tournament windows to crawl: (competition_slug, start_date, end_date)
WINDOWS = [
    # World Cup
    ("fifa.world",        date(2026, 6, 11),  date(2026, 7, 19)),
    ("fifa.world",        date(2022, 11, 20), date(2022, 12, 18)),
    ("fifa.world",        date(2018, 6, 14),  date(2018, 7, 15)),
    ("fifa.world",        date(2014, 6, 12),  date(2014, 7, 13)),
    # Euro
    ("uefa.euro",         date(2024, 6, 14),  date(2024, 7, 14)),
    ("uefa.euro",         date(2021, 6, 11),  date(2021, 7, 11)),
    ("uefa.euro",         date(2016, 6, 10),  date(2016, 7, 10)),
    # Copa America
    ("conmebol.america",  date(2024, 6, 20),  date(2024, 7, 14)),
    ("conmebol.america",  date(2021, 6, 13),  date(2021, 7, 10)),
    ("conmebol.america",  date(2019, 6, 14),  date(2019, 7,  7)),
    ("conmebol.america",  date(2016, 6,  3),  date(2016, 6, 26)),
    ("conmebol.america",  date(2015, 6, 11),  date(2015, 7,  4)),
    # Gold Cup
    ("concacaf.gold",     date(2023, 6, 24),  date(2023, 7, 16)),
    ("concacaf.gold",     date(2021, 7, 10),  date(2021, 8,  1)),
    ("concacaf.gold",     date(2019, 6, 15),  date(2019, 7,  7)),
    # AFCON
    ("caf.nations",       date(2023, 1, 13),  date(2023, 2, 11)),
    ("caf.nations",       date(2022, 1,  9),  date(2022, 2,  6)),
    ("caf.nations",       date(2019, 6, 21),  date(2019, 7, 19)),
    # AFC Asian Cup
    ("afc.asian.cup",     date(2023, 1, 12),  date(2023, 2, 11)),
    ("afc.asian.cup",     date(2019, 1,  5),  date(2019, 2,  1)),
    # UEFA Nations League (selected finals/semis only to limit volume)
    ("uefa.nations",      date(2023, 6, 14),  date(2023, 6, 18)),
    ("uefa.nations",      date(2021, 10,  7), date(2021, 10, 10)),
    # CONCACAF Nations League
    ("concacaf.nations.league", date(2023, 6, 15), date(2023, 6, 18)),
]

_NAME_MAP = {
    "United States":      "USA",
    "Korea Republic":     "South Korea",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "IR Iran":            "Iran",
    "Cote d'Ivoire":      "Ivory Coast",
}


def _ml_to_decimal(ml_str: str) -> float | None:
    """Convert American moneyline string to decimal odds."""
    try:
        ml = int(str(ml_str).replace("+", ""))
        if ml > 0:
            return round(ml / 100 + 1, 3)
        elif ml < 0:
            return round(100 / abs(ml) + 1, 3)
    except (ValueError, TypeError):
        pass
    return None


def _fetch_day(slug: str, d: date) -> list[dict]:
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/soccer/"
        f"{slug}/scoreboard?dates={d.strftime('%Y%m%d')}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
    except Exception:
        return []

    rows = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            odds_list = comp.get("odds", []) or []
            if not odds_list:
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            # ESPN: index 0 = home, index 1 = away (for soccer)
            home_team = _NAME_MAP.get(
                competitors[0].get("team", {}).get("displayName", ""),
                competitors[0].get("team", {}).get("displayName", ""),
            )
            away_team = _NAME_MAP.get(
                competitors[1].get("team", {}).get("displayName", ""),
                competitors[1].get("team", {}).get("displayName", ""),
            )
            if not home_team or not away_team:
                continue

            home_score = competitors[0].get("score")
            away_score = competitors[1].get("score")

            od = odds_list[0] or {}
            ml = od.get("moneyline") or {}

            # Prefer closing line; fall back to opening if close absent
            def _odds(side: str) -> str | None:
                node = ml.get(side, {}) or {}
                return (
                    (node.get("close") or {}).get("odds")
                    or (node.get("open") or {}).get("odds")
                )

            def _draw_odds(od) -> str | None:
                draw_node = od.get("drawOdds") or {}
                # drawOdds can be {"moneyLine": 250} or {"close":{"odds":...}}
                if "moneyLine" in draw_node:
                    return str(draw_node["moneyLine"])
                close = (draw_node.get("close") or {}).get("odds")
                if close:
                    return str(close)
                open_ = (draw_node.get("open") or {}).get("odds")
                return str(open_) if open_ else None

            h_dec = _ml_to_decimal(_odds("home"))
            d_dec = _ml_to_decimal(_draw_odds(od))
            a_dec = _ml_to_decimal(_odds("away"))

            if not (h_dec and d_dec and a_dec):
                continue
            if not all(1.0 < x < 100 for x in (h_dec, d_dec, a_dec)):
                continue

            row = {
                "date":       d.strftime("%Y-%m-%d"),
                "home_team":  home_team,
                "away_team":  away_team,
                "home_odds":  h_dec,
                "draw_odds":  d_dec,
                "away_odds":  a_dec,
                "competition": COMPETITIONS.get(slug, slug),
            }
            if home_score is not None and away_score is not None:
                try:
                    row["home_goals"] = int(float(home_score))
                    row["away_goals"] = int(float(away_score))
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def scrape(save_path: str = "historical_odds.csv", delay: float = 0.3) -> pd.DataFrame:
    all_rows: list[dict] = []
    seen: set[tuple] = set()

    for slug, start, end in WINDOWS:
        comp_name = COMPETITIONS.get(slug, slug)
        d = start
        while d <= end:
            rows = _fetch_day(slug, d)
            for row in rows:
                key = (row["date"], row["home_team"], row["away_team"])
                if key not in seen:
                    seen.add(key)
                    all_rows.append(row)
            d += timedelta(days=1)
            time.sleep(delay)
        if any(r["competition"] == comp_name for r in all_rows):
            n = sum(1 for r in all_rows if r["competition"] == comp_name)
            print(f"  {comp_name} ({start.year}): {n} matches")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No odds scraped.")
        return df

    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"\nSaved {len(df):,} matches with odds → {save_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="historical_odds.csv")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()
    df = scrape(save_path=args.out, delay=args.delay)
    if not df.empty:
        print(df.groupby("competition").size().to_string())
