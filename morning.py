#!/usr/bin/env python3
"""
Morning betting workflow — fetch odds, run model, verify against melbet, print bet sheet.

Usage:
    python morning.py          # interactive menu
    python morning.py wc       # World Cup
    python morning.py epl      # Premier League
    python morning.py laliga   # La Liga
    python morning.py bundesliga
    python morning.py seriea
    python morning.py ligue1
    python morning.py all      # all club leagues
"""
import json
import sys
import urllib.request
from datetime import date

import pandas as pd

from data_loader import load_fixtures
from dixon_coles import DixonColesModel
from kelly import (
    build_bet_sheet, half_kelly,
    TOTAL_BANKROLL_ETB, MAX_MATCHDAY_FRACTION, MAX_SINGLE_BET_FRACTION,
    SLIPPAGE, EDGE_FLOOR,
)

# ── Constants ─────────────────────────────────────────────────────────────────

CLUB_LEAGUES = {
    "2": ("epl",         "Premier League"),
    "3": ("laliga",      "La Liga"),
    "4": ("bundesliga",  "Bundesliga"),
    "5": ("seriea",      "Serie A"),
    "6": ("ligue1",      "Ligue 1"),
}

CLI_ALIASES = {
    "wc": "1", "epl": "2", "laliga": "3",
    "bundesliga": "4", "seriea": "5", "ligue1": "6", "all": "7",
}

# ── UI helpers ────────────────────────────────────────────────────────────────

def banner():
    today = date.today().strftime("%B %d, %Y")
    br    = f"{TOTAL_BANKROLL_ETB:,.2f}"
    w     = 46
    print(f"\n╔{'═'*w}╗")
    print(f"║{'World Cup Quant — Morning Run':^{w}}║")
    print(f"║{today:^{w}}║")
    print(f"║{f'Bankroll: {br} ETB':^{w}}║")
    print(f"╚{'═'*w}╝\n")

def menu() -> str:
    print("  What would you like to run today?\n")
    print("    [1]  World Cup")
    print("    [2]  Premier League")
    print("    [3]  La Liga")
    print("    [4]  Bundesliga")
    print("    [5]  Serie A")
    print("    [6]  Ligue 1")
    print("    [7]  All club leagues")
    print("    [0]  Exit\n")
    return input("  Choice: ").strip()

def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

# ── Odds fetchers ─────────────────────────────────────────────────────────────

def _dec(node: dict, key: str = "close"):
    v = (node.get(key) or {}).get("odds") or (node.get("open") or {}).get("odds")
    try:
        v = int(str(v).replace("+", ""))
        return round(v / 100 + 1, 3) if v > 0 else round(100 / abs(v) + 1, 3)
    except Exception:
        return None

def fetch_wc_odds() -> pd.DataFrame:
    req = urllib.request.Request(
        "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=12) as r:
        data = json.loads(r.read())

    rows = []
    for event in data.get("events", []):
        if event.get("status", {}).get("type", {}).get("name") != "STATUS_SCHEDULED":
            continue
        for comp in event.get("competitions", []):
            cs = comp.get("competitors", [])
            if len(cs) < 2:
                continue
            home = cs[0]["team"]["displayName"]
            away = cs[1]["team"]["displayName"]
            od = (comp.get("odds") or [{}])[0] or {}
            ml = od.get("moneyline") or {}
            dn = od.get("drawOdds") or {}
            h  = _dec(ml.get("home") or {})
            a  = _dec(ml.get("away") or {})
            dv = dn.get("close", {}).get("odds") or dn.get("open", {}).get("odds") or dn.get("moneyLine")
            try:
                dv = int(str(dv).replace("+", ""))
                d  = round(dv / 100 + 1, 3) if dv > 0 else round(100 / abs(dv) + 1, 3)
            except Exception:
                d = None
            if h and d and a:
                rows.append({"home_team": home, "away_team": away,
                             "home_odds": h, "draw_odds": d, "away_odds": a})
    return pd.DataFrame(rows)

def fetch_club_odds(league_key: str) -> pd.DataFrame:
    from club_main import fetch_odds
    return fetch_odds(league_key)

# ── Model runners ─────────────────────────────────────────────────────────────

def _bet_sheet(odds_df: pd.DataFrame, model: DixonColesModel) -> pd.DataFrame:
    bets = []
    for _, row in odds_df.iterrows():
        ph, pd_, pa = model.match_outcome_probs(row["home_team"], row["away_team"])
        label = f"{row['home_team']} vs {row['away_team']}"
        for prob, odds, choice in [
            (ph, row["home_odds"],  row["home_team"]),
            (pd_, row["draw_odds"], "Draw"),
            (pa, row["away_odds"],  row["away_team"]),
        ]:
            bets.append({"match": label, "choice": choice,
                         "model_prob": round(prob, 4), "decimal_odds": odds})
    return build_bet_sheet(bets, bankroll=TOTAL_BANKROLL_ETB)

def run_wc(odds_df: pd.DataFrame) -> pd.DataFrame:
    print("  Loading WC fixtures & fitting model...")
    df     = load_fixtures("real_fixtures.csv")
    ref    = pd.Timestamp.today().normalize()
    model  = DixonColesModel(host_team="Canada").fit(df, ref)
    cal_df = df[df["date"] >= ref - pd.Timedelta(days=365)]
    model.fit_calibration(cal_df)
    print(f"  γ={model.gamma_:.4f}  ρ={model.rho_:.4f}  T={model.temperature_:.4f}")
    return _bet_sheet(odds_df, model)

def run_club(league_key: str, odds_df: pd.DataFrame) -> pd.DataFrame:
    print("  Loading club fixtures & fitting model...")
    df     = load_fixtures("club_fixtures.csv")
    ref    = pd.Timestamp.today().normalize()
    model  = DixonColesModel(host_team=None).fit(df, ref)
    cal_df = df[df["date"] >= ref - pd.Timedelta(days=365)]
    model.fit_calibration(cal_df)
    print(f"  γ={model.gamma_:.4f}  ρ={model.rho_:.4f}  T={model.temperature_:.4f}")
    return _bet_sheet(odds_df, model)

# ── Display ───────────────────────────────────────────────────────────────────

def display_actionable(sheet: pd.DataFrame, label: str) -> pd.DataFrame:
    act = sheet[sheet["Allocation (Br)"] > 0].copy().reset_index(drop=True)
    if act.empty:
        print(f"  No qualifying bets for {label} (all below edge floor).")
        return act
    print(f"\n  {label} — {len(act)} qualifying bet(s):\n")
    for i, r in act.iterrows():
        print(f"    [{i+1}] {r['Match']}")
        print(f"         {r['Recommended Choice']:<22} "
              f"@ {r['Market Odds']:<7}  →  {r['Allocation (Br)']:.0f} ETB  "
              f"(model: {r['Model Probability']*100:.1f}%)")
    return act

# ── Melbet odds verification ──────────────────────────────────────────────────

def verify_melbet_odds(actionable: pd.DataFrame) -> pd.DataFrame:
    """
    For each qualifying bet, ask for melbet's live odds and recompute Kelly.
    Drops bets whose edge disappears at melbet prices.
    """
    print("\n  ESPN odds are a starting point — melbet will differ.")
    print("  Enter melbet's actual odds for each bet, or press Enter to keep ESPN's.\n")

    rows = []
    for _, r in actionable.iterrows():
        espn_odds = r["Market Odds"]
        prob      = r["Model Probability"]

        raw = input(f"  {r['Match']}  |  {r['Recommended Choice']}\n"
                    f"    ESPN: {espn_odds}  →  Melbet odds (Enter to keep): ").strip()

        if raw == "":
            melbet_odds = espn_odds
        else:
            try:
                melbet_odds = float(raw)
            except ValueError:
                print("    Invalid number — keeping ESPN odds.")
                melbet_odds = espn_odds

        fk = half_kelly(prob, melbet_odds)

        if fk == 0.0:
            edge = (prob * (melbet_odds - SLIPPAGE)) - 1
            print(f"    ✗ DROPPED — no edge at melbet odds "
                  f"({edge*100:.2f}% < {EDGE_FLOOR*100:.1f}% floor)\n")
            continue

        rows.append({
            "Match":              r["Match"],
            "Recommended Choice": r["Recommended Choice"],
            "Model Probability":  prob,
            "Market Odds":        melbet_odds,
            "_espn_odds":         espn_odds,
            "_raw_f":             fk,
        })
        print()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reapply 15% daily cap
    total_f = df["_raw_f"].sum()
    if total_f > MAX_MATCHDAY_FRACTION:
        df["_raw_f"] *= MAX_MATCHDAY_FRACTION / total_f

    # Per-bet fraction cap
    df["_raw_f"] = df["_raw_f"].clip(upper=MAX_SINGLE_BET_FRACTION)

    df["Allocation (Br)"] = (df["_raw_f"] * TOTAL_BANKROLL_ETB).round(2)
    df["Half-Kelly %"]    = (df["_raw_f"] * 100).round(4)

    return df.drop(columns=["_raw_f", "_espn_odds"])

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    banner()

    if len(sys.argv) > 1:
        alias  = sys.argv[1].lower()
        choice = CLI_ALIASES.get(alias)
        if not choice:
            print(f"  Unknown alias '{alias}'. Options: {list(CLI_ALIASES)}")
            sys.exit(1)
        print(f"  Running: {alias.upper()}\n")
    else:
        choice = menu()

    if choice == "0":
        print("  Goodbye.")
        return

    all_actionable: list[pd.DataFrame] = []

    # ── World Cup ──
    if choice == "1":
        section("World Cup")
        print("  Fetching live WC odds from ESPN...")
        odds_df = fetch_wc_odds()
        if odds_df.empty:
            print("  No WC fixtures scheduled today with odds.")
        else:
            print(f"  {len(odds_df)} fixture(s) found.")
            sheet = run_wc(odds_df)
            all_actionable.append(display_actionable(sheet, "World Cup"))

    # ── Single club league ──
    elif choice in CLUB_LEAGUES:
        league_key, display = CLUB_LEAGUES[choice]
        section(display)
        print("  Fetching live odds from ESPN...")
        odds_df = fetch_club_odds(league_key)
        if odds_df.empty:
            print("  No fixtures scheduled today with odds.")
        else:
            print(f"  {len(odds_df)} fixture(s) found.")
            sheet = run_club(league_key, odds_df)
            all_actionable.append(display_actionable(sheet, display))

    # ── All club leagues ──
    elif choice == "7":
        for _, (league_key, display) in CLUB_LEAGUES.items():
            section(display)
            print("  Fetching live odds from ESPN...")
            try:
                odds_df = fetch_club_odds(league_key)
            except Exception as e:
                print(f"  Failed: {e}")
                continue
            if odds_df.empty:
                print("  No fixtures today.")
                continue
            print(f"  {len(odds_df)} fixture(s) found.")
            sheet = run_club(league_key, odds_df)
            all_actionable.append(display_actionable(sheet, display))

    else:
        print("  Invalid choice.")
        return

    # ── Combine & verify ──────────────────────────────────────────────────────
    combined = pd.concat(
        [a for a in all_actionable if not a.empty], ignore_index=True
    ) if all_actionable else pd.DataFrame()

    if combined.empty:
        print("\n  No qualifying bets today. Check back tomorrow.\n")
        return

    total = combined["Allocation (Br)"].sum()
    pct   = 100 * total / TOTAL_BANKROLL_ETB
    section(f"Model output: {len(combined)} bet(s)  |  {total:.0f} ETB  ({pct:.1f}% of bankroll)")

    combined = verify_melbet_odds(combined)

    if combined.empty:
        print("\n  No bets remain after odds check. Nothing to place today.\n")
        return

    total = combined["Allocation (Br)"].sum()
    pct   = 100 * total / TOTAL_BANKROLL_ETB
    section(f"Final bet sheet — place these manually on melbet")
    print()
    for _, r in combined.iterrows():
        print(f"  {r['Match']}")
        print(f"    Bet    : {r['Recommended Choice']}")
        print(f"    Odds   : {r['Market Odds']}")
        print(f"    Stake  : {r['Allocation (Br)']:.0f} ETB")
        print(f"    Model  : {r['Model Probability']*100:.1f}% probability")
        print()
    print(f"  Total stake : {total:.0f} ETB  ({pct:.1f}% of bankroll)")
    print(f"\n  Tonight, update bankroll:")
    print(f"  python -c \"from kelly import update_bankroll; update_bankroll(X)\"\n")
    print(f"  Think about Overiding bets if:  Key player confirmed absent or Rotating Squad(Dead rubber)\n")


if __name__ == "__main__":
    main()
