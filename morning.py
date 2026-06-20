#!/usr/bin/env python3
"""
Morning betting workflow — interactive odds fetch, model run, and bet placement.

Usage:
    python morning.py          # interactive menu
    python morning.py wc       # skip menu, run World Cup directly
    python morning.py epl      # skip menu, run EPL directly
"""
import asyncio
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from data_loader import load_fixtures
from dixon_coles import DixonColesModel
from kelly import build_bet_sheet, half_kelly, TOTAL_BANKROLL_ETB, MAX_MATCHDAY_FRACTION, SLIPPAGE, EDGE_FLOOR
from bet_executor import place_bet, AUTH_FILE

# ── Constants ────────────────────────────────────────────────────────────────

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
    df       = load_fixtures("real_fixtures.csv")
    ref      = pd.Timestamp.today().normalize()
    model    = DixonColesModel(host_team="Canada").fit(df, ref)
    cal_df   = df[df["date"] >= ref - pd.Timedelta(days=365)]
    model.fit_calibration(cal_df)
    print(f"  γ={model.gamma_:.4f}  ρ={model.rho_:.4f}  T={model.temperature_:.4f}")
    return _bet_sheet(odds_df, model)

def run_club(league_key: str, odds_df: pd.DataFrame) -> pd.DataFrame:
    print(f"  Loading club fixtures & fitting model...")
    df     = load_fixtures("club_fixtures.csv")
    ref    = pd.Timestamp.today().normalize()
    model  = DixonColesModel(host_team=None).fit(df, ref)
    cal_df = df[df["date"] >= ref - pd.Timedelta(days=365)]
    model.fit_calibration(cal_df)
    print(f"  γ={model.gamma_:.4f}  ρ={model.rho_:.4f}  T={model.temperature_:.4f}")
    return _bet_sheet(odds_df, model)

# ── Display & placement ───────────────────────────────────────────────────────

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

def verify_melbet_odds(actionable: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through each qualifying bet, let user correct ESPN odds to melbet's
    actual odds, then recompute Kelly stakes. Bets that lose edge at melbet
    odds are dropped with a warning.
    """
    print("\n  ESPN odds may differ from melbet. Check each one and enter the")
    print("  real melbet odds — or press Enter to keep the ESPN figure.\n")

    rows = []
    for _, r in actionable.iterrows():
        espn_odds = r["Market Odds"]
        prob      = r["Model Probability"]
        label     = f"  {r['Match']}  |  {r['Recommended Choice']}"

        raw = input(f"{label}\n    ESPN: {espn_odds}  →  Melbet odds (Enter to keep): ").strip()

        if raw == "":
            melbet_odds = espn_odds
        else:
            try:
                melbet_odds = float(raw)
            except ValueError:
                print("    Invalid — keeping ESPN odds.")
                melbet_odds = espn_odds

        fk = half_kelly(prob, melbet_odds)

        if fk == 0.0:
            adj = melbet_odds - SLIPPAGE
            edge = (prob * adj) - 1
            if edge < EDGE_FLOOR:
                print(f"    ✗ DROPPED — edge vanishes at melbet odds "
                      f"(edge {edge*100:.2f}% < {EDGE_FLOOR*100:.1f}% floor)\n")
                continue

        rows.append({
            "Match":               r["Match"],
            "Recommended Choice":  r["Recommended Choice"],
            "Model Probability":   prob,
            "Market Odds":         melbet_odds,
            "_espn_odds":          espn_odds,
            "_raw_f":              fk,
        })
        print()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reapply 15% daily cap
    total_f = df["_raw_f"].sum()
    if total_f > MAX_MATCHDAY_FRACTION:
        df["_raw_f"] *= MAX_MATCHDAY_FRACTION / total_f

    br = TOTAL_BANKROLL_ETB
    df["Allocation (Br)"] = (df["_raw_f"] * br).round(2)
    df["Half-Kelly %"]    = (df["_raw_f"] * 100).round(4)

    # Show any stake changes
    section("Recalculated stakes using melbet odds")
    for _, r in df.iterrows():
        espn = r["_espn_odds"]
        mb   = r["Market Odds"]
        flag = f"  (ESPN was {espn})" if mb != espn else ""
        print(f"  {r['Match']}  |  {r['Recommended Choice']}")
        print(f"    Odds: {mb}  →  {r['Allocation (Br)']:.0f} ETB{flag}\n")

    return df.drop(columns=["_raw_f", "_espn_odds"])


async def place_all_interactive(actionable: pd.DataFrame) -> int:
    if actionable.empty:
        return 0
    if not Path(AUTH_FILE).exists():
        print(f"\n  ⚠  '{AUTH_FILE}' not found.")
        print("     Run  python bet_executor.py auth  first, then come back.")
        return 0

    placed = 0
    print()
    for i, row in actionable.iterrows():
        stake  = row["Allocation (Br)"]
        choice = row["Recommended Choice"]
        odds   = row["Market Odds"]
        match  = row["Match"]

        prompt = (f"  Place [{i+1}] {match}  |  {choice} @ {odds}"
                  f"  |  {stake:.0f} ETB   [y / n / q to quit]: ")
        ans = input(prompt).strip().lower()

        if ans == "q":
            print("  Stopping placement.")
            break
        if ans != "y":
            print("  Skipped.")
            continue

        print(f"\n  Go to melbet-et.com, find this match, and:")
        print(f"  1. Right-click the '{choice}' odds button → Inspect")
        print(f"  2. Right-click the highlighted element → Copy → Copy selector")
        url      = input("  Paste the match URL       : ").strip()
        selector = input("  Paste the CSS selector    : ").strip()

        print("  Launching browser to place bet...")
        try:
            await place_bet(url, selector, float(stake))
            print(f"  ✓ Bet placed: {choice} @ {odds} for {stake:.0f} ETB")
            placed += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        print()

    return placed

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    banner()

    # CLI shortcut: python morning.py wc / epl / laliga / etc.
    if len(sys.argv) > 1:
        alias = sys.argv[1].lower()
        choice = CLI_ALIASES.get(alias)
        if not choice:
            print(f"Unknown alias '{alias}'. Options: {list(CLI_ALIASES)}")
            sys.exit(1)
        print(f"  Running: {alias.upper()}")
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
            act   = display_actionable(sheet, "World Cup")
            all_actionable.append(act)

    # ── Single club league ──
    elif choice in CLUB_LEAGUES:
        league_key, display = CLUB_LEAGUES[choice]
        section(display)
        print(f"  Fetching live odds from ESPN...")
        odds_df = fetch_club_odds(league_key)
        if odds_df.empty:
            print("  No fixtures scheduled today with odds.")
        else:
            print(f"  {len(odds_df)} fixture(s) found.")
            sheet = run_club(league_key, odds_df)
            act   = display_actionable(sheet, display)
            all_actionable.append(act)

    # ── All club leagues ──
    elif choice == "7":
        for _, (league_key, display) in CLUB_LEAGUES.items():
            section(display)
            print(f"  Fetching live odds from ESPN...")
            try:
                odds_df = fetch_club_odds(league_key)
            except Exception as e:
                print(f"  Failed to fetch odds: {e}")
                continue
            if odds_df.empty:
                print("  No fixtures today.")
                continue
            print(f"  {len(odds_df)} fixture(s) found.")
            sheet = run_club(league_key, odds_df)
            act   = display_actionable(sheet, display)
            all_actionable.append(act)

    else:
        print("  Invalid choice.")
        return

    # ── Summary & placement ──
    combined = pd.concat(
        [a for a in all_actionable if not a.empty], ignore_index=True
    ) if all_actionable else pd.DataFrame()

    if combined.empty:
        print("\n  No qualifying bets today. Check back tomorrow.\n")
        return

    total = combined["Allocation (Br)"].sum()
    pct   = 100 * total / TOTAL_BANKROLL_ETB
    section(f"Summary: {len(combined)} bet(s)  |  {total:.0f} ETB at risk  ({pct:.1f}% of bankroll)")

    # ── Verify odds against melbet before placing ──────────────────────────────
    combined = verify_melbet_odds(combined)
    if combined.empty:
        print("\n  No bets remain after melbet odds verification.\n")
        return

    total = combined["Allocation (Br)"].sum()
    pct   = 100 * total / TOTAL_BANKROLL_ETB
    print(f"  Final: {len(combined)} bet(s)  |  {total:.0f} ETB at risk  ({pct:.1f}% of bankroll)\n")

    go = input("  Proceed to place bets on melbet? [y/n]: ").strip().lower()
    if go != "y":
        print("\n  Exiting without placing bets.")
        print(f"  Run again when ready, or place manually using the sheet above.\n")
        return

    placed = await place_all_interactive(combined)

    section(f"Done: {placed}/{len(combined)} bets placed")
    if placed > 0:
        net = input("\n  After results: enter net P&L to update bankroll (or press Enter to skip): ").strip()
        if net:
            try:
                from kelly import update_bankroll
                update_bankroll(float(net))
            except ValueError:
                print("  Invalid number — update bankroll manually.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
