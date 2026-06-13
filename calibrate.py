"""
Calibrate the time-decay parameter PHI by sweeping half-life values and
measuring out-of-sample Brier score. Picks the half-life that makes the
model most accurate on data it never trained on.

Usage:
    python calibrate.py                     # uses real_fixtures.csv
    python calibrate.py --fixtures my.csv
"""
import argparse
import math

import numpy as np
import pandas as pd

import dixon_coles          # we'll monkey-patch PHI on each sweep
from data_loader import load_fixtures
from dixon_coles import DixonColesModel


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - outcomes) ** 2, axis=1)))


def evaluate_phi(df: pd.DataFrame, phi: float, n_windows: int = 6) -> float:
    """
    Walk-forward evaluation for a given PHI.
    Uses the last `n_windows` test periods (recent data = most relevant).
    Returns mean Brier score across those windows.
    """
    dixon_coles.PHI = phi  # patch the module-level constant

    last = df["date"].max()
    step = pd.Timedelta(days=180)
    train_window = pd.Timedelta(days=4 * 365)

    scores = []
    for i in range(n_windows, 0, -1):
        ref  = last - step * i
        test_end = ref + step
        train = df[df["date"] < ref]
        test  = df[(df["date"] >= ref) & (df["date"] < test_end)]
        if len(train) < 300 or len(test) < 20:
            continue

        try:
            model = DixonColesModel().fit(train, ref)
        except Exception:
            continue

        probs, outcomes = [], []
        for _, row in test.iterrows():
            try:
                ph, pd_, pa = model.match_outcome_probs(row["home_team"], row["away_team"])
            except Exception:
                continue
            probs.append([ph, pd_, pa])
            outcomes.append([
                int(row["home_goals"] > row["away_goals"]),
                int(row["home_goals"] == row["away_goals"]),
                int(row["home_goals"] < row["away_goals"]),
            ])

        if probs:
            scores.append(_brier(np.array(probs), np.array(outcomes)))

    return float(np.mean(scores)) if scores else float("nan")


def calibrate(fixtures_path: str = "real_fixtures.csv") -> pd.DataFrame:
    df = load_fixtures(fixtures_path)

    # Half-lives to sweep (in days): from 6 months to "no decay" (infinity proxy)
    half_lives = [180, 365, 547, 730, 1095, 1460, 9999]

    results = []
    original_phi = dixon_coles.PHI

    print(f"{'Half-life':>12}  {'PHI':>10}  {'Brier score':>12}")
    print("-" * 40)
    for hl in half_lives:
        phi = math.log(2) / hl if hl < 9000 else 0.0
        label = f"{hl}d" if hl < 9000 else "no decay"
        brier = evaluate_phi(df, phi)
        marker = " ←" if not results or brier < min(r["brier"] for r in results) else ""
        print(f"{label:>12}  {phi:>10.6f}  {brier:>12.5f}{marker}")
        results.append({"half_life_days": hl, "phi": phi, "brier": brier})

    dixon_coles.PHI = original_phi  # restore

    df_out = pd.DataFrame(results)
    best = df_out.loc[df_out["brier"].idxmin()]
    print(f"\nBest half-life: {best['half_life_days']} days  "
          f"(PHI={best['phi']:.6f}, Brier={best['brier']:.5f})")
    print(f"\nTo apply: edit dixon_coles.py line ~10 →")
    print(f"  PHI = math.log(2) / {int(best['half_life_days'])}")

    return df_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="real_fixtures.csv")
    args = ap.parse_args()
    calibrate(args.fixtures)
