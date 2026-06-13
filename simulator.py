"""Monte Carlo World Cup tournament simulator (100k runs, vectorized)."""
import numpy as np
import pandas as pd
from dixon_coles import DixonColesModel

N_SIMS = 100_000
RNG = np.random.default_rng(42)


def _penalty_winner(home: str, away: str, home_elo: dict, away_elo: dict) -> np.ndarray:
    """Return array of 0 (home wins) or 1 (away wins) per sim row."""
    diff = home_elo.get(home, 1500) - away_elo.get(away, 1500)
    p_home = 1 / (1 + 10 ** (-diff / 400)) * 0.55 + 0.225  # scaled to ~[0.2,0.8]
    p_home = float(np.clip(p_home, 0.2, 0.8))
    return RNG.random() > (1 - p_home)


def simulate_match_vectorized(lam: float, mu: float, n: int = N_SIMS) -> tuple[np.ndarray, np.ndarray]:
    """Draw n Poisson goal samples."""
    h = RNG.poisson(lam, n)
    a = RNG.poisson(mu, n)
    return h, a


def simulate_group(teams: list[str], model: DixonColesModel, n: int = N_SIMS) -> np.ndarray:
    """
    Simulate a group of 4 teams across n simulations.
    Returns array of shape (n, 4) with points, then resolves to top-2 indices per sim.
    Returns standings array (n, 4) -> top2 indices (n, 2).
    """
    pts   = np.zeros((n, 4), dtype=np.int32)
    gd    = np.zeros((n, 4), dtype=np.int32)
    gs    = np.zeros((n, 4), dtype=np.int32)
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    for i, j in pairs:
        lam, mu = model.predict_lambda_mu(teams[i], teams[j])
        h, a = simulate_match_vectorized(lam, mu, n)
        # points
        home_wins = h > a; draws = h == a; away_wins = h < a
        pts[:, i] += 3 * home_wins + draws
        pts[:, j] += 3 * away_wins + draws
        # gd & gs
        gd[:, i] += h - a;  gd[:, j] += a - h
        gs[:, i] += h;      gs[:, j] += a

    # Tiebreaker: pts desc, gd desc, gs desc -> argsort last two cols ascending then negate
    # Build composite sort key: primary=-pts, secondary=-gd, tertiary=-gs
    order = np.lexsort((-gs, -gd, -pts), axis=1)  # shape (n,4) ascending best
    top2 = order[:, :2]  # first two columns = rank 0 and rank 1
    return top2  # shape (n, 2) with team indices into `teams`


def simulate_knockout_match(
    home: str, away: str, model: DixonColesModel,
    elo: dict, n: int = N_SIMS
) -> np.ndarray:
    """Returns boolean array length n: True = home advances."""
    lam, mu = model.predict_lambda_mu(home, away)
    h, a = simulate_match_vectorized(lam, mu, n)
    home_adv = h > a
    draw_mask = h == a
    if draw_mask.any():
        # Penalty shootout: bernoulli per draw
        diff = elo.get(home, 1500) - elo.get(away, 1500)
        p_home_pen = float(np.clip(1 / (1 + 10 ** (-diff / 400)) * 0.55 + 0.225, 0.2, 0.8))
        pen = RNG.random(n) < p_home_pen
        home_adv[draw_mask] = pen[draw_mask]
    return home_adv


class TournamentSimulator:
    def __init__(self, model: DixonColesModel, groups: dict[str, list[str]], elo: dict = None):
        """
        groups: {'A': ['Team1','Team2','Team3','Team4'], ...}
        elo:    optional {team: elo_rating}
        """
        self.model = model
        self.groups = groups
        self.elo = elo or {t: 1500 for t in model.teams_}
        self.all_teams = [t for grp in groups.values() for t in grp]

    def run(self, n: int = N_SIMS) -> pd.DataFrame:
        team_idx = {t: i for i, t in enumerate(self.all_teams)}
        n_teams = len(self.all_teams)
        # Track round advancement counts
        reached = {stage: np.zeros(n_teams, dtype=np.int64)
                   for stage in ("qf", "sf", "final", "winner")}

        # ---- Group stage ----
        r16_slots = {}  # slot_name -> array of team indices (length n)
        slot_id = 0
        for grp_name, teams in self.groups.items():
            top2 = simulate_group(teams, self.model, n)  # (n,2)
            for rank in range(2):
                # top2[:,rank] contains local team index (0-3) in the group
                global_indices = np.array([team_idx[teams[t]] for t in range(4)])
                adv = global_indices[top2[:, rank]]  # (n,) global team indices
                r16_slots[f"{grp_name}_{rank}"] = adv  # winner=0, runner-up=1

        # ---- Build R16 bracket (standard WC bracket: 1A vs 2B, 1B vs 2A, ...) ----
        grp_names = list(self.groups.keys())
        # Pair groups: (A,B),(C,D),(E,F),(G,H) -> 8 matches
        brackets = []
        for k in range(0, len(grp_names), 2):
            g1, g2 = grp_names[k], grp_names[k + 1]
            brackets.append((f"{g1}_0", f"{g2}_1"))  # 1st group A vs 2nd group B
            brackets.append((f"{g2}_0", f"{g1}_1"))  # 1st group B vs 2nd group A

        # Track alive sims per bracket path
        alive = {slot: np.ones(n, dtype=bool) for slot in r16_slots}
        adv_teams = dict(r16_slots)  # slot -> (n,) array of global team index

        def run_ko_round(matches: list[tuple[str, str]], stage: str, next_slots: list[str]):
            new_adv = {}
            for match_idx, (s_home, s_away) in enumerate(matches):
                home_arr = adv_teams[s_home]  # (n,)
                away_arr = adv_teams[s_away]
                home_wins = np.zeros(n, dtype=bool)
                # Vectorize per-unique-matchup
                pairs_unique = set(zip(home_arr.tolist(), away_arr.tolist()))
                for (hi, ai) in pairs_unique:
                    mask = (home_arr == hi) & (away_arr == ai)
                    if not mask.any():
                        continue
                    ht = self.all_teams[hi]
                    at = self.all_teams[ai]
                    lam, mu = self.model.predict_lambda_mu(ht, at)
                    h_g = RNG.poisson(lam, mask.sum())
                    a_g = RNG.poisson(mu, mask.sum())
                    hw = h_g > a_g
                    draw = h_g == a_g
                    diff = self.elo.get(ht, 1500) - self.elo.get(at, 1500)
                    p_pen = float(np.clip(1 / (1 + 10 ** (-diff / 400)) * 0.55 + 0.225, 0.2, 0.8))
                    pen = RNG.random(mask.sum()) < p_pen
                    hw[draw] = pen[draw]
                    home_wins[mask] = hw

                winner = np.where(home_wins, home_arr, away_arr)
                loser  = np.where(home_wins, away_arr, home_arr)
                new_adv[next_slots[match_idx]] = winner
                # Record stage for losers (QF losers reach SF but don't advance; need to handle separately)
                if stage in reached:
                    for idx in range(n_teams):
                        mask_adv = winner == idx
                        reached[stage][idx] += mask_adv.sum()
            return new_adv

        # R16 -> QF (winners reach QF)
        qf_slots = [f"qf_{i}" for i in range(8)]
        new_adv = run_ko_round(brackets, "qf", qf_slots)
        # Record QF participants (all r16 winners = QF participants)
        for slot in qf_slots:
            for idx in range(n_teams):
                reached["qf"][idx] += (new_adv[slot] == idx).sum()
        adv_teams.update(new_adv)

        # QF -> SF
        qf_brackets = [(qf_slots[i], qf_slots[i + 1]) for i in range(0, 8, 2)]
        sf_slots = [f"sf_{i}" for i in range(4)]
        new_adv = run_ko_round(qf_brackets, "sf", sf_slots)
        for slot in sf_slots:
            for idx in range(n_teams):
                reached["sf"][idx] += (new_adv[slot] == idx).sum()
        adv_teams.update(new_adv)

        # SF -> Final
        sf_brackets = [(sf_slots[i], sf_slots[i + 1]) for i in range(0, 4, 2)]
        final_slots = ["final_0", "final_1"]
        new_adv = run_ko_round(sf_brackets, "final", final_slots)
        for slot in final_slots:
            for idx in range(n_teams):
                reached["final"][idx] += (new_adv[slot] == idx).sum()
        adv_teams.update(new_adv)

        # Final
        winner_arr = run_ko_round([(final_slots[0], final_slots[1])], "winner", ["champion"])
        for idx in range(n_teams):
            reached["winner"][idx] += (winner_arr["champion"] == idx).sum()

        df = pd.DataFrame({
            "team": self.all_teams,
            "P(Quarterfinal)": reached["qf"]  / n,
            "P(Semifinal)":    reached["sf"]   / n,
            "P(Final)":        reached["final"] / n,
            "P(Winner)":       reached["winner"]/ n,
        })
        return df.sort_values("P(Winner)", ascending=False).reset_index(drop=True)
