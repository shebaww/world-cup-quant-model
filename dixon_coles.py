"""Dixon-Coles Poisson model with time-decay weighting and MLE fitting."""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln as sp_gammaln
from scipy.stats import poisson
from data_loader import IMPORTANCE

PHI = math.log(2) / 730  # half-life = 2 years


def compute_weights(df: pd.DataFrame, ref_date: pd.Timestamp) -> np.ndarray:
    days_elapsed = (ref_date - df["date"]).dt.days.values
    tier_k = df["match_importance_tier"].map(IMPORTANCE).values
    return tier_k * np.exp(-PHI * days_elapsed)


def tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction (scalar version used in match_outcome_probs)."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


L2_LAMBDA = 0.1  # Ridge regularization — pulls unproven teams toward global mean


def dc_log_likelihood(params, teams, home_idx, away_idx, home_g, away_g, weights, neutral_mask):
    n = len(teams)
    log_alpha = params[:n]
    log_beta  = params[n:2*n]
    alpha = np.exp(log_alpha)
    beta  = np.exp(log_beta)
    gamma = math.exp(params[2*n])
    rho   = params[2*n + 1]

    # Suppress home advantage for neutral-venue matches
    gamma_arr = np.where(neutral_mask, 1.0, gamma)
    lam = alpha[home_idx] * beta[away_idx] * gamma_arr
    mu  = alpha[away_idx] * beta[home_idx]

    # Vectorized Poisson log-likelihood (avoids per-match Python loop)
    poisson_ll = (home_g * np.log(lam) - lam - sp_gammaln(home_g + 1) +
                  away_g * np.log(mu)  - mu  - sp_gammaln(away_g + 1))

    # Tau corrections for low-score cells (0-0, 1-0, 0-1, 1-1)
    tau_vals = np.ones(len(home_g))
    tau_vals[(home_g == 0) & (away_g == 0)] = 1 - lam[(home_g == 0) & (away_g == 0)] * mu[(home_g == 0) & (away_g == 0)] * rho
    tau_vals[(home_g == 1) & (away_g == 0)] = 1 + mu[(home_g == 1) & (away_g == 0)] * rho
    tau_vals[(home_g == 0) & (away_g == 1)] = 1 + lam[(home_g == 0) & (away_g == 1)] * rho
    tau_vals[(home_g == 1) & (away_g == 1)] = 1 - rho

    if np.any(tau_vals <= 0):
        return 1e9

    ll = np.sum(weights * (np.log(tau_vals) + poisson_ll))
    l2_penalty = L2_LAMBDA * (np.sum(log_alpha ** 2) + np.sum(log_beta ** 2))
    return -ll + l2_penalty


class DixonColesModel:
    # FIFA rank → (attack_alpha, defense_beta) priors for unseen teams.
    # alpha: attacking strength (higher = more goals scored)
    # beta:  defensive weakness (higher = more goals conceded)
    PRIORS: dict[str, tuple[float, float]] = {
        "Argentina":              (1.45, 0.72),
        "France":                 (1.42, 0.74),
        "England":                (1.38, 0.76),
        "Brazil":                 (1.35, 0.78),
        "Belgium":                (1.32, 0.80),
        "Portugal":               (1.30, 0.81),
        "Netherlands":            (1.28, 0.82),
        "Spain":                  (1.26, 0.83),
        "Croatia":                (1.18, 0.87),
        "Morocco":                (1.12, 0.90),
        "USA":                    (1.10, 0.91),
        "Mexico":                 (1.08, 0.92),
        "Canada":                 (1.06, 0.94),
        "Uruguay":                (1.14, 0.89),
        "Colombia":               (1.10, 0.91),
        "Germany":                (1.30, 0.81),
        "Italy":                  (1.22, 0.86),
        "Switzerland":            (1.10, 0.91),
        "Denmark":                (1.08, 0.92),
        "Austria":                (1.05, 0.95),
        "Bosnia and Herzegovina": (0.98, 1.02),
        "Paraguay":               (0.90, 1.08),
        "Saudi Arabia":           (0.88, 1.10),
        "Iran":                   (0.86, 1.12),
        "Senegal":                (1.00, 1.00),
        "Tunisia":                (0.85, 1.13),
        "Cameroon":               (0.88, 1.10),
        "Ghana":                  (0.87, 1.11),
        "Ecuador":                (0.92, 1.06),
        "Costa Rica":             (0.84, 1.14),
        "Qatar":                  (0.80, 1.18),
        "South Korea":            (0.95, 1.04),
        "Japan":                  (1.00, 1.00),
        "Australia":              (0.92, 1.06),
        "Serbia":                 (0.98, 1.02),
        "Poland":                 (0.96, 1.03),
    }

    def __init__(self, host_team: str = None, priors: dict[str, tuple[float, float]] = None):
        self.host_team = host_team
        self._priors = {**self.PRIORS, **(priors or {})}
        self.teams_ = None
        self.alpha_ = None
        self.beta_ = None
        self.gamma_ = None
        self.rho_ = None
        self.temperature_ = 1.0  # calibration scalar; < 1 sharpens, > 1 softens

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp, min_matches: int = 15):
        # Only fit parameters for teams with enough data; use priors/fallback for the rest.
        # Keeps parameter space under ~100 teams for tractable optimization.
        # Falls back to a lower threshold if min_matches would leave too few teams.
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        threshold = min_matches
        for threshold in [min_matches, max(min_matches // 3, 3), 1]:
            qualified = set(counts[counts >= threshold].index)
            if len(qualified) >= 2:
                break
        df = df[df["home_team"].isin(qualified) & df["away_team"].isin(qualified)].copy()

        teams = sorted(qualified)
        self.teams_ = teams
        t2i = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        home_idx = df["home_team"].map(t2i).values
        away_idx = df["away_team"].map(t2i).values

        # Goal cap: clamp margin > 3 to reduce blowout noise
        home_g_raw = df["home_goals"].values.copy()
        away_g_raw = df["away_goals"].values.copy()
        diff = home_g_raw - away_g_raw
        home_g = home_g_raw - np.maximum(diff - 3, 0)
        away_g = away_g_raw - np.maximum(-diff - 3, 0)

        weights      = compute_weights(df, ref_date)
        neutral_mask = df["neutral"].values.astype(bool) if "neutral" in df.columns else np.zeros(len(df), dtype=bool)

        x0 = np.zeros(2 * n + 2)
        x0[2*n]     = math.log(1.2)
        x0[2*n + 1] = -0.1

        # L-BFGS-B: much faster than SLSQP for large parameter spaces.
        # Post-hoc normalization enforces mean(log_alpha)=0 (identifiability constraint).
        alpha_beta_bounds = [(-3.0, 3.0)] * (2 * n)
        extra_bounds      = [(-0.5, 1.5), (-0.5, 0.5)]
        bounds = alpha_beta_bounds + extra_bounds

        result = minimize(
            dc_log_likelihood,
            x0,
            args=(teams, home_idx, away_idx, home_g, away_g, weights, neutral_mask),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-7},
        )

        params = result.x.copy()
        # Enforce mean(log_alpha) = 0 post-hoc for identifiability
        params[:n] -= params[:n].mean()

        self.alpha_ = dict(zip(teams, np.exp(params[:n])))
        self.beta_  = dict(zip(teams, np.exp(params[n:2*n])))
        self.gamma_ = float(np.exp(np.clip(params[2*n], -0.5, 1.5)))
        self.rho_   = float(np.clip(params[2*n + 1], -0.5, 0.5))
        self._avg_alpha = float(np.mean(list(self.alpha_.values())))
        self._avg_beta  = float(np.mean(list(self.beta_.values())))
        return self

    def fit_calibration(self, cal_df: pd.DataFrame) -> "DixonColesModel":
        """Fit temperature T on held-out matches to correct systematic over/under-confidence."""
        from scipy.optimize import minimize_scalar

        probs, actuals = [], []
        for _, row in cal_df.iterrows():
            try:
                ph, pd_, pa = self._raw_outcome_probs(row["home_team"], row["away_team"])
            except Exception:
                continue
            probs.append([ph, pd_, pa])
            actuals.append([
                int(row["home_goals"] > row["away_goals"]),
                int(row["home_goals"] == row["away_goals"]),
                int(row["home_goals"] < row["away_goals"]),
            ])

        if len(probs) < 20:
            return self

        p = np.array(probs, dtype=float)
        y = np.array(actuals, dtype=float)

        def nll(T):
            logits = np.log(np.clip(p, 1e-10, 1.0)) / T
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            cal = exp_l / exp_l.sum(axis=1, keepdims=True)
            return -float(np.sum(y * np.log(np.clip(cal, 1e-10, 1.0))))

        res = minimize_scalar(nll, bounds=(0.3, 3.0), method="bounded")
        self.temperature_ = float(res.x)
        return self

    def _raw_outcome_probs(self, home: str, away: str, neutral: bool = False, max_goals: int = 10) -> tuple[float, float, float]:
        """Uncalibrated outcome probabilities straight from the Poisson model."""
        lam, mu = self.predict_lambda_mu(home, away, neutral=neutral)
        prob_matrix = np.outer(
            poisson.pmf(np.arange(max_goals + 1), lam),
            poisson.pmf(np.arange(max_goals + 1), mu),
        )
        for h in range(2):
            for a in range(2):
                prob_matrix[h, a] *= tau(h, a, lam, mu, self.rho_)
        p_home = float(np.tril(prob_matrix, -1).sum())
        p_draw = float(np.trace(prob_matrix))
        p_away = float(np.triu(prob_matrix,  1).sum())
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def _fallback(self, team: str) -> tuple[float, float]:
        """Returns (alpha, beta) for a team not in the fitted dataset."""
        if team in self._priors:
            return self._priors[team]
        return self._avg_alpha, self._avg_beta

    def predict_lambda_mu(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        if neutral:
            gamma = 1.0
        else:
            gamma = self.gamma_ if (self.host_team is None or home == self.host_team) else 1.0
        a_home = self.alpha_.get(home) or self._fallback(home)[0]
        b_away = self.beta_.get(away)  or self._fallback(away)[1]
        a_away = self.alpha_.get(away) or self._fallback(away)[0]
        b_home = self.beta_.get(home)  or self._fallback(home)[1]
        lam = a_home * b_away * gamma
        mu  = a_away * b_home
        return lam, mu

    def match_outcome_probs(self, home: str, away: str, neutral: bool = False, max_goals: int = 10) -> tuple[float, float, float]:
        """Returns calibrated (P_home_win, P_draw, P_away_win)."""
        ph, pd_, pa = self._raw_outcome_probs(home, away, neutral=neutral, max_goals=max_goals)
        if self.temperature_ == 1.0:
            return ph, pd_, pa
        p = np.array([ph, pd_, pa], dtype=float)
        logits = np.log(np.clip(p, 1e-10, 1.0)) / self.temperature_
        logits -= logits.max()
        exp_l = np.exp(logits)
        p_cal = exp_l / exp_l.sum()
        return float(p_cal[0]), float(p_cal[1]), float(p_cal[2])
