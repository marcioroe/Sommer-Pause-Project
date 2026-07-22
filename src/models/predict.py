import math

from .features import build_match_row
from .poisson_regressor import PoissonRegressor


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * rate**k / math.factorial(k)


def predict_score(
    model: PoissonRegressor,
    home_team: str,
    away_team: str,
    team_index: dict[str, int],
    max_goals: int = 10,
) -> tuple[int, int]:
    """Predict the most likely final score for a match.

    Args:
        model: A fitted PoissonRegressor.
        home_team: Name of the home team.
        away_team: Name of the away team.
        team_index: Mapping from team name to feature column offset.
        max_goals: Highest goal count considered per team when searching
            for the most likely scoreline.

    Returns:
        The (home_goals, away_goals) combination with the highest joint
        probability under independent Poisson distributions.
    """
    home_row, away_row = build_match_row(home_team, away_team, team_index)
    lambda_home = model.predict(home_row.reshape(1, -1))[0]
    lambda_away = model.predict(away_row.reshape(1, -1))[0]

    best_score = (0, 0)
    best_prob = -1.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = _poisson_pmf(i, lambda_home) * _poisson_pmf(j, lambda_away)
            if prob > best_prob:
                best_prob = prob
                best_score = (i, j)

    return best_score
