import math

from .features import build_match_row
from ..poisson_regressor import PoissonRegressor
from ..scoring import kicktipp_points


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * rate**k / math.factorial(k)


def predict_score(
    model: PoissonRegressor,
    home_team: str,
    away_team: str,
    team_index: dict[str, int],
    current_form: dict[str, dict[str, float]],
    max_goals: int = 10,
) -> tuple[int, int]:
    """Predict a final score by maximizing expected Kicktipp points.

    Same decision rule as the baseline model: weighs every candidate
    scoreline by the Kicktipp points it would earn against every
    possible actual outcome, scaled by how likely that outcome is, and
    picks the candidate with the highest expected payoff.

    Args:
        model: A fitted PoissonRegressor.
        home_team: Name of the home team.
        away_team: Name of the away team.
        team_index: Mapping from team name to feature column offset.
        current_form: Output of features.compute_current_form(), giving
            each team's rolling form heading into this match.
        max_goals: Highest goal count considered per team when building
            the scoreline grid.

    Returns:
        The predicted (home_goals, away_goals).
    """
    home_row, away_row = build_match_row(home_team, away_team, team_index, current_form)
    lambda_home = model.predict(home_row.reshape(1, -1))[0]
    lambda_away = model.predict(away_row.reshape(1, -1))[0]

    probs = {
        (i, j): _poisson_pmf(i, lambda_home) * _poisson_pmf(j, lambda_away)
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    }

    expected_points = {
        candidate: sum(
            prob * kicktipp_points(candidate, actual) for actual, prob in probs.items()
        )
        for candidate in probs
    }

    return max(expected_points, key=expected_points.get)
