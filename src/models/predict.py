import math

from .features import build_match_row
from .poisson_regressor import PoissonRegressor
from .scoring import kicktipp_points


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * rate**k / math.factorial(k)


def predict_score(
    model: PoissonRegressor,
    home_team: str,
    away_team: str,
    team_index: dict[str, int],
    max_goals: int = 10,
) -> tuple[int, int]:
    """Predict a final score by maximizing expected Kicktipp points.

    Rather than picking the single most probable scoreline (which tends
    to collapse to a flat 1:1 for evenly matched teams) or the most
    probable tendency (which structurally underpredicts draws, since one
    side is almost always at least slightly favored), this weighs every
    candidate scoreline by the Kicktipp points it would earn against
    every possible actual outcome, scaled by how likely that outcome is.
    The candidate with the highest expected payoff wins - this naturally
    favors a draw when it is genuinely the better bet, without hardcoding
    a rule for it.

    Args:
        model: A fitted PoissonRegressor.
        home_team: Name of the home team.
        away_team: Name of the away team.
        team_index: Mapping from team name to feature column offset.
        max_goals: Highest goal count considered per team when building
            the scoreline grid.

    Returns:
        The predicted (home_goals, away_goals).
    """
    home_row, away_row = build_match_row(home_team, away_team, team_index)
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
