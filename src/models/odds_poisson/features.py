import numpy as np
import pandas as pd


def implied_probabilities(avg_h: float, avg_d: float, avg_a: float) -> tuple[float, float, float]:
    """Convert average market odds to normalized outcome probabilities.

    Raw 1/odds values sum to slightly more than 1 (the bookmakers'
    margin/overround) - dividing by their sum removes that margin so the
    three probabilities add up to exactly 1.

    Args:
        avg_h: Average market odds for a home win.
        avg_d: Average market odds for a draw.
        avg_a: Average market odds for an away win.

    Returns:
        A tuple (p_home, p_draw, p_away) summing to 1.
    """
    raw = np.array([1 / avg_h, 1 / avg_d, 1 / avg_a])
    normalized = raw / raw.sum()
    return tuple(normalized)


def build_design_matrix(
    matches: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the Poisson regression design matrix from odds alone.

    No team one-hot columns here, unlike the other models - team
    identity and market odds turned out to be too collinear to combine
    usefully (odds already price in team strength), and odds-only
    outperformed the combined version on the 25/26 holdout. Unlike the
    form-based models, odds are set before the match by definition, so
    there is no leakage concern and no rolling window needed - each
    match's own pre-match odds are used directly.

    Args:
        matches: Cleaned match data with columns 'fthg', 'ftag', 'avgh',
            'avgd', 'avga' (average market odds for home win / draw /
            away win).

    Returns:
        A tuple (X, y, team_index): the feature matrix, the target goal
        counts, and an empty team_index (kept for interface parity with
        the other models - this model doesn't use team identity).
    """
    rows = []
    targets = []

    for _, match in matches.iterrows():
        p_home, p_draw, p_away = implied_probabilities(
            match["avgh"], match["avgd"], match["avga"]
        )

        # [is_home, own_win_prob, opponent_win_prob, draw_prob]
        rows.append([1.0, p_home, p_away, p_draw])
        targets.append(match["fthg"])

        rows.append([0.0, p_away, p_home, p_draw])
        targets.append(match["ftag"])

    X = np.array(rows)
    y = np.array(targets, dtype=float)
    return X, y, {}


def build_match_row(
    avg_h: float,
    avg_d: float,
    avg_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature rows for a single match, given its market odds.

    Args:
        avg_h: Average market odds for a home win.
        avg_d: Average market odds for a draw.
        avg_a: Average market odds for an away win.

    Returns:
        A tuple (home_row, away_row) of feature rows.
    """
    p_home, p_draw, p_away = implied_probabilities(avg_h, avg_d, avg_a)
    home_row = np.array([1.0, p_home, p_away, p_draw])
    away_row = np.array([0.0, p_away, p_home, p_draw])
    return home_row, away_row
