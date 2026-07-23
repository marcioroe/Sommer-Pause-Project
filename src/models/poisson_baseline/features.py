import numpy as np
import pandas as pd


def build_design_matrix(
    matches: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the Poisson regression design matrix from match results.

    Each match becomes two rows: one for the home team's goals, one for
    the away team's goals, with one-hot columns for attack/defense per
    team and a home-advantage indicator.

    Args:
        matches: Cleaned match data with columns 'hometeam', 'awayteam',
            'fthg' (home goals) and 'ftag' (away goals).

    Returns:
        A tuple (X, y, team_index): the feature matrix, the target goal
        counts, and a mapping from team name to its column offset.
    """
    teams = sorted(set(matches["hometeam"]) | set(matches["awayteam"]))
    team_index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    n_features = 1 + 2 * n_teams

    rows = []
    targets = []

    for _, match in matches.iterrows():
        home_idx = team_index[match["hometeam"]]
        away_idx = team_index[match["awayteam"]]

        home_row = np.zeros(n_features)
        home_row[0] = 1
        home_row[1 + home_idx] = 1
        home_row[1 + n_teams + away_idx] = 1
        rows.append(home_row)
        targets.append(match["fthg"])

        away_row = np.zeros(n_features)
        away_row[1 + away_idx] = 1
        away_row[1 + n_teams + home_idx] = 1
        rows.append(away_row)
        targets.append(match["ftag"])

    X = np.array(rows)
    y = np.array(targets, dtype=float)
    return X, y, team_index


def build_match_row(
    home_team: str,
    away_team: str,
    team_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature rows for a single upcoming match.

    Teams not present in team_index (e.g. newly promoted/relegated teams
    with no training history) are treated as average: their attack/defense
    columns are simply left at zero.

    Args:
        home_team: Name of the home team.
        away_team: Name of the away team.
        team_index: Mapping from team name to feature column offset.

    Returns:
        A tuple (home_row, away_row) of feature rows.
    """
    n_teams = len(team_index)
    n_features = 1 + 2 * n_teams

    home_row = np.zeros(n_features)
    home_row[0] = 1
    if home_team in team_index:
        home_row[1 + team_index[home_team]] = 1
    if away_team in team_index:
        home_row[1 + n_teams + team_index[away_team]] = 1

    away_row = np.zeros(n_features)
    if away_team in team_index:
        away_row[1 + team_index[away_team]] = 1
    if home_team in team_index:
        away_row[1 + n_teams + team_index[home_team]] = 1

    return home_row, away_row
