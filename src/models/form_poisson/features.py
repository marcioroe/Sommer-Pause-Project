import numpy as np
import pandas as pd

FORM_FEATURES = ["form_points", "form_goals_scored", "form_goals_conceded"]


def _team_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape one row per match into one row per team per match."""
    home = matches[["date", "hometeam", "awayteam", "fthg", "ftag"]].rename(
        columns={
            "hometeam": "team",
            "awayteam": "opponent",
            "fthg": "goals_scored",
            "ftag": "goals_conceded",
        }
    )
    away = matches[["date", "hometeam", "awayteam", "fthg", "ftag"]].rename(
        columns={
            "awayteam": "team",
            "hometeam": "opponent",
            "ftag": "goals_scored",
            "fthg": "goals_conceded",
        }
    )
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["points"] = np.select(
        [
            long_df["goals_scored"] > long_df["goals_conceded"],
            long_df["goals_scored"] == long_df["goals_conceded"],
        ],
        [3, 1],
        default=0,
    )
    return long_df.sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def _rolling_form_columns(long_df: pd.DataFrame, window: int, shift: bool) -> pd.DataFrame:
    """Attach rolling-average form columns to a long-format frame.

    Args:
        long_df: Output of _team_long_format, sorted by team then date.
        window: Number of past matches to average over.
        shift: If True, exclude each row's own match from its rolling
            window - needed for historical training rows, so a match's
            result never leaks into its own feature. If False, include
            it - used when computing a team's form heading into a future
            match that isn't in the data yet.
    """
    long_df = long_df.copy()
    for col in ["points", "goals_scored", "goals_conceded"]:
        series = long_df.groupby("team")[col]
        if shift:
            long_df[f"form_{col}"] = series.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
        else:
            long_df[f"form_{col}"] = series.transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
        long_df[f"form_{col}"] = long_df[f"form_{col}"].fillna(long_df[col].mean())
    return long_df


def build_design_matrix(
    matches: pd.DataFrame,
    form_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the Poisson regression design matrix, extended with form.

    Same team attack/defense/home-advantage one-hot structure as the
    baseline model, plus each side's rolling form (points, goals scored,
    goals conceded over their last `form_window` matches, computed
    without leaking the current match's own result).

    Args:
        matches: Cleaned match data with columns 'date', 'hometeam',
            'awayteam', 'fthg' (home goals) and 'ftag' (away goals).
        form_window: Number of past matches each team's form is
            averaged over.

    Returns:
        A tuple (X, y, team_index): the feature matrix, the target goal
        counts, and a mapping from team name to its column offset.
    """
    teams = sorted(set(matches["hometeam"]) | set(matches["awayteam"]))
    team_index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    long_df = _team_long_format(matches)
    long_df = _rolling_form_columns(long_df, form_window, shift=True)
    form_lookup = long_df.set_index(["team", "date"])[FORM_FEATURES]

    n_features = 1 + 2 * n_teams + len(FORM_FEATURES)
    rows = []
    targets = []

    for _, match in matches.iterrows():
        home_idx = team_index[match["hometeam"]]
        away_idx = team_index[match["awayteam"]]
        home_form = form_lookup.loc[(match["hometeam"], match["date"])]
        away_form = form_lookup.loc[(match["awayteam"], match["date"])]

        home_row = np.zeros(n_features)
        home_row[0] = 1
        home_row[1 + home_idx] = 1
        home_row[1 + n_teams + away_idx] = 1
        home_row[1 + 2 * n_teams] = home_form["form_points"]
        home_row[1 + 2 * n_teams + 1] = home_form["form_goals_scored"]
        home_row[1 + 2 * n_teams + 2] = away_form["form_goals_conceded"]
        rows.append(home_row)
        targets.append(match["fthg"])

        away_row = np.zeros(n_features)
        away_row[1 + away_idx] = 1
        away_row[1 + n_teams + home_idx] = 1
        away_row[1 + 2 * n_teams] = away_form["form_points"]
        away_row[1 + 2 * n_teams + 1] = away_form["form_goals_scored"]
        away_row[1 + 2 * n_teams + 2] = home_form["form_goals_conceded"]
        rows.append(away_row)
        targets.append(match["ftag"])

    X = np.array(rows)
    y = np.array(targets, dtype=float)
    return X, y, team_index


def compute_current_form(
    matches: pd.DataFrame, form_window: int = 5
) -> dict[str, dict[str, float]]:
    """Compute each team's rolling form based on their most recent matches.

    Unlike the training-time computation, this uses each team's latest
    match too (no shifting) - we want form heading into a match that
    isn't in `matches` yet, so there is nothing to leak.

    Args:
        matches: Historical match data up to (not including) the match
            to be predicted.
        form_window: Number of past matches each team's form is
            averaged over.

    Returns:
        A mapping from team name to its current form stats. Includes a
        special "__average__" entry (the league-wide average) to use as
        a fallback for teams with no history (e.g. newly promoted teams).
    """
    long_df = _team_long_format(matches)
    long_df = _rolling_form_columns(long_df, form_window, shift=False)
    latest = long_df.groupby("team").tail(1)

    form = {row["team"]: {col: row[col] for col in FORM_FEATURES} for _, row in latest.iterrows()}
    form["__average__"] = {col: long_df[col].mean() for col in FORM_FEATURES}
    return form


def build_match_row(
    home_team: str,
    away_team: str,
    team_index: dict[str, int],
    current_form: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature rows for a single upcoming match.

    Teams missing from team_index/current_form (e.g. newly promoted or
    relegated teams with no history) are treated as average: their
    one-hot attack/defense columns are left at zero, and their form
    values fall back to the league-wide average.

    Args:
        home_team: Name of the home team.
        away_team: Name of the away team.
        team_index: Mapping from team name to feature column offset.
        current_form: Output of compute_current_form().

    Returns:
        A tuple (home_row, away_row) of feature rows.
    """
    n_teams = len(team_index)
    n_features = 1 + 2 * n_teams + len(FORM_FEATURES)
    league_avg = current_form.get("__average__", {})
    home_form = current_form.get(home_team, league_avg)
    away_form = current_form.get(away_team, league_avg)

    home_row = np.zeros(n_features)
    home_row[0] = 1
    if home_team in team_index:
        home_row[1 + team_index[home_team]] = 1
    if away_team in team_index:
        home_row[1 + n_teams + team_index[away_team]] = 1
    home_row[1 + 2 * n_teams] = home_form["form_points"]
    home_row[1 + 2 * n_teams + 1] = home_form["form_goals_scored"]
    home_row[1 + 2 * n_teams + 2] = away_form["form_goals_conceded"]

    away_row = np.zeros(n_features)
    if away_team in team_index:
        away_row[1 + team_index[away_team]] = 1
    if home_team in team_index:
        away_row[1 + n_teams + team_index[home_team]] = 1
    away_row[1 + 2 * n_teams] = away_form["form_points"]
    away_row[1 + 2 * n_teams + 1] = away_form["form_goals_scored"]
    away_row[1 + 2 * n_teams + 2] = home_form["form_goals_conceded"]

    return home_row, away_row
