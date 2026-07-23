import numpy as np
import pandas as pd

FORM_FEATURES = ["form_shots_for", "form_shots_against"]

# Shots-on-target counts (roughly 2-15 per match) are on a much larger
# scale than the 0/1 one-hot columns and the goals-based form we tried
# before - without rescaling, gradient descent at the learning rate that
# works for the rest of the matrix overshoots and diverges. Dividing by
# a fixed constant keeps everything in a comparable range without
# needing to fit/store separate normalization stats for train vs.
# prediction time.
SHOTS_SCALE = 10.0


def _team_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape one row per match into one row per team per match."""
    cols = ["date", "hometeam", "awayteam", "fthg", "ftag", "hst", "ast"]
    home = matches[cols].rename(
        columns={
            "hometeam": "team",
            "awayteam": "opponent",
            "fthg": "goals_scored",
            "ftag": "goals_conceded",
            "hst": "shots_for",
            "ast": "shots_against",
        }
    )
    away = matches[cols].rename(
        columns={
            "awayteam": "team",
            "hometeam": "opponent",
            "ftag": "goals_scored",
            "fthg": "goals_conceded",
            "ast": "shots_for",
            "hst": "shots_against",
        }
    )
    long_df = pd.concat([home, away], ignore_index=True)
    long_df["shots_for"] = long_df["shots_for"].fillna(long_df["shots_for"].mean())
    long_df["shots_against"] = long_df["shots_against"].fillna(long_df["shots_against"].mean())
    return long_df.sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def _rolling_form_columns(long_df: pd.DataFrame, window: int, shift: bool) -> pd.DataFrame:
    """Attach rolling-average shots-on-target form columns.

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
    for raw_col, form_col in [("shots_for", "form_shots_for"), ("shots_against", "form_shots_against")]:
        series = long_df.groupby("team")[raw_col]
        if shift:
            long_df[form_col] = series.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
        else:
            long_df[form_col] = series.transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
        long_df[form_col] = long_df[form_col].fillna(long_df[raw_col].mean())
        long_df[form_col] = long_df[form_col] / SHOTS_SCALE
    return long_df


def build_design_matrix(
    matches: pd.DataFrame,
    form_window: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build the Poisson regression design matrix, extended with a
    shots-on-target-based form signal (a lower-noise proxy for team
    quality than goals, similar in spirit to expected goals).

    Args:
        matches: Cleaned match data with columns 'date', 'hometeam',
            'awayteam', 'fthg', 'ftag', 'hst' (home shots on target) and
            'ast' (away shots on target).
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
        home_row[1 + 2 * n_teams] = home_form["form_shots_for"]
        home_row[1 + 2 * n_teams + 1] = away_form["form_shots_against"]
        rows.append(home_row)
        targets.append(match["fthg"])

        away_row = np.zeros(n_features)
        away_row[1 + away_idx] = 1
        away_row[1 + n_teams + home_idx] = 1
        away_row[1 + 2 * n_teams] = away_form["form_shots_for"]
        away_row[1 + 2 * n_teams + 1] = home_form["form_shots_against"]
        rows.append(away_row)
        targets.append(match["ftag"])

    X = np.array(rows)
    y = np.array(targets, dtype=float)
    return X, y, team_index


def compute_current_form(
    matches: pd.DataFrame, form_window: int = 10
) -> dict[str, dict[str, float]]:
    """Compute each team's current shots-on-target form.

    Uses each team's latest match too (no shifting) - we want form
    heading into a match that isn't in `matches` yet, so there is
    nothing to leak.

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
    home_row[1 + 2 * n_teams] = home_form["form_shots_for"]
    home_row[1 + 2 * n_teams + 1] = away_form["form_shots_against"]

    away_row = np.zeros(n_features)
    if away_team in team_index:
        away_row[1 + team_index[away_team]] = 1
    if home_team in team_index:
        away_row[1 + n_teams + team_index[home_team]] = 1
    away_row[1 + 2 * n_teams] = away_form["form_shots_for"]
    away_row[1 + 2 * n_teams + 1] = home_form["form_shots_against"]

    return home_row, away_row
