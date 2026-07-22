def kicktipp_points(predicted: tuple[int, int], actual: tuple[int, int]) -> int:
    """Score a prediction using standard Kicktipp rules.

    4 points for the exact result, 3 for the correct goal difference (this
    also covers a correctly tipped draw with the wrong exact score, since a
    draw always has a goal difference of 0), 1 for correctly tipping only
    the winner/draw, 0 otherwise.

    Args:
        predicted: The predicted (home_goals, away_goals).
        actual: The actual (home_goals, away_goals).

    Returns:
        Points awarded for this prediction.
    """
    if predicted == actual:
        return 4

    predicted_diff = predicted[0] - predicted[1]
    actual_diff = actual[0] - actual[1]
    if predicted_diff == actual_diff:
        return 3

    predicted_tendency = (predicted_diff > 0) - (predicted_diff < 0)
    actual_tendency = (actual_diff > 0) - (actual_diff < 0)
    if predicted_tendency == actual_tendency:
        return 1

    return 0
