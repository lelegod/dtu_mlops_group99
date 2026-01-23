import numpy as np


def map_score(score: str | int) -> int | float:
    """Convert tennis score to numeric value.

    Maps standard tennis scores ("0", "15", "30", "40", "AD") to integers 0-4.
    For tiebreak scores, returns the integer value directly.

    Args:
        score: Tennis score as string or integer.

    Returns:
        Numeric score value (0-4 for regular games, or higher int for tiebreaks).
        Returns np.nan if invalid.
    """
    regular_score_map = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, 0: 0, 15: 1, 30: 2, 40: 3}
    if score in regular_score_map:
        return regular_score_map[score]
    try:
        return int(score)
    except Exception:
        return np.nan


def input_preprocessing(raw_point: dict) -> np.ndarray:
    """Transform raw point data into model features.

    Converts player-centric input (P1/P2) to server/receiver perspective.

    Args:
        raw_point: Dictionary with keys: SetNo, GameNo, PointNumber, PointServer,
            ServeIndicator, P1GamesWon, P1SetsWon, P1Score, P1PointsWon, P1Momentum,
            P2GamesWon, P2SetsWon, P2Score, P2PointsWon, P2Momentum.

    Returns:
        Feature array of shape (1, 20) ready for model prediction.
    """
    point_server = raw_point["PointServer"]
    is_p1_serving = point_server == 1

    p1_score = map_score(raw_point["P1Score"])
    p2_score = map_score(raw_point["P2Score"])

    p1_momentum = raw_point["P1Momentum"]
    p2_momentum = raw_point["P2Momentum"]

    server_games_won = raw_point["P1GamesWon"] if is_p1_serving else raw_point["P2GamesWon"]
    receiver_games_won = raw_point["P2GamesWon"] if is_p1_serving else raw_point["P1GamesWon"]
    server_score = p1_score if is_p1_serving else p2_score
    receiver_score = p2_score if is_p1_serving else p1_score
    server_points_won = raw_point["P1PointsWon"] if is_p1_serving else raw_point["P2PointsWon"]
    receiver_points_won = raw_point["P2PointsWon"] if is_p1_serving else raw_point["P1PointsWon"]
    server_momentum = p1_momentum if is_p1_serving else p2_momentum
    receiver_momentum = p2_momentum if is_p1_serving else p1_momentum
    server_sets_won = raw_point["P1SetsWon"] if is_p1_serving else raw_point["P2SetsWon"]
    receiver_sets_won = raw_point["P2SetsWon"] if is_p1_serving else raw_point["P1SetsWon"]

    game_diff = server_games_won - receiver_games_won
    set_diff = server_sets_won - receiver_sets_won
    point_diff = server_points_won - receiver_points_won
    momentum_diff = server_momentum - receiver_momentum

    is_break_point = 1 if (receiver_score >= 3 and receiver_score > server_score) else 0
    set_pressure = 6 - max(server_games_won, receiver_games_won)
    is_second_serve = 1 if raw_point["ServeIndicator"] == 2 else 0

    features = np.array(
        [
            raw_point["SetNo"],
            raw_point["GameNo"],
            raw_point["PointNumber"],
            server_games_won,
            receiver_games_won,
            server_score,
            receiver_score,
            server_points_won,
            receiver_points_won,
            server_momentum,
            receiver_momentum,
            server_sets_won,
            receiver_sets_won,
            game_diff,
            set_diff,
            point_diff,
            momentum_diff,
            is_break_point,
            set_pressure,
            is_second_serve,
        ],
        dtype=np.float32,
    ).reshape(1, -1)

    return features
