"""Tests for agstkb.alphazero_stockfish_baseline."""

import math

from morie.fn import _array_core as np

from morie.fn.agstkb import alphazero_stockfish_baseline


def test_agstkb_basic():
    """Test basic functionality."""
    # games: one row per rung, each row is (wins, draws, losses)
    games = [
        [10, 2, 3],   # rung 0
        [5, 0, 5],    # rung 1
        [2, 4, 6],    # rung 2
        [0, 1, 9],    # rung 3
        [8, 2, 2],    # rung 4
    ]
    ladder = [2400.0, 2500.0, 2600.0, 2700.0, 2800.0]

    result = alphazero_stockfish_baseline(games, ladder)

    # RichResult exposes payload via attributes
    assert hasattr(result, "estimate")
    assert hasattr(result, "per_rung")
    assert hasattr(result, "scores")
    assert hasattr(result, "n_games")

    # Independent computation of the expected estimate and per_rung
    c_elo = 1.0 / 400.0
    num = 0.0
    den = 0.0
    expected_per = []
    for i, (w, d, l) in enumerate(games):
        n = w + d + l
        s = (w + 0.5 * d) / n
        if 0.0 < s < 1.0:
            odds = math.log(s / (1.0 - s))
            r = ladder[i] + odds / c_elo  # base="e" default
        elif s <= 0.0:
            r = float("-inf")
        else:
            r = float("inf")
        expected_per.append(r)
        if r == r and abs(r) != float("inf"):
            num += n * r
            den += n
    expected_estimate = num / den

    # scores and n_games
    expected_scores = [
        (w + 0.5 * d) / (w + d + l) for (w, d, l) in games
    ]
    expected_ns = [w + d + l for (w, d, l) in games]

    assert len(result.per_rung) == len(games)
    assert len(result.scores) == len(games)
    assert len(result.n_games) == len(games)

    for i in range(len(games)):
        assert math.isclose(result.per_rung[i], expected_per[i], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(result.scores[i], expected_scores[i], rel_tol=1e-9, abs_tol=1e-9)
        assert result.n_games[i] == expected_ns[i]

    assert math.isclose(result.estimate, expected_estimate, rel_tol=1e-9, abs_tol=1e-9)

    # Also check wins/draws/losses tallies
    assert result.wins == sum(row[0] for row in games)
    assert result.draws == sum(row[1] for row in games)
    assert result.losses == sum(row[2] for row in games)


def test_agstkb_edge():
    """Test edge cases."""
    # Two rungs, one all-wins, one all-losses -> +-inf per rung, estimate based on finite rungs only
    games = [
        [10, 0, 0],
        [0, 0, 10],
        [5, 0, 5],  # 50% against this rung -> finite rating
    ]
    ladder = [2600.0, 2700.0, 2650.0]

    result = alphazero_stockfish_baseline(games, ladder)

    assert result.scores[0] == 1.0
    assert result.scores[1] == 0.0
    assert math.isclose(result.scores[2], 0.5, rel_tol=1e-9)

    # Only the third rung has 0 < s < 1, so estimate == its per_rung value
    c_elo = 1.0 / 400.0
    s = 0.5
    odds = math.log(s / (1.0 - s))  # 0.0
    expected = ladder[2] + odds / c_elo
    assert math.isclose(result.estimate, expected, rel_tol=1e-9, abs_tol=1e-9)

    # The +-inf rungs should be inf / -inf
    assert result.per_rung[0] == float("inf")
    assert result.per_rung[1] == float("-inf")
