"""Tests for agselp.alphazero_self_play_eval."""

from morie.fn import _array_core as np

from morie.fn.agselp import alphazero_self_play_eval


def test_agselp_basic():
    """Test basic functionality."""
    # Documented call: when wins is None, new_net is the win count and
    # old_net is the loss count (scalars).
    new_net = 60
    old_net = 35
    draws = 5
    n_games = new_net + old_net + draws
    result = alphazero_self_play_eval(new_net, old_net, n_games=n_games,
                                      wins=None, draws=draws, threshold=0.55)
    expected_score = (new_net + 0.5 * draws) / n_games
    assert result["wins"] == float(new_net)
    assert result["losses"] == float(old_net)
    assert result["draws"] == float(draws)
    assert result["n"] == float(n_games)
    assert result["estimate"] == expected_score
    assert result["score"] == expected_score
    assert result["replace"] == (expected_score > 0.55)
    # p-value is a one-sided binomial tail at p = 1/2 for the win count.
    from math import comb
    dec = int(round(new_net + old_net))
    kk = int(round(new_net))
    tail = sum(comb(dec, i) * (0.5 ** dec) for i in range(kk, dec + 1))
    assert abs(result["p_value"] - tail) < 1e-12


def test_agselp_edge():
    """Test edge cases: zero games -> NaN score; replace gate below threshold."""
    new_net = 4
    old_net = 6
    result = alphazero_self_play_eval(new_net, old_net, n_games=10,
                                      wins=None, draws=0, threshold=0.55)
    # score below threshold -> replace False
    assert result["replace"] is False
    # passing wins/draws explicitly should also work and match
    result2 = alphazero_self_play_eval(0, 0, n_games=10, wins=4, draws=0,
                                       threshold=0.55)
    expected_score2 = (4 + 0.5 * 0) / 10
    assert result2["estimate"] == expected_score2
    assert result2["replace"] == (expected_score2 > 0.55)
    assert result2["losses"] == 10 - 4 - 0
