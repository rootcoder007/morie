"""Tests for alfqud.alphadev_quicksort_disc."""

from morie.fn import _array_core as np

from morie.fn.alfqud import alphadev_quicksort_disc


def test_alfqud_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    target = [list(rng.integers(0, 100, 5).tolist()) for _ in range(4)]
    reward_fn = lambda prog, inputs, targets, n_reg: 0
    result = alphadev_quicksort_disc(target, reward_fn=reward_fn,
                                     n_reg=2, max_len=2,
                                     search="bfs", seed=0)
    assert isinstance(result, dict)
    assert "program" in result
    assert "score" in result
    assert "correct" in result
    assert "max_correct" in result
    assert "solved" in result
    assert "outputs" in result
    assert "targets" in result
    assert "length" in result
    assert "nodes" in result
    assert "n_actions" in result
    assert "search" in result
    assert result["n_mem"] == 5
    assert result["n_reg"] == 2
    assert result["max_len"] == 2
    assert result["latency_weight"] == 0.0
    assert result["search"] == "bfs"
    full = sum(len(t) for t in target)
    assert result["max_correct"] == full
    expected_targets = [sorted(list(x)) for x in target]
    assert result["targets"] == expected_targets


def test_alfqud_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    target = [list(rng.integers(0, 100, 3).tolist()) for _ in range(2)]
    reward_fn = lambda prog, inputs, targets, n_reg: 0
    result = alphadev_quicksort_disc(target, reward_fn=reward_fn,
                                     n_reg=2, max_len=1,
                                     search="bfs", seed=0)
    assert isinstance(result, dict)
    assert result["length"] <= 1
    assert result["max_len"] == 1
