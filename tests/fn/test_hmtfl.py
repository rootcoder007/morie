"""Tests for hmtfl.geron_transfer_learning."""

from morie.fn import _array_core as np

from morie.fn.hmtfl import geron_transfer_learning


def test_hmtfl_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # Pretrained model: 2 layers with chained shapes (3,4) and (4,1)
    W0 = rng.normal(0, 0.5, (3, 4))
    W1 = rng.normal(0, 0.5, (4, 1))
    pretrained_model = [W0, W1]

    n = 40
    X = rng.normal(0, 1, (n, 3))
    y = rng.normal(0, 1, n)

    result = geron_transfer_learning(pretrained_model, X, y, n_frozen=1, epochs=50)

    assert isinstance(result, dict)
    # The frozen layer must come back bit-identical
    assert bool(np.array_equal(np.asarray(result["weights"][0]), np.asarray(W0)))
    # The trainable layer should have moved
    assert bool(not np.array_equal(np.asarray(result["weights"][1]), np.asarray(W1)))
    # Loss must drop over fine-tuning
    assert result["final_loss"] < result["initial_loss"]
    # W1 has 4 entries, W0 has 12; total 16
    assert int(result["trainable_params"]) == 4
    assert int(result["total_params"]) == 16
    assert int(result["n"]) == n
    assert int(result["frozen"]) == 1
    assert len(result["loss_curve"]) > 0


def test_hmtfl_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # Pretrained model: 3 layers, only the last one is trainable
    W0 = rng.normal(0, 0.5, (3, 4))
    W1 = rng.normal(0, 0.5, (4, 4))
    W2 = rng.normal(0, 0.5, (4, 1))
    pretrained_model = [W0, W1, W2]

    n = 40
    X = rng.normal(0, 1, (n, 3))
    y = rng.normal(0, 1, n)

    result = geron_transfer_learning(pretrained_model, X, y, n_frozen=2, epochs=20)

    assert isinstance(result, dict)
    # The first two frozen layers must come back bit-identical
    assert bool(np.array_equal(np.asarray(result["weights"][0]), np.asarray(W0)))
    assert bool(np.array_equal(np.asarray(result["weights"][1]), np.asarray(W1)))
    # The single trainable layer should have moved
    assert bool(not np.array_equal(np.asarray(result["weights"][2]), np.asarray(W2)))
    assert int(result["frozen"]) == 2
    assert len(result["weights"]) == 3
    assert len(result["loss_curve"]) > 0
    assert result["final_loss"] < result["initial_loss"]


# --- appended: the module's own worked example as a gate -----------
import doctest as _doctest

import morie.fn.hmtfl as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
