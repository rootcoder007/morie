"""Tests for kmlora.kamath_lora_weight_update."""

import math

from morie.fn import _array_core as np

from morie.fn.kmlora import kamath_lora_weight_update


def test_kmlora_basic():
    """Test basic functionality with random matrices of valid shapes."""
    rng = np.random.default_rng(42)
    d, k, r = 20, 15, 3
    W0 = rng.normal(0, 1, (d, k))
    A = rng.normal(0, 1, (r, k))
    B = rng.normal(0, 1, (d, r))
    alpha = 0.05
    x = rng.normal(0, 1, k)
    result = kamath_lora_weight_update(W0, A, B, alpha, r, x)
    assert isinstance(result, dict)
    for key in ("h", "base", "delta", "estimate", "scaling",
                "rank", "alpha", "n_trainable", "n_frozen",
                "n", "method"):
        assert key in result
    assert len(result["h"]) == d
    assert len(result["base"]) == d
    assert len(result["delta"]) == d
    assert result["n"] == d
    assert result["rank"] == r
    assert math.isclose(result["scaling"], alpha / r)
    assert math.isclose(result["alpha"], alpha)
    assert result["n_trainable"] == A.size + B.size
    assert result["n_frozen"] == W0.size
    assert all(
        math.isclose(h_i, b_i + dl_i)
        for h_i, b_i, dl_i in zip(
            result["h"], result["base"], result["delta"])
    )
    assert math.isclose(result["estimate"], result["h"][0])
    assert math.isfinite(result["estimate"])


def test_kmlora_edge():
    """Edge case: small valid input matching the docstring example."""
    W0 = [[1.0, 0.0], [0.0, 1.0]]
    A = [[1.0, 0.0]]
    B = [[0.0], [2.0]]
    alpha = 4.0
    r = 1
    x = [3.0, 5.0]
    result = kamath_lora_weight_update(W0, A, B, alpha, r, x)
    assert isinstance(result, dict)
    assert result["h"] == [3.0, 29.0]
    assert result["base"] == [3.0, 5.0]
    assert result["delta"] == [0.0, 24.0]
    assert result["n_trainable"] == 4
    assert result["n_frozen"] == 4
    assert result["scaling"] == 4.0
    assert result["rank"] == 1
    assert result["n"] == 2


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmlora as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
