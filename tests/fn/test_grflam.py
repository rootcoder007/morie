"""Tests for grflam.geron_flamingo_cross_modal_attn."""

from morie.fn import _array_core as np

from morie.fn.grflam import geron_flamingo_cross_modal_attn

import math


def test_grflam_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    T, Tv, d_model = 3, 4, 2
    h = rng.normal(0, 1, (T, d_model))
    visual_features = rng.normal(0, 1, (Tv, d_model))
    alpha = 0.05
    WQ = rng.normal(0, 1, (d_model, d_model))
    WK = rng.normal(0, 1, (d_model, d_model))
    WV = rng.normal(0, 1, (d_model, d_model))
    weights = {"WQ": WQ, "WK": WK, "WV": WV}
    result = geron_flamingo_cross_modal_attn(h, visual_features, alpha, weights)
    assert isinstance(result, dict)
    assert "h_new" in result
    assert "gate" in result
    assert "is_identity" in result
    assert "delta_norm" in result
    assert "attention_weights" in result
    # h_new has the same number of rows (tokens) as h
    assert len(result["h_new"]) == T
    # gate is tanh(alpha) by construction
    assert math.isclose(result["gate"], math.tanh(alpha), rel_tol=1e-10)
    assert math.isfinite(result["delta_norm"])


def test_grflam_edge():
    """Test edge cases."""
    # Identity at alpha = 0: hidden states must come back unchanged
    # and the layer must flag itself as the identity mapping.
    I = [[1.0, 0.0], [0.0, 1.0]]
    vis = [[5.0, 5.0], [-5.0, 3.0]]
    h = [[1.0, 0.0]]
    weights = {"WQ": I, "WK": I, "WV": I}
    result = geron_flamingo_cross_modal_attn(h, vis, 0.0, weights)
    assert isinstance(result, dict)
    assert result["is_identity"] is True
    assert result["gate"] == 0.0
    assert result["h_new"] == [[1.0, 0.0]]
    assert result["delta_norm"] == 0.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grflam as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
