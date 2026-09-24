"""Tests for hmrvat.geron_rnn_visual_attention."""

import math

from morie.fn import _array_core as np

from morie.fn.hmrvat import geron_rnn_visual_attention


def test_hmrvat_basic():
    """Test basic functionality with a 2D feature matrix (N, D)."""
    rng = np.random.default_rng(42)
    N, D, d_h, k = 10, 4, 5, 3
    features = rng.normal(0, 1, (N, D))
    h = rng.normal(0, 1, d_h)
    W = rng.normal(0, 1, (k, D))
    U = rng.normal(0, 1, (k, d_h))
    v = rng.normal(0, 1, k)
    result = geron_rnn_visual_attention(features, h, W, U, v)
    assert isinstance(result, dict)
    for key in ("context", "alpha", "scores", "entropy", "estimate", "n", "method"):
        assert key in result
    # alpha is a probability distribution, so it sums to 1
    alpha_sum = float(sum(result["alpha"]))
    assert math.isclose(alpha_sum, 1.0, abs_tol=1e-9)
    # every alpha weight is non-negative
    for a in result["alpha"]:
        assert float(a) >= 0.0
    # context has D entries and all are finite
    ctx = result["context"]
    assert len(ctx) == D
    for c in ctx:
        assert math.isfinite(float(c))
    # entropy is finite and non-negative
    ent = float(result["entropy"])
    assert math.isfinite(ent) and ent >= 0.0
    # estimate is a vector of length D and all entries are finite
    est = result["estimate"]
    assert len(est) == D
    for e in est:
        assert math.isfinite(float(e))
    # for 2D features n equals the row count N
    assert int(result["n"]) == N


def test_hmrvat_edge():
    """Test a 3D spatial feature map (H, W, D) keeps its shape in alpha_map."""
    rng = np.random.default_rng(42)
    H, Wd, D, d_h, k = 3, 4, 2, 5, 3
    features = rng.normal(0, 1, (H, Wd, D))
    h = rng.normal(0, 1, d_h)
    W = rng.normal(0, 1, (k, D))
    U = rng.normal(0, 1, (k, d_h))
    v = rng.normal(0, 1, k)
    result = geron_rnn_visual_attention(features, h, W, U, v)
    assert isinstance(result, dict)
    for key in ("context", "alpha", "alpha_map", "scores", "entropy", "estimate", "n", "method"):
        assert key in result
    # alpha_map preserves the spatial grid shape
    amap = result["alpha_map"]
    assert len(amap) == H
    assert len(amap[0]) == Wd
    # the flattened attention map still sums to 1
    flat_sum = 0.0
    for row in amap:
        for x in row:
            flat_sum += float(x)
    assert math.isclose(flat_sum, 1.0, abs_tol=1e-9)
    # context has D entries and all are finite
    ctx = result["context"]
    assert len(ctx) == D
    for c in ctx:
        assert math.isfinite(float(c))
    # estimate is a vector of length D and all entries are finite
    est = result["estimate"]
    assert len(est) == D
    for e in est:
        assert math.isfinite(float(e))
    # n equals the number of spatial locations H*W
    assert int(result["n"]) == H * Wd


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmrvat as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
