"""Test multi-head attention."""

from morie.fn import _array_core as np

from morie.fn.mhatt import mhatt


def test_mhatt_basic():
    """Test basic attention."""
    q = np.random.randn(5, 512)
    k = np.random.randn(5, 512)
    v = np.random.randn(5, 512)
    result = mhatt(q, k, v, num_heads=8, d_model=512)
    assert result["output"].shape == q.shape


def test_mhatt_weights_sum():
    """Test attention weights sum to 1."""
    q = np.random.randn(4, 64)
    k = np.random.randn(4, 64)
    v = np.random.randn(4, 64)
    result = mhatt(q, k, v, num_heads=4, d_model=64)
    sums = np.sum(result["attention_weights"], axis=-1)
    assert np.allclose(sums, 1.0)


def test_mhatt_invalid_heads():
    """Test invalid num_heads."""
    q = np.random.randn(5, 64)
    k = np.random.randn(5, 64)
    v = np.random.randn(5, 64)
    try:
        mhatt(q, k, v, num_heads=5, d_model=64)
        assert False
    except ValueError:
        pass


def test_mhatt_each_head_attends_on_its_own_slice():
    """Recomputed from Vaswani et al. Sec. 3.2 with identity
    projections: head i uses columns i*d_k .. (i+1)*d_k - 1 and
    softmax(Q_i K_i^T / sqrt(d_k)) V_i; the output concatenates heads."""
    import math

    import pytest

    rng = np.random.default_rng(5)
    q, k, v = (rng.normal(0, 1, (3, 6)).tolist() for _ in range(3))
    r = mhatt(q, k, v, num_heads=3, d_model=6)
    out = r["output"].tolist()
    W = r["attention_weights"].tolist()
    for h in range(3):
        cols = (2 * h, 2 * h + 1)
        for i in range(3):
            s = [sum(q[i][c] * k[j][c] for c in cols) / math.sqrt(2) for j in range(3)]
            m = max(s)
            e = [math.exp(x - m) for x in s]
            w = [x / sum(e) for x in e]
            assert W[h][i] == pytest.approx(w, rel=1e-13)
            for c in cols:
                assert out[i][c] == pytest.approx(sum(w[j] * v[j][c] for j in range(3)), rel=1e-12, abs=1e-14)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import importlib as _importlib

_doctest_module = _importlib.import_module("morie.fn.mhatt")


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
