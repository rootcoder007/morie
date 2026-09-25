"""Tests for hmkvc.geron_kv_cache_compress (per-head symmetric quantisation)."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.hmkvc import geron_kv_cache_compress


def _cache(seed):
    return np.random.default_rng(seed).normal(size=(3, 5, 4))


def _quant(vals, bits):
    # symmetric: s = max|x| / (2^(b-1) - 1), q = round-half-even(x / s)
    qmax = 2 ** (bits - 1) - 1
    s = max(abs(v) for v in vals) / qmax
    return [max(-qmax, min(qmax, round(v / s))) * s for v in vals]


def test_hmkvc_basic():
    """4-bit per-head round trip recomputed head by head, and the byte
    accounting: 2 tensors x 60 fp16 values before; ceil(120 * 4 / 8)
    packed bytes plus 6 fp32 scales after."""
    K, V = _cache(1), _cache(2)
    result = geron_kv_cache_compress(K, V, n_bits=4)
    assert isinstance(result, dict)
    err = 0.0
    for name, T in (("K", K), ("V", V)):
        got = np.asarray(result[name + "_dequantized"]).tolist()
        for h in range(3):
            flat = [v for row in T[h].tolist() for v in row]
            exp = _quant(flat, 4)
            gflat = [v for row in got[h] for v in row]
            assert gflat == pytest.approx(exp, rel=1e-12, abs=1e-15)
            err = max(err, max(abs(a - b) for a, b in zip(exp, flat)))
    assert result["bytes_before"] == 120 * 2
    assert result["bytes_after"] == math.ceil(120 * 4 / 8) + 6 * 4
    assert result["max_error"] == pytest.approx(err, rel=1e-12)


def test_hmkvc_edge():
    """A 2-D cache is one head; shapes that differ are refused."""
    K = np.random.default_rng(3).normal(size=(6, 2))
    r = geron_kv_cache_compress(K, K, n_bits=8, per_head=False)
    assert r["bytes_after"] == math.ceil(24 * 8 / 8) + 2 * 4
    with pytest.raises(ValueError):
        geron_kv_cache_compress(K, np.zeros((5, 2)))


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmkvc as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
