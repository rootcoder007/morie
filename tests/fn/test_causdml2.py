import math

"""Tests for causdml2.causal_dml_partial_lin."""

from morie.fn import _array_core as np
from morie.fn.causdml2 import causal_dml_partial_lin, causal_dml_plr_gcv, rmorie_folds


def test_causdml2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_dml_partial_lin(y, D, X)
    assert isinstance(result, dict)
    assert "theta" in result


def test_causdml2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = causal_dml_partial_lin(y, D, X)
    assert isinstance(result, dict)


def _rmorie_rows():
    # the deterministic frame rmorie's tests/testthat/test-mrm_design.R builds for the same constants
    i = list(range(1, 151))
    x1 = [math.sin(v) for v in i]
    x2 = [math.cos(1.4 * v) for v in i]
    d = [int(0.7 * a + 0.4 * math.sin(3.1 * v) > 0) for a, v in zip(x1, i)]
    y = [1 + 0.5 * dd + a + 0.3 * math.cos(2.2 * v) for dd, a, v in zip(d, x1, i)]
    return y, d, [[a, b] for a, b in zip(x1, x2)]


def test_rmorie_folds_match_the_r_rule():
    from morie.tps_hawkes_advanced import splitmix_uniforms

    n, k, seed = 23, 5, 42
    f = rmorie_folds(n, k, seed)
    # R: folds[order(core_uniforms(n, seed))] <- rep_len(seq_len(k), n), zero-based here
    u = splitmix_uniforms(n, seed)
    order = sorted(range(n), key=lambda j: u[j])
    expect = [0] * n
    for pos, j in enumerate(order):
        expect[j] = pos % k
    assert list(f) == expect
    assert sorted(set(f.tolist())) == list(range(k))


def test_causal_dml_plr_gcv_equals_rmorie_double_ml():
    y, d, X = _rmorie_rows()
    r = causal_dml_plr_gcv(y, d, X)
    # rmorie: morie_estimate_double_ml(df, "Y", "D", c("x1", "x2")) on the same frame (seed 42, 5 folds)
    assert abs(r["theta"] - 0.468147062199554) < 1e-9
    assert abs(r["se"] - 0.0543268562926612) < 1e-9
    assert r["n_folds"] == 5 and r["n"] == 150 and r["p"] == 2
