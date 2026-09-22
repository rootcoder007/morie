"""Tests for dssm.dssm."""

import math

from morie.fn import _array_core as np

from morie.fn.dssm import dssm


def _cosine(a, b):
    a = list(a)
    b = list(b)
    num = sum(x * y for x, y in zip(a, b))
    den_a = math.sqrt(sum(x * x for x in a))
    den_b = math.sqrt(sum(y * y for y in b))
    return num / (den_a * den_b)


def test_dssm_basic():
    """Test basic functionality."""
    rng_q = np.random.default_rng(42)
    rng_c = rng_q  # placeholder; overwritten below
    query = rng_q.normal(0, 1, 100)

    # The function expects (query_vector, clicked_vector,
    # unclicked_vectors, gamma=...). Provide one or more unclicked
    # documents so the posterior is well-defined.
    rng_c = np.random.default_rng(7)
    clicked = rng_c.normal(0, 1, 100)

    unclicked = [
        np.random.default_rng(101 + i).normal(0, 1, 100)
        for i in range(3)
    ]

    result = dssm(query, clicked, unclicked)
    assert isinstance(result, dict)

    # Documented return keys (function returns a RichResult dict-like with
    # these names).
    assert "estimate" in result
    assert "posterior_clicked" in result
    assert result["estimate"] == result["posterior_clicked"]
    assert "posterior" in result
    assert "similarities" in result
    assert "gamma" in result
    assert "loss" in result
    assert "n_negatives" in result
    assert result["n_negatives"] == len(unclicked)
    assert len(result["posterior"]) == 1 + len(unclicked)
    assert len(result["similarities"]) == 1 + len(unclicked)

    # Independent recomputation of the formula from the same inputs:
    # similarities are cosine(query, doc_i) for i in {clicked} + unclicked,
    # then softmax over gamma * similarities.
    sims = [_cosine(query, clicked)]
    for d in unclicked:
        sims.append(_cosine(query, d))

    g = float(result["gamma"])
    sc = [g * v for v in sims]
    m = max(sc)
    e = [math.exp(v - m) for v in sc]
    z = sum(e)
    p_expected = [v / z for v in e]

    for got, exp in zip(result["posterior"], p_expected):
        assert math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-12)

    assert math.isclose(
        result["estimate"], p_expected[0], rel_tol=1e-9, abs_tol=1e-12
    )

    # Posterior sums to 1.
    assert math.isclose(
        sum(result["posterior"]), 1.0, rel_tol=1e-9, abs_tol=1e-12
    )

    # Loss is -log(estimate), guarded by a small epsilon in the impl, so
    # compare against max(estimate, eps).
    eps = 1e-300  # matches the implementation's _EPS
    loss_expected = -math.log(max(result["estimate"], eps))
    assert math.isclose(
        result["loss"], loss_expected, rel_tol=1e-9, abs_tol=1e-12
    )


def test_dssm_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    doc = np.random.default_rng(7).normal(0, 1, 100)
    unclicked = [np.random.default_rng(101).normal(0, 1, 100)]
    result = dssm(query, doc, unclicked)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "posterior" in result
    assert len(result["posterior"]) == 2
    assert math.isclose(
        sum(result["posterior"]), 1.0, rel_tol=1e-9, abs_tol=1e-12
    )
