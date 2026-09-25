"""Tests for glm.glr_test."""

import math

import pytest

from morie.fn.glm import glr_test


def _page(scores):
    """max_k sum_{i=k}^n z_i by brute force over k, for every n."""
    return [max(0.0, max(sum(scores[k:n + 1]) for k in range(n + 1)))
            for n in range(len(scores))]


def test_glm_basic():
    """The one-pass statistic equals the brute-force maximum of eq. (2.3)."""
    x = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    r = glr_test(x, 0.2, 0.6)
    a, b = math.log(0.6 / 0.2), math.log(0.4 / 0.8)
    z = [a if v else b for v in x]
    assert r["scores"] == pytest.approx(z, rel=1e-15)
    assert r["estimate"] == pytest.approx(_page(z), abs=1e-12)
    assert r["kl"] == pytest.approx(0.6 * a + 0.4 * b, rel=1e-15)


def test_glm_edge():
    """Normal and Poisson scores; the change is dated to the last minimum."""
    r = glr_test([0.1, -0.3, 0.2, 1.4, 0.9, 1.3], 0.0, 1.0, family="normal")
    assert r["estimate"] == pytest.approx(_page([v - 0.5 for v in [0.1, -0.3, 0.2, 1.4, 0.9, 1.3]]), abs=1e-12)
    assert r["changepoint"] == 3 and r["kl"] == 0.5
    x = [1, 0, 2, 5, 4]
    r = glr_test(x, 1.0, 3.0, family="poisson")
    assert r["estimate"] == pytest.approx(_page([v * math.log(3) - 2 for v in x]), abs=1e-12)
    assert r["kl"] == pytest.approx(3 * math.log(3) - 2, rel=1e-15)
    with pytest.raises(ValueError, match="must differ"):
        glr_test(x, 1.0, 1.0, family="poisson")
    with pytest.raises(ValueError, match="0 or 1"):
        glr_test([0.5], 0.1, 0.5)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.glm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
