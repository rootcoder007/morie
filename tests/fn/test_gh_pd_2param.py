"""Tests for gh_pd_2param.ghosal_poisson_dirichlet."""

import pytest

from morie.fn.gh_pd_2param import poisdir


def test_gh_pd_2param_basic():
    """Sticks V_j ~ Be(1 - sigma, M + j sigma): E V_j = (1 - sigma) / (M +
    1 + (j - 1) sigma) and E W_j = E V_j prod_{l<j} (1 - E V_l)."""
    r = poisdir(0.3, 2.0, 4)
    ev = [0.7 / (3.0 + (j - 1) * 0.3) for j in range(1, 5)]
    assert r["expected_stick"] == pytest.approx(ev, rel=1e-15)
    w, rest = [], 1.0
    for e in ev:
        w.append(rest * e)
        rest *= 1 - e
    assert r["weights"] == pytest.approx(w, rel=1e-15)
    assert sum(r["weights"]) + r["remaining"] == pytest.approx(1.0, rel=1e-15)


def test_gh_pd_2param_edge():
    """For n = 2 the EPPF sums to one: V_{2,1} (1 - sigma) + V_{2,2} = 1;
    sigma = 0 is the Dirichlet process, V_{n,k} = M^(k-1) / (M+1)...(M+n-1)."""
    s, M = 0.3, 2.0
    v21 = poisdir(s, M, 1, n=2)["Vnk"]
    v22 = poisdir(s, M, 2, n=2)["Vnk"]
    assert v21 * (1 - s) + v22 == pytest.approx(1.0, rel=1e-14)
    assert poisdir(0.0, 2.0, 3, n=4)["Vnk"] == pytest.approx(2.0 ** 2 / (3 * 4 * 5), rel=1e-14)
    with pytest.raises(ValueError, match="sigma"):
        poisdir(1.0, 2.0, 3)


