"""Tests for gh_c8_9.ghosal_markov_crt."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c8_9 import ghosal_markov_crt


def _expected_rate(a0, b0, ns, seed):
    """Independent reimplementation of the documented posterior-risk rate."""
    rng = np.random.default_rng(seed)
    risks = []
    for n in ns:
        x = 0
        c = [[0, 0], [0, 0]]
        for _ in range(n):
            p = a0 if x == 0 else b0
            u = float(rng.uniform(0, 1))
            nxt = 1 - x if u < p else x
            if nxt != x:
                c[x][1] += 1
            else:
                c[x][0] += 1
            x = nxt
        s0 = c[0][0] + c[0][1]
        s1 = c[1][0] + c[1][1]
        va = ((1 + c[0][1]) * (1 + c[0][0])
              / ((2 + s0) ** 2 * (3 + s0)))
        vb = ((1 + c[1][1]) * (1 + c[1][0])
              / ((2 + s1) ** 2 * (3 + s1)))
        risks.append(va + vb)
    rate_hat = math.log(risks[0] / risks[-1]) \
        / math.log(float(ns[-1]) / ns[0])
    return rate_hat, risks


def test_gh_c8_9_basic():
    """Default-arg call: keys, finiteness, and the documented rate expression."""
    result = ghosal_markov_crt()
    assert "estimate" in result
    assert "posterior_var_by_n" in result
    assert result["method"] == "Markov contraction (GvdV 2017 sec. 8.3.3)"

    est = result["estimate"]
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))

    ns = (200, 800, 3200)
    expected, risks = _expected_rate(0.3, 0.5, ns, 42)
    assert est == expected
    assert list(result["posterior_var_by_n"]) == risks
    assert len(risks) == len(ns)
    assert risks[0] > risks[-1] > 0


def test_gh_c8_9_custom_seed():
    """A different seed must reproduce the same closed-form result for the
    same (a0, b0, ns)."""
    expected, _ = _expected_rate(0.7, 0.2, (100, 400), 7)
    result = ghosal_markov_crt(a0=0.7, b0=0.2, ns=(100, 400), seed=7)
    assert result["estimate"] == expected


def test_gh_c8_9_edge():
    """Two-element ns gives a finite, finite-valued rate."""
    result = ghosal_markov_crt(ns=(50, 200), seed=1)
    est = result["estimate"]
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))
    expected, _ = _expected_rate(0.3, 0.5, (50, 200), 1)
    assert est == expected
