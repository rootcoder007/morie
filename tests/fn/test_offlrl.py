"""Tests for offlrl.offline_rl_cql (Kumar et al. 2020, CQL(H) eq. 4)."""

import math

import pytest

from morie.fn.offlrl import offline_rl_cql


D = [("s", "a0", 1.0, "s", True), ("s", "a0", 1.0, "s", True),
     ("s", "a1", 0.0, "s", True), ("s", "a1", 0.0, "s", True), ("s", "a1", 1.0, "s", True)]


def test_offlrl_basic():
    """One state, terminal transitions: the CQL(H) objective
    alpha (logsumexp Q - E_beta Q) + (1/N) sum 1/2 (Q - r)^2 is strictly
    convex and its stationary point satisfies
    alpha (softmax_a - beta_a) + (n_a / N)(Q_a - rbar_a) = 0, which forces
    the count-weighted Q to equal the count-weighted reward; alpha = 0 is
    plain fitted Q (Q = rbar)."""
    alpha = 0.5
    r = offline_rl_cql(D, alpha=alpha, iters=20000)
    q = [r["q"][("s", "a0")], r["q"][("s", "a1")]]
    beta, nfrac, rbar = [0.4, 0.6], [0.4, 0.6], [1.0, 1.0 / 3.0]
    sm = [math.exp(v) / sum(math.exp(u) for u in q) for v in q]
    for i in range(2):
        assert alpha * (sm[i] - beta[i]) + nfrac[i] * (q[i] - rbar[i]) == pytest.approx(0.0, abs=1e-10)
    assert r["behavior"] == {("s", "a0"): 0.4, ("s", "a1"): 0.6}
    lse = math.log(sum(math.exp(v) for v in q))
    assert r["penalty"] == pytest.approx(lse - (0.4 * q[0] + 0.6 * q[1]), rel=1e-12)
    assert q[0] < 1.0      # the better action is pushed down: conservatism
    plain = offline_rl_cql(D, alpha=0.0, iters=20000)
    assert [plain["q"][("s", "a0")], plain["q"][("s", "a1")]] == pytest.approx(rbar, abs=1e-10)
    assert plain["greedy"]["s"] == "a0"


def test_offlrl_edge():
    """Unknown variants, negative alpha and variant='mu' without mu raise."""
    for kw in ({"variant": "Q"}, {"alpha": -1.0}, {"variant": "mu"}):
        with pytest.raises(ValueError):
            offline_rl_cql(D, **kw)
