"""Tests for dreamr.dreamer."""

import pytest

from morie.fn.dreamr import lambda_return


def test_dreamr_basic():
    """Eq. 6 equals the recursion V(tau) = r_tau + gamma ((1 - lam)
    v_{tau+1} + lam V(tau+1)) with V(H) = v_H."""
    r = [0.5, -0.2, 1.0, 0.3, 0.8]
    v = [0.1, 0.4, -0.3, 0.9, 0.2, 0.6]
    g, lam = 0.9, 0.7
    got = lambda_return(r, v, gamma=g, lam=lam)["returns"]
    rec = [0.0] * 5
    nxt = v[5]
    for t in range(4, -1, -1):
        rec[t] = r[t] + g * ((1 - lam) * v[t + 1] + lam * nxt)
        nxt = rec[t]
    assert got == pytest.approx(rec, rel=1e-12)


def test_dreamr_edge():
    """lam = 1 is the discounted return with a bootstrap at the horizon;
    eq. 4 is the plain sum; eq. 5 with k = 1 is the TD(0) target."""
    r = [0.5, -0.2, 1.0]
    v = [0.1, 0.4, -0.3, 0.9]
    one = lambda_return(r, v, gamma=0.9, lam=1.0)["returns"]
    assert one[0] == pytest.approx(0.5 - 0.9 * 0.2 + 0.81 * 1.0 + 0.729 * 0.9, rel=1e-12)
    assert lambda_return(r, v, estimator="reward")["returns"] == pytest.approx([1.3, 0.8, 1.0], rel=1e-12)
    td = lambda_return(r, v, gamma=0.9, estimator="k-step", k=1)["returns"]
    assert td == pytest.approx([r[t] + 0.9 * v[t + 1] for t in range(3)], rel=1e-12)
    with pytest.raises(ValueError, match="one more entry"):
        lambda_return(r, v[:3])


