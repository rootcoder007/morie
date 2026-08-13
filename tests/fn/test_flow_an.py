"""Tests for flow_an. Full anchor: ledger/wave3/anchor_ts_family.py."""
import math
import pytest
from morie.fn import _array_core as np
from morie.fn.flow_an import (alternating_masks, anomaly_score,
                              flow_forward, flow_inverse, log_prob)

D, NL = 4, 4


def layers(seed):
    r = np.random.default_rng(seed)
    masks = alternating_masks(D, NL)
    out = []
    for t in range(NL):
        M = lambda: [[r.standard_normal() * 0.3 for _ in range(D)]
                     for _ in range(D)]
        b = lambda: [r.standard_normal() * 0.1 for _ in range(D)]
        out.append((masks[t], M(), b(), M(), b()))
    return out


@pytest.fixture(scope="module")
def lay():
    return layers(3)


def test_the_flow_inverts_exactly(lay):
    x = [0.4, -1.1, 0.7, 0.2]
    z, ld = flow_forward(x, lay)
    xr, ldi = flow_inverse(z, lay)
    assert xr == pytest.approx(x, abs=1e-12)
    assert ld + ldi == pytest.approx(0.0, abs=1e-12)


def test_the_log_determinant_matches_the_jacobian(lay):
    """A sign error here leaves the model trainable and the likelihood
    wrong, which never surfaces on its own."""
    x = [0.4, -1.1, 0.7, 0.2]
    _, ld = flow_forward(x, lay)
    h = 1e-6
    J = []
    for i in range(D):
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        zp, _ = flow_forward(xp, lay)
        zm, _ = flow_forward(xm, lay)
        J.append([(zp[j] - zm[j]) / (2 * h) for j in range(D)])
    M = [[J[j][i] for j in range(D)] for i in range(D)]
    det = 0.0
    for i in range(D):
        piv = max(range(i, D), key=lambda r_: abs(M[r_][i]))
        M[i], M[piv] = M[piv], M[i]
        det += math.log(abs(M[i][i]))
        for r_ in range(i + 1, D):
            f = M[r_][i] / M[i][i]
            for c_ in range(i, D):
                M[r_][c_] -= f * M[i][c_]
    assert ld == pytest.approx(det, abs=1e-4)


def test_masks_must_alternate_or_channels_are_never_modelled(lay):
    same = [(alternating_masks(D, 1)[0], l[1], l[2], l[3], l[4])
            for l in lay]
    z1, _ = flow_forward([1.0, 0.0, 0.0, 0.0], same)
    z2, _ = flow_forward([2.0, 0.0, 0.0, 0.0], same)
    assert z1[0] == pytest.approx(1.0, abs=1e-12)
    assert z2[0] == pytest.approx(2.0, abs=1e-12)
    with pytest.raises(ValueError):
        alternating_masks(1, 2)


def test_the_outlier_scores_highest(lay):
    rng = np.random.default_rng(5)
    X = [[rng.standard_normal() * 0.5 for _ in range(D)]
         for _ in range(50)]
    X.append([9.0, -9.0, 9.0, -9.0])
    r = anomaly_score(X, lay, threshold_quantile=0.9)
    assert r["score"].index(max(r["score"])) == len(X) - 1
    lp, _, _ = log_prob(X[0], lay)
    assert math.isfinite(lp)
    with pytest.raises(ValueError):
        anomaly_score(X, lay, threshold_quantile=1.5)
