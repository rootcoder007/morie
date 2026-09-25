"""Tests for spcgme.spatial_concordance_kappa (neighbour-pair kappa)."""

import pytest

from morie.fn.spcgme import spatial_concordance_kappa


def _line(n=6):
    return [[1.0 if abs(i - j) == 1 else 0.0 for j in range(n)] for i in range(n)]


def test_spcgme_basic():
    """p_o = sum w_ij I{x_i = y_j}/w..; p_e = sum_c px_c qy_c with the
    weight-marginal shares; kappa = (p_o - p_e)/(1 - p_e).  Worked by hand
    on a 6-site line (w.. = 10)."""
    w = _line()
    x = [1, 1, 2, 2, 3, 3]
    y = [1, 2, 2, 2, 3, 1]
    n = 6
    s0 = sum(map(sum, w))
    po = sum(w[i][j] for i in range(n) for j in range(n) if x[i] == y[j]) / s0
    rows = [sum(r) for r in w]
    cols = [sum(w[i][j] for i in range(n)) for j in range(n)]
    pe = sum(sum(rows[i] for i in range(n) if x[i] == c) / s0
             * sum(cols[j] for j in range(n) if y[j] == c) / s0 for c in (1, 2, 3))
    r = spatial_concordance_kappa(x, y, w)
    assert r["p_observed"] == pytest.approx(po, abs=1e-15)
    assert r["p_expected"] == pytest.approx(pe, abs=1e-15)
    assert r["kappa"] == pytest.approx((po - pe) / (1 - pe), abs=1e-15)
    # the pairs (i,j) with x_i = y_j on the line: (0,1)? x0=1,y1=2 no;
    # (1,0) 1=1, (1,2) 1=2 no, (2,1) 2=2, (2,3) 2=2, (3,2) 2=2,
    # (3,4) 2=3 no, (4,3) 3=2 no, (4,5) 3=1 no, (5,4) 3=3 -> 5 of 10
    assert po == 0.5
    assert r["s0"] == 10.0 and r["n"] == 6


def test_spcgme_edge():
    """Non-integer codes, a non-zero diagonal, negative weights and
    two constant equal maps (p_e = 1) raise."""
    w = _line()
    with pytest.raises(ValueError):
        spatial_concordance_kappa([1.5] * 6, [1] * 6, w)
    wd = [r[:] for r in w]
    wd[0][0] = 1.0
    with pytest.raises(ValueError):
        spatial_concordance_kappa([1, 2] * 3, [1, 2] * 3, wd)
    wn = [r[:] for r in w]
    wn[0][1] = -1.0
    with pytest.raises(ValueError):
        spatial_concordance_kappa([1, 2] * 3, [1, 2] * 3, wn)
    with pytest.raises(ValueError):
        spatial_concordance_kappa([4] * 6, [4] * 6, w)
