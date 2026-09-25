"""Tests for meglt.matrix_completion_low_rank (singular value thresholding,
Cai, Candes & Shen 2010)."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.meglt import matrix_completion_low_rank as svt

M = [[1.0, 2.0, -1.0, 0.5], [2.0, 4.0, -2.0, 1.0], [0.5, 1.0, -0.5, 0.25]]
OBS = [(0, 0), (0, 2), (1, 1), (1, 3), (2, 0), (2, 1), (2, 3), (0, 3)]


def test_meglt_basic():
    """Two SVT sweeps from Y = 0: X_1 = 0, so the first residual is
    ||P_Omega(M)||_F and Y_1 = step P_Omega(M); X_2 soft-thresholds the
    singular values of Y_1 by tau, recomputed here from its SVD."""
    step, tau = 1.9, 1.5
    r = svt(M, OBS, tau=tau, step=step, iters=2)
    assert isinstance(r, dict)
    po = [[M[i][j] if (i, j) in OBS else 0.0 for j in range(4)] for i in range(3)]
    assert r["residual_history"][0] == pytest.approx(
        math.sqrt(sum(v * v for row in po for v in row)), rel=1e-14)
    U, s, Vt = np.linalg.svd(np.array([[step * v for v in row] for row in po]),
                             full_matrices=False)
    U, s, Vt = U.tolist(), [max(0.0, float(v) - tau) for v in s], Vt.tolist()
    X2 = [[sum(U[i][q] * s[q] * Vt[q][j] for q in range(len(s))) for j in range(4)]
          for i in range(3)]
    for i in range(3):
        assert r["X"][i] == pytest.approx(X2[i], rel=1e-10, abs=1e-12)


def test_meglt_edge():
    """No observed entries is refused; the reported fraction is |Omega|/n1n2."""
    r = svt(M, OBS, iters=1)
    assert r["fraction_observed"] == len(OBS) / 12
    with pytest.raises(ValueError):
        svt(M, [])


