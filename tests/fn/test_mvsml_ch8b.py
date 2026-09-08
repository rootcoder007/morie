"""Known-answer tests for MVSML chapter 8, eq. (8.4)-(8.5)."""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm131 import (mvsml_categorical_count_eq_8_4,
                             mvsml_arccos_kernel)
from morie.fn.msm132 import mvsml_categorical_count_eq_8_5

X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]


def test_arccos_preserves_the_norm_on_the_diagonal():
    # p.265: AK(x_i, x_i) = ||x_i||^2
    K = gp.arccos_kernel(X)
    for i, row in enumerate(X):
        n2 = sum(v * v for v in row)
        assert abs(K[i][i] - n2) < 1e-12


def test_arccos_of_opposite_vectors_is_zero():
    # p.265: AK(x_i, -x_i) = 0
    K = gp.arccos_kernel([[1.0, 2.0]], Z=[[-1.0, -2.0]])
    assert abs(K[0][0]) < 1e-12


def test_arccos_hand_value_for_orthogonal_inputs():
    # theta = pi/2, J = sin(pi/2) + (pi - pi/2) cos(pi/2) = 1
    # so AK = (1/pi) * 1 * 1 * 1
    K = gp.arccos_kernel([[1.0, 0.0]], Z=[[0.0, 1.0]])
    assert abs(K[0][0] - 1.0 / math.pi) < 1e-12


def test_arccos_kernel_is_symmetric_and_psd():
    r = mvsml_categorical_count_eq_8_4(X)
    K = r["kernel"]
    for i in range(len(X)):
        for j in range(len(X)):
            assert abs(K[i][j] - K[j][i]) < 1e-12
    assert r["positive_semidefinite"] is True


def test_diagonal_is_heterogeneous_unlike_the_gaussian():
    # p.265: the AK diagonal expresses heterogeneous variances
    K = gp.arccos_kernel(X)
    diag = [K[i][i] for i in range(len(X))]
    assert max(diag) - min(diag) > 1e-6
    G = gp.kernel_matrix(X, kernel="gaussian")
    gdiag = [G[i][i] for i in range(len(X))]
    assert max(gdiag) - min(gdiag) < 1e-12       # all ones


def test_depth_one_matches_eq_8_4():
    a = gp.arccos_kernel(X, depth=1)
    b = mvsml_categorical_count_eq_8_5(X, depth=1)["kernel"]
    for i in range(len(X)):
        for j in range(len(X)):
            assert abs(a[i][j] - b[i][j]) < 1e-12


def test_deeper_layers_stay_psd_and_change_the_kernel():
    r2 = mvsml_categorical_count_eq_8_5(X, depth=2)
    r3 = mvsml_categorical_count_eq_8_5(X, depth=3)
    assert r2["positive_semidefinite"] is True
    assert r3["positive_semidefinite"] is True
    K1 = gp.arccos_kernel(X, depth=1)
    assert any(abs(r2["kernel"][i][j] - K1[i][j]) > 1e-9
               for i in range(len(X)) for j in range(len(X)))


def test_median_normalization_matches_the_book_r_code():
    # the book's K.AK1_Final divides by median(AK1)
    K = gp.arccos_kernel(X, normalize_median=True)
    raw = gp.arccos_kernel(X)
    flat = sorted(v for row in raw for v in row)
    n = len(flat)
    med = flat[n // 2] if n % 2 else 0.5 * (flat[n // 2 - 1]
                                            + flat[n // 2])
    assert abs(K[0][0] - raw[0][0] / med) < 1e-12


def test_canonical_alias():
    assert mvsml_arccos_kernel is mvsml_categorical_count_eq_8_4
