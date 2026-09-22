"""Tests for gh_c11_1.ghosal_gp_def_rkhs."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_1 import ghosal_gp_def_rkhs


def test_gh_c11_1_basic():
    """Test basic functionality."""
    # W ~ N(0, Sigma): build a small symmetric covariance and two vectors a, b.
    Sigma = np.array([[2.0, 0.5, 0.0],
                      [0.5, 1.0, 0.25],
                      [0.0, 0.25, 3.0]])
    a = np.array([1.0, -2.0, 0.5])
    b = np.array([0.5, 1.0, -1.0])

    result = ghosal_gp_def_rkhs(Sigma, a, b)

    # The inner product <Sigma a, Sigma b>_H = a' Sigma b (Ex 11.15).
    expected_ip = float(
        a[0] * (Sigma[0, 0] * b[0] + Sigma[0, 1] * b[1] + Sigma[0, 2] * b[2])
        + a[1] * (Sigma[1, 0] * b[0] + Sigma[1, 1] * b[1] + Sigma[1, 2] * b[2])
        + a[2] * (Sigma[2, 0] * b[0] + Sigma[2, 1] * b[1] + Sigma[2, 2] * b[2])
    )

    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert abs(float(result["estimate"]) - expected_ip) < 1e-10

    # h = Sigma a (the RKHS function in the range of Sigma).
    expected_h = [
        Sigma[0, 0] * a[0] + Sigma[0, 1] * a[1] + Sigma[0, 2] * a[2],
        Sigma[1, 0] * a[0] + Sigma[1, 1] * a[1] + Sigma[1, 2] * a[2],
        Sigma[2, 0] * a[0] + Sigma[2, 1] * a[1] + Sigma[2, 2] * a[2],
    ]
    h = [float(v) for v in result["h"]]
    for hv, ehv in zip(h, expected_h):
        assert abs(hv - ehv) < 1e-10

    # Reproducing property h(t) = <h, K(t, .)>_H (eq. 11.8) holds at
    # every coordinate t of the sample space.
    assert float(result["reproducing_gap"]) < 1e-10


def test_gh_c11_1_edge():
    """Test edge cases."""
    # Trivial 1-D case: W ~ N(0, [[sigma^2]]), h = sigma^2 * a.
    sigma2 = np.array([[4.0]])
    a = np.array([3.0])
    b = np.array([-2.5])

    result = ghosal_gp_def_rkhs(sigma2, a, b)

    # a' Sigma b = a * sigma^2 * b.
    expected_ip = float(a[0] * sigma2[0, 0] * b[0])
    assert abs(float(result["estimate"]) - expected_ip) < 1e-10
    assert abs(float(result["h"][0]) - float(sigma2[0, 0] * a[0])) < 1e-10
    assert float(result["reproducing_gap"]) < 1e-10
