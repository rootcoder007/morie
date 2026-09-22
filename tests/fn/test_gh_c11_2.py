"""Tests for gh_c11_2.ghosal_rkhs_norm."""

from morie.fn import _array_core as np

from morie.fn.gh_c11_2 import ghosal_rkhs_norm


def test_gh_c11_2_basic():
    """Test basic functionality."""
    f0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    f0v = np.asarray(f0, dtype=float).ravel()
    lam = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    lamv = np.asarray(lam, dtype=float).ravel()
    eps = 1.0
    result = ghosal_rkhs_norm(f0, lam, eps)
    assert "estimate" in result
    estimate = result["estimate"]
    arr_est = np.asarray(estimate, dtype=float)
    assert np.all(np.isfinite(arr_est))
    # Compute expected decentering norm2 with same greedy truncation as impl
    eps_f = float(eps)
    idx = sorted(range(len(f0v)), key=lambda i: -abs(float(f0v[i])))
    h = [0.0] * len(f0v)
    norm = 0.0
    for j in idx:
        norm = float(np.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(f0v, h))))
        if norm <= eps_f:
            break
        h[j] = float(f0v[j])
    hn2 = sum((float(h[i]) ** 2) / float(lamv[i]) for i in range(len(h)))
    # estimate = 0.5 * hn2 + small_ball; small_ball >= 0 and finite
    assert arr_est >= 0.5 * hn2 - 1e-12
    assert arr_est <= 0.5 * hn2 + 40.0  # reasonable bound for small_ball <= log(n_sim)


def test_gh_c11_2_edge():
    """Test edge cases."""
    f0 = np.array([42.0])
    lam = np.array([1.0])
    eps = 0.5
    result = ghosal_rkhs_norm(f0, lam, eps)
    assert "estimate" in result
    arr_est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(arr_est))
