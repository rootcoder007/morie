"""rng143: Wiener autocorrelation matrix (Rangayyan 2024, Eq. 3.163-3.165, pp. 174-175)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsaadapt import rangayyan_ch3_autocorrelation_matrix as phi_mat


def test_rng143_is_symmetric_toeplitz():
    """Eq. (3.164) prints the matrix as Toeplitz with phi(i-k) = phi(k-i)."""
    rng = np.random.default_rng(4)
    Phi = phi_mat(rng.standard_normal(500), 5)["array"]
    assert Phi.shape == (5, 5)
    assert np.allclose(Phi, Phi.T)
    for k in range(5):
        diag = np.diag(Phi, k)
        assert np.allclose(diag, diag[0]), f"diagonal {k} is not constant"


def test_rng143_lags_computed_by_hand():
    """phi(k) = E[x(n)x(n-k)] over the n where the tap vector exists."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    M = 3
    r = phi_mat(x, M)
    #   k=0: x[2:5]*x[2:5] = (9,16,25), mean 50/3
    #   k=1: x[2:5]*x[1:4] = (6,12,20), mean 38/3
    #   k=2: x[2:5]*x[0:3] = (3, 8,15), mean 26/3
    assert r["phi"] == pytest.approx([50 / 3, 38 / 3, 26 / 3])
    assert r["array"][0, 1] == pytest.approx(38 / 3)
    assert r["array"][2, 0] == pytest.approx(26 / 3)


def test_rng143_white_noise_is_near_diagonal():
    """For white noise of variance s^2, Phi -> s^2 * I."""
    rng = np.random.default_rng(2)
    s = 1.5
    Phi = phi_mat(rng.standard_normal(200_000) * s, 4)["array"]
    assert np.allclose(np.diag(Phi), s**2, rtol=0.02)
    off = Phi - np.diag(np.diag(Phi))
    assert np.max(np.abs(off)) < 0.03


def test_rng143_is_positive_definite_for_real_data():
    """Required for the Wiener-Hopf solve and for the RLS matrix-inversion lemma."""
    rng = np.random.default_rng(8)
    x = np.convolve(rng.standard_normal(4000), np.ones(5) / 5, mode="same")
    Phi = phi_mat(x, 6)["array"]
    assert np.all(np.linalg.eigvalsh(Phi) > 0)


def test_rng143_rejects_bad_filter_length():
    with pytest.raises(ValueError, match=">= 1"):
        phi_mat(np.zeros(10), 0)
    with pytest.raises(ValueError, match="exceeds the signal length"):
        phi_mat(np.zeros(10), 11)
