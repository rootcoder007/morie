"""spstvg -- spatio-temporal semivariogram, Schabenberger & Gotway Sec. 9.4."""

from morie.fn import _array_core as np
import pytest

from morie.fn._schab_st import (semivariogram_from_covariance,
                                separable_covariance)
from morie.fn.spstvg import schabenberger_st_variogram

CS = lambda h: 2.0 * np.exp(-h / 3.0)          # noqa: E731
CT = lambda k: 1.5 * np.exp(-k / 2.0)          # noqa: E731
COV = lambda d, u: separable_covariance(d, u, CS, CT)      # noqa: E731


def _data(n=80, seed=3):
    rs = np.random.RandomState(seed)
    return (rs.uniform(0, 10, size=(n, 2)),
            rs.uniform(0, 5, n),
            rs.normal(0, 1, n))


def test_gamma_zero_zero_is_zero_and_reaches_the_sill():
    """gamma(h,k) = C(0,0) - C(h,k), Sec. 9.4."""
    assert semivariogram_from_covariance(
        np.array([0.0]), np.array([0.0]), COV)[0] == pytest.approx(0.0, abs=1e-12)
    assert semivariogram_from_covariance(
        np.array([1e3]), np.array([1e3]), COV)[0] == pytest.approx(3.0)


def test_estimator_matches_its_own_definition_exactly():
    """eq (9.18): gamma_hat = sum (Z_i - Z_j)^2 / (2 |N(h,k)|).

    Computed independently here from the same pairs, so the factor of 2 and
    the pair count are both pinned exactly rather than statistically.
    """
    coords, times, z = _data()
    r = schabenberger_st_variogram(coords, times, z, n_space_bins=4,
                                   n_time_bins=3)
    i, j = np.triu_indices(z.size, k=1)
    d = np.linalg.norm(coords[i] - coords[j], axis=1)
    u = np.abs(times[i] - times[j])
    sq = (z[i] - z[j]) ** 2
    de, ue = r["space_edges"], r["time_edges"]
    keep = (d <= de[-1]) & (u <= ue[-1])
    d, u, sq = d[keep], u[keep], sq[keep]
    di = np.clip(np.digitize(d, de) - 1, 0, len(de) - 2)
    ui = np.clip(np.digitize(u, ue) - 1, 0, len(ue) - 2)
    tot = np.zeros(r["st_variogram"].shape)
    cnt = np.zeros(r["st_variogram"].shape)
    np.add.at(tot, (di, ui), sq)
    np.add.at(cnt, (di, ui), 1.0)
    want = np.where(cnt > 0, tot / (2.0 * np.maximum(cnt, 1)), np.nan)

    assert np.array_equal(r["counts"], cnt.astype(int))
    got = r["st_variogram"]
    both = ~np.isnan(want) & ~np.isnan(got)
    assert np.allclose(got[both], want[both], rtol=0, atol=1e-12)
    # and the half really is a half
    assert not np.allclose(got[both], 2.0 * want[both])


def test_empty_cells_are_nan_not_zero():
    """An unestimated semivariogram and a zero one are different claims."""
    coords, times, z = _data(n=25, seed=11)
    r = schabenberger_st_variogram(coords, times, z, n_space_bins=12,
                                   n_time_bins=8)
    empty = r["counts"] == 0
    if empty.any():
        assert np.all(np.isnan(r["st_variogram"][empty]))
        assert "warning" in r


def test_conditional_estimator_uses_only_that_time_slice():
    """eq (9.19) is a different quantity from eq (9.18)."""
    rs = np.random.RandomState(4)
    coords = rs.uniform(0, 10, size=(60, 2))
    times = np.repeat([0.0, 1.0, 2.0], 20)
    z = rs.normal(0, 1, 60)
    r = schabenberger_st_variogram(coords, times, z, n_space_bins=3,
                                   n_time_bins=3, at_time=1.0)
    assert r["conditional"]["n_at_time"] == 20


def test_wls_objective_prefers_the_true_model():
    """The Sec. 9.4 weighted least squares criterion."""
    rs = np.random.RandomState(9)
    n = 120
    coords = rs.uniform(0, 10, size=(n, 2))
    times = rs.uniform(0, 5, n)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    k = np.abs(times[:, None] - times[None, :])
    c = COV(d, k)
    c[np.diag_indices_from(c)] += 1e-8
    z = np.linalg.cholesky(c) @ rs.normal(0, 1, n)

    truth = lambda h, u: semivariogram_from_covariance(h, u, COV)  # noqa: E731
    wrong = lambda h, u: semivariogram_from_covariance(               # noqa: E731
        h, u, lambda dd, uu: separable_covariance(
            dd, uu, lambda x: 2.0 * np.exp(-x / 40.0), CT))
    a = schabenberger_st_variogram(coords, times, z, n_space_bins=4,
                                   n_time_bins=3, model_fn=truth)
    b = schabenberger_st_variogram(coords, times, z, n_space_bins=4,
                                   n_time_bins=3, model_fn=wrong)
    assert a["wls_objective"] < b["wls_objective"]


def test_mismatched_lengths_rejected():
    coords, times, z = _data(n=20)
    with pytest.raises(ValueError, match="same length"):
        schabenberger_st_variogram(coords, times[:-1], z)
