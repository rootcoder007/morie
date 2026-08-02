"""spperiod -- the periodogram, Schabenberger & Gotway Sec. 4.7.1."""

from morie.fn import _array_core as np
import pytest

from morie.fn.spperiod import schabenberger_periodogram as per


def _field(r=8, c=6, seed=4):
    rs = np.random.RandomState(seed)
    return rs.standard_normal((r, c)) + 0.5 * np.add.outer(np.arange(r), np.arange(c))


def test_eq_4_59_identity_holds_to_machine_precision():
    """The periodogram IS the Fourier transform of Chat, away from the origin."""
    p = per(_field())
    assert p["identity_holds"]
    assert p["identity_max_abs_diff"] < 1e-12


def test_the_2pi_normalisation_is_what_closes_the_identity():
    """Scaling by n instead of (2 pi)^2 rc -- the stub's version -- breaks (4.59)."""
    p = per(_field())
    wrong = p["periodogram"] * ((2 * np.pi) ** 2 * p["r"] * p["c"]) / (p["r"] * p["c"])
    m = p["nonzero_mask"]
    assert np.abs(wrong[m] - p["periodogram_from_covariance"][m]).max() > 1.0


def test_one_fourier_frequency_per_row_and_column():
    p = per(_field(8, 6))
    assert p["omega1"].size == 8 and p["omega2"].size == 6
    assert np.all(np.abs(p["omega1"]) <= np.pi + 1e-12)
    assert np.all(np.abs(p["omega2"]) <= np.pi + 1e-12)
    assert p["j"][0] == -3 and p["j"][-1] == 4
    assert p["k"][0] == -2 and p["k"][-1] == 3


def test_mean_invariance_off_the_origin():
    """p. 191: sum_u cos(w_j u) = 0, so adding a constant changes nothing."""
    z = _field()
    a = per(z)
    b = per(z + 57.0)
    m = a["nonzero_mask"]
    assert a["mean_invariant"] and b["mean_invariant"]
    assert np.allclose(a["periodogram"][m], b["periodogram"][m])


def test_periodogram_is_nonnegative():
    assert np.all(per(_field())["periodogram"] >= -1e-12)


def test_pure_cosine_concentrates_at_its_frequency():
    """A single Fourier mode puts its energy at exactly that frequency."""
    r, c = 12, 10
    u = np.arange(1, r + 1)
    z = np.outer(np.cos(2 * np.pi * 3 * u / r), np.ones(c))
    p = per(z)
    flat = p["periodogram"]
    peak = np.unravel_index(np.argmax(flat), flat.shape)
    assert abs(p["j"][peak[0]]) == 3
    assert p["k"][peak[1]] == 0
    assert flat[peak] > 100 * np.median(flat[flat > 0])


def test_covariance_at_zero_lag_is_the_sample_variance():
    z = _field()
    p = per(z)
    r, c = z.shape
    assert p["covariance"][r - 1, c - 1] == pytest.approx(
        ((z - z.mean()) ** 2).sum() / (r * c))


def test_covariance_is_even_in_its_lags():
    p = per(_field())
    cov = p["covariance"]
    assert np.allclose(cov, cov[::-1, ::-1])


def test_identity_failure_is_a_warning_not_silence():
    p = per(_field())
    assert "warning" not in p
    assert p["identity_holds"] is True


def test_white_noise_is_flat_on_average():
    """For iid noise E[I] = sigma^2/(2 pi)^2 at every non-zero frequency.

    Periodogram ordinates are asymptotically exponential, so each has
    relative sd 1 and the mean of R replicates has relative sd 1/sqrt(R).
    The per-frequency bound is therefore 5/sqrt(R) -- five standard errors,
    not a taste threshold -- and the pooled mean, which averages ~84
    frequencies as well, gets a much tighter one.
    """
    rs = np.random.RandomState(9)
    R = 40
    acc = []
    for _ in range(R):
        z = rs.standard_normal((10, 10))
        p = per(z)
        acc.append(p["periodogram"][p["nonzero_mask"]])
    mean_i = np.mean(np.stack(acc), axis=0)
    expect = 1.0 / (2 * np.pi) ** 2
    assert np.abs(mean_i / expect - 1.0).max() < 5.0 / np.sqrt(R)
    assert abs(float(np.mean(mean_i)) / expect - 1.0) < 0.05


def test_coords_argument_is_ignored_by_design():
    z = _field()
    a = per(z)
    b = per(z, coords=np.arange(48).reshape(-1, 2))
    m = a["nonzero_mask"]
    assert np.allclose(a["periodogram"][m], b["periodogram"][m])


def test_rejects_non_lattice_input():
    with pytest.raises(ValueError):
        per(np.arange(10.0))
    with pytest.raises(ValueError):
        per(np.ones((1, 5)))
    bad = _field()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        per(bad)
