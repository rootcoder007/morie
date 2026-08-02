"""Horowitz template-B repairs: kernel core, single index, additive,
series and quantile estimators.

Tests target the properties each method is CHOSEN for -- boundary bias,
rate in d, identification normalisations -- not just output shape."""

from morie.fn import _array_core as np
import pytest

from morie.fn._horowitz import check_rate, kde, local_linear, nw_regression
from morie.fn.hrzbkft import hrz_backfitting
from morie.fn.hrzbwopt import hrz_bandwidth_optimal
from morie.fn.hrzderiv import hrz_density_derivative
from morie.fn.hrzich import hrz_ichimura
from morie.fn.hrzkd2 import hrz_kde_multivariate
from morie.fn.hrzkde import hrz_kde
from morie.fn.hrzkqre import hrz_kernel_quantile
from morie.fn.hrzllqr import hrz_local_linear_quantile
from morie.fn.hrzllr import hrz_local_linear
from morie.fn.hrzmscr import hrz_maximum_score
from morie.fn.hrznls import hrz_semiparametric_ls
from morie.fn.hrznwr import hrz_nw_regression
from morie.fn.hrznwrg import hrz_index_nw
from morie.fn.hrzplr import hrz_partially_linear
from morie.fn.hrzsieqr import hrz_series_quantile
from morie.fn.hrzsier import hrz_series_regression
from morie.fn.hrzsms import hrz_smoothed_max_score


def test_kde_integrates_to_one_and_recovers_a_normal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    out = hrz_kde(x)
    assert out["integrates_to"] == pytest.approx(1.0, abs=0.02)
    assert out["rate_exponent"] == pytest.approx(-0.4)
    # the peak sits near 0 with roughly the standard normal height
    ipk = int(np.argmax(out["density"]))
    assert abs(out["grid"][ipk]) < 0.25
    assert out["density"][ipk] == pytest.approx(0.399, abs=0.06)
    with pytest.raises(ValueError):
        hrz_kde([1.0])


def test_optimal_bandwidth_formula_and_its_unknown_functional():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    ref = hrz_bandwidth_optimal(x)
    assert ref["normal_reference_used"] is True
    # the Gaussian kernel's R(K) = 1/(2 sqrt(pi)) and mu2 = 1
    assert ref["R_K"] == pytest.approx(1 / (2 * np.sqrt(np.pi)), abs=1e-4)
    assert ref["mu2_K"] == pytest.approx(1.0, abs=1e-4)
    # supplying the true functional changes the answer, showing the
    # formula really depends on it
    known = hrz_bandwidth_optimal(x, f_second_deriv_l2=ref["f2_l2"] * 32)
    assert known["normal_reference_used"] is False
    assert known["h_opt"] < ref["h_opt"]
    assert known["h_opt"] == pytest.approx(ref["h_opt"] * 32 ** (-0.2), rel=1e-9)


def test_multivariate_rate_shows_the_curse_explicitly():
    rng = np.random.default_rng(2)
    for d, expected in ((1, -2 / 5), (3, -2 / 7), (5, -2 / 9)):
        X = rng.standard_normal((200, d))
        out = hrz_kde_multivariate(X, grid=np.zeros((1, d)))
        assert out["rate_exponent"] == pytest.approx(expected)
        assert out["d"] == d
        assert out["density"][0] > 0
    with pytest.raises(ValueError):
        hrz_kde_multivariate(rng.standard_normal((200, 2)), h=[1.0, 2.0, 3.0])


def test_local_linear_beats_nw_at_the_boundary():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.uniform(0, 1, n)
    truth = lambda z: 2.0 * z          # a straight line
    y = truth(x) + rng.standard_normal(n) * 0.05
    edge = np.array([0.02, 0.05])
    ll = hrz_local_linear(x, y, grid=edge)["fitted"]
    nw = hrz_nw_regression(x, y, grid=edge)["fitted"]
    # local linear reproduces a line exactly even at the edge; NW
    # cannot, because a local CONSTANT is biased where the window is
    # one-sided
    assert np.max(np.abs(ll - truth(edge))) < np.max(np.abs(nw - truth(edge)))
    assert np.max(np.abs(ll - truth(edge))) < 0.05
    # the local slope estimates m'(x) = 2 for free
    assert np.median(hrz_local_linear(x, y)["slope"]) == pytest.approx(2.0, abs=0.3)


def test_density_derivative_uses_a_wider_bandwidth_and_has_the_right_sign():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(3000)
    d = hrz_density_derivative(x, grid=np.array([-1.0, 0.0, 1.0]))
    # phi'(z) = -z phi(z): positive left of 0, negative right of it
    assert d["derivative"][0] > 0
    assert d["derivative"][2] < 0
    assert abs(d["derivative"][1]) < 0.05
    # and the bandwidth exceeds the density-optimal one
    assert d["bandwidth"] > hrz_kde(x)["bandwidth"]
    assert d["rate_exponent"] == pytest.approx(-1 / 5)
    with pytest.raises(ValueError):
        hrz_density_derivative(x, r=0)


def test_index_regression_keeps_the_one_dimensional_rate():
    rng = np.random.default_rng(5)
    n, d = 500, 4
    X = rng.standard_normal((n, d))
    beta = np.array([1.0, -0.5, 0.25, 0.75])
    v = X @ beta
    y = np.tanh(v) + rng.standard_normal(n) * 0.1
    out = hrz_index_nw(X, y, beta)
    assert out["rate_exponent"] == pytest.approx(-0.4)  # independent of d
    assert out["d"] == 4
    # the fitted link tracks tanh
    ok = np.isfinite(out["G"])
    assert np.corrcoef(out["G"][ok], np.tanh(out["index_grid"][ok]))[0, 1] > 0.98
    with pytest.raises(ValueError):
        hrz_index_nw(X, y, beta[:2])


def test_ichimura_recovers_the_index_direction_under_its_normalisation():
    rng = np.random.default_rng(6)
    n = 400
    X = rng.standard_normal((n, 2))
    beta_true = np.array([1.0, -0.6])  # already normalised: beta[0] = 1
    y = np.tanh(X @ beta_true) + rng.standard_normal(n) * 0.15
    out = hrz_ichimura(X, y)
    assert out["beta"][0] == 1.0  # the |b1| = 1 normalisation
    assert out["beta"][1] == pytest.approx(-0.6, abs=0.3)
    assert out["root_n"] is True
    assert hrz_semiparametric_ls(X, y)["beta"][0] == 1.0
    with pytest.raises(ValueError):
        hrz_ichimura(X[:, :1], y)


def test_max_score_and_its_smoothed_version_differ_in_what_they_promise():
    rng = np.random.default_rng(7)
    n = 400
    X = rng.standard_normal((n, 2))
    beta_true = np.array([1.0, -0.8])
    # heteroskedastic error: probit would be inconsistent, max score is not
    scale = 0.5 + np.abs(X[:, 0])
    y = ((X @ beta_true + scale * rng.standard_normal(n)) > 0).astype(float)
    ms = hrz_maximum_score(X, y)
    assert ms["beta"][0] == 1.0
    assert ms["rate_exponent"] == pytest.approx(-1 / 3)
    assert ms["standard_errors_valid"] is False  # Chernoff limit
    sm = hrz_smoothed_max_score(X, y)
    assert sm["standard_errors_valid"] is True   # normality restored
    assert sm["rate_exponent"] == pytest.approx(-2 / 5)
    # Point estimates over seeds, not one draw: at n = 400 the spread
    # is sd ~0.21, so a single seed can sit 2 sd out (seed 7 gives
    # -1.44). Measured medians: n=400 -> -0.974, n=2000 -> -0.900,
    # converging on the true -0.8.
    def _sms_median(nn, reps=6):
        out = []
        for s in range(reps):
            r = np.random.default_rng(s)
            Xs = r.standard_normal((nn, 2))
            sc = 0.5 + np.abs(Xs[:, 0])
            ys = ((Xs @ beta_true + sc * r.standard_normal(nn)) > 0).astype(float)
            out.append(hrz_smoothed_max_score(Xs, ys)["beta"][1])
        return float(np.median(out))

    small, large = _sms_median(400), _sms_median(1600)
    assert abs(small - (-0.8)) < 0.4
    assert abs(large - (-0.8)) < abs(small - (-0.8))  # consistency
    with pytest.raises(ValueError):
        hrz_maximum_score(X, np.full(n, 2.0))


def test_partially_linear_is_root_n_despite_the_nonparametric_part():
    rng = np.random.default_rng(8)
    n = 600
    Z = rng.uniform(-2, 2, n)
    X = rng.standard_normal((n, 2)) + 0.3 * Z[:, None]
    beta_true = np.array([1.5, -0.7])
    y = X @ beta_true + np.sin(2 * Z) + rng.standard_normal(n) * 0.2
    out = hrz_partially_linear(X, Z, y)
    assert out["beta"] == pytest.approx(beta_true, abs=0.15)
    assert np.all(out["se"] > 0)
    assert out["root_n"] is True
    # the estimate is far better than ignoring g(Z) entirely
    naive = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.max(np.abs(out["beta"] - beta_true)) < np.max(np.abs(naive - beta_true))
    with pytest.raises(ValueError):
        hrz_partially_linear(X, Z[:10], y)


def test_series_regression_needs_K_to_grow_and_backfitting_centres_components():
    rng = np.random.default_rng(9)
    n = 500
    x = rng.uniform(0, 1, n)
    y = np.sin(3 * x) + rng.standard_normal(n) * 0.1
    lo = hrz_series_regression(x, y, K=2)
    hi = hrz_series_regression(x, y, K=8)
    assert hi["r_squared"] > lo["r_squared"]  # K=2 cannot fit a sine
    assert hi["df_ratio"] == pytest.approx(8 / n)
    with pytest.raises(ValueError):
        hrz_series_regression(x, y, K=0)
    # backfitting: components are centred, only the sum is identified
    X = rng.uniform(-1, 1, (400, 2))
    ya = 1.0 + X[:, 0] ** 2 + np.sin(3 * X[:, 1]) + rng.standard_normal(400) * 0.1
    bf = hrz_backfitting(X, ya)
    assert abs(bf["components"][:, 0].mean()) < 1e-8
    assert abs(bf["components"][:, 1].mean()) < 1e-8
    assert bf["mu"] == pytest.approx(ya.mean())
    assert bf["rate_exponent"] == pytest.approx(-0.4)
    assert np.corrcoef(bf["fitted"], ya)[0, 1] > 0.9


def test_quantile_estimators_and_the_crossing_distinction():
    rng = np.random.default_rng(10)
    n = 800
    x = rng.uniform(0, 1, n)
    y = 2 * x + rng.standard_normal(n) * (0.2 + x)  # heteroskedastic
    med = hrz_local_linear_quantile(x, y, tau=0.5, grid=np.array([0.5]))
    assert med["quantile"][0] == pytest.approx(1.0, abs=0.35)
    # the kernel-CDF route gives all tau from ONE CDF, so no crossing
    kq = hrz_kernel_quantile(x, y, tau=[0.1, 0.5, 0.9], grid=np.array([0.3, 0.7]))
    assert kq["monotone_in_tau"] is True
    assert np.all(np.diff(kq["quantile"], axis=1) >= 0)  # monotone at every x
    # spread grows with x, as the DGP dictates
    spread = kq["quantile"][:, 2] - kq["quantile"][:, 0]
    assert spread[1] > spread[0]
    # the series route fits each tau separately and says so
    sq = hrz_series_quantile(x, y, tau=0.9, K=4)
    assert "cross" in sq["crossing_warning"]
    with pytest.raises(ValueError):
        hrz_kernel_quantile(x, y, tau=[0.5, 1.5])


def test_rate_checker_measures_the_exponent():
    # a sequence decaying exactly at n^{-2/5}
    n_grid = np.array([100, 400, 1600, 6400], dtype=float)
    errs = 3.0 * n_grid ** (-0.4)
    out = check_rate(errs, n_grid, -0.4)
    assert out["observed_exponent"] == pytest.approx(-0.4, abs=1e-9)
    assert out["consistent"] is True
    # a slower sequence is correctly flagged as inconsistent
    assert check_rate(3.0 * n_grid ** (-0.2), n_grid, -0.4)["consistent"] is False
    with pytest.raises(ValueError):
        check_rate([1.0, 2.0], [10.0, 20.0], -0.4)
