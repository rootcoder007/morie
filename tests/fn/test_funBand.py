"""Tests for morie.fn.funBand -- Wahba's Bayesian smoothing-spline bands.

The anchors are algebraic properties of the influence matrix and the
paper's own printed formulas, not comparisons of the module with itself.
"""

import math

import pytest

from morie.fn.funBand import funBand, influence_matrix, gcv_score


N = 12
X = [i / 12.0 for i in range(1, N + 1)]
Y = [0.4, 1.1, 0.9, 2.2, 1.8, 3.1, 2.6, 4.0, 3.5, 4.9, 4.2, 5.6]


def _ols_line(x, y):
    n = float(len(x))
    sx, sy = sum(x), sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(x[i] * y[i] for i in range(len(x)))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - b * sx) / n, b


@pytest.mark.parametrize("lam", [1e-6, 1.0, 1e6, 1e15])
def test_influence_matrix_reproduces_straight_lines(lam):
    """K annihilates constants and linear terms, so A(lambda) fixes any
    straight line at EVERY lambda. This is what the null-space
    thresholding in the spectral solve exists to preserve."""
    A = influence_matrix(X, lam)
    assert max(abs(sum(row) - 1.0) for row in A) < 1e-11
    lin = [sum(A[i][j] * X[j] for j in range(N)) for i in range(N)]
    assert max(abs(lin[i] - X[i]) for i in range(N)) < 1e-11


def test_trace_falls_from_n_to_exactly_two():
    """Tr A is the equivalent degrees of freedom for signal: n when the
    spline interpolates, exactly 2 when it is the least-squares line."""
    assert sum(influence_matrix(X, 1e-12)[i][i]
               for i in range(N)) == pytest.approx(float(N), abs=1e-5)
    for lam in (1e6, 1e12, 1e18):
        tr = sum(influence_matrix(X, lam)[i][i] for i in range(N))
        assert tr == pytest.approx(2.0, abs=1e-6)


def test_influence_matrix_is_symmetric():
    A = influence_matrix(X, 0.5)
    assert max(abs(A[i][j] - A[j][i])
               for i in range(N) for j in range(N)) < 1e-12


def test_large_lambda_is_the_least_squares_line():
    a, b = _ols_line(X, Y)
    r = funBand(Y, x=X, lam=1e16)
    assert max(abs(r["fitted"][i] - (a + b * X[i]))
               for i in range(N)) < 1e-9


def test_small_lambda_interpolates():
    r = funBand(Y, x=X, lam=1e-14)
    assert max(abs(r["fitted"][i] - Y[i]) for i in range(N)) < 1e-6


def test_degrees_of_freedom_partition_n():
    """Tr A + Tr(I - A) = n identically."""
    r = funBand(Y, x=X)
    assert r["edf_signal"] + r["edf_error"] == pytest.approx(float(N),
                                                             abs=1e-10)


def test_sigma2_is_rss_over_residual_edf():
    """The paper's estimator: sigma_hat^2 = RSS(lambda)/n(1 - a(lambda))."""
    r = funBand(Y, x=X)
    assert r["sigma2"] == pytest.approx(r["rss"] / r["edf_error"], rel=1e-12)


def test_gcv_matches_equation_2_16_recomputed_by_hand():
    r = funBand(Y, x=X)
    A = influence_matrix(X, r["lambda"])
    fit = [sum(A[i][j] * Y[j] for j in range(N)) for i in range(N)]
    rss = sum((Y[i] - fit[i]) ** 2 for i in range(N))
    tr_ia = float(N) - sum(A[i][i] for i in range(N))
    want = (rss / N) / ((tr_ia / N) ** 2)
    assert r["gcv"] == pytest.approx(want, rel=1e-12)


def test_band_is_the_theorem_1_expression():
    """Theorem 1 gives cov = sigma^2 A, so the half width at t_i is
    z * sigma_hat * sqrt(a_ii) -- the DIAGONAL of the influence matrix."""
    r = funBand(Y, x=X)
    for i in range(N):
        want = r["multiplier"] * r["sigma"] * math.sqrt(r["diag_A"][i])
        assert r["half_width"][i] == pytest.approx(want, rel=1e-12)
        assert r["upper"][i] - r["lower"][i] == pytest.approx(2.0 * want,
                                                              rel=1e-12)


def test_posterior_variance_is_sigma2_times_diag_A():
    r = funBand(Y, x=X)
    for i in range(N):
        assert r["posterior_variance"][i] == pytest.approx(
            r["sigma2"] * r["diag_A"][i], rel=1e-12)


def test_normal_multiplier_is_1_96_at_the_paper_level():
    """The paper's Monte Carlo used 1.96; quantile='normal' reproduces it."""
    r = funBand(Y, x=X, quantile="normal")
    assert r["multiplier"] == pytest.approx(1.959963985, rel=1e-8)


def test_t_multiplier_exceeds_the_normal_one():
    """The paper notes the t point on EDF degrees of freedom would have
    improved small-n coverage; it must therefore be the wider one."""
    rn = funBand(Y, x=X, lam=0.01, quantile="normal")
    rt = funBand(Y, x=X, lam=0.01, quantile="t")
    assert rt["multiplier"] > rn["multiplier"]


def test_coverage_is_reported_across_the_function():
    """With the truth supplied, coverage is the fraction of the n
    intervals containing it -- an across-the-function rate."""
    truth = [0.5 * v + 0.2 for v in X]
    r = funBand(Y, x=X, truth=truth)
    assert 0.0 <= r["coverage"] <= 1.0
    manual = sum(1 for i in range(N)
                 if r["lower"][i] <= truth[i] <= r["upper"][i]) / float(N)
    assert r["coverage"] == pytest.approx(manual, abs=1e-12)


def test_wider_bands_when_sigma_is_larger():
    noisy = [Y[i] + (1.0 if i % 2 else -1.0) for i in range(N)]
    r1 = funBand(Y, x=X, lam=0.01)
    r2 = funBand(noisy, x=X, lam=0.01)
    assert r2["sigma"] > r1["sigma"]
    assert r2["half_width"][0] > r1["half_width"][0]


def test_inputs_are_validated():
    with pytest.raises(ValueError, match="four"):
        funBand([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="alpha"):
        funBand(Y, alpha=1.5)
    with pytest.raises(ValueError, match="increasing"):
        funBand(Y, x=[0.1, 0.3, 0.2] + [0.4 + 0.1 * i for i in range(9)])
    with pytest.raises(ValueError, match="quantile"):
        funBand(Y, quantile="bootstrap")
