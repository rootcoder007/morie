"""spgwrb -- GWR bandwidth selection.

Sources: Schabenberger & Gotway Sec. 6.1.3.1 pp. 316-317 (model, hat matrix,
Cressie's sigma^2); Bivand et al. (2013) Sec. 9.4.3 p. 318 (LOO CV);
Charlton, GWR White Paper p. 8 and spgwr R/gwr.cv.R (AICc, AIC, CV score).
"""

from morie.fn import _array_core as np
import pytest

from morie.fn._schab_gwr import (aic_from_parts, aicc_from_parts, cv_score,
                                 gwr_fit, kernel_weights, pairwise_distances)
from morie.fn.spgwrb import schabenberger_gwr_bandwidth as select


def _varying(n=35, seed=5):
    """A response whose slope genuinely varies in space, so the optimum is interior."""
    rs = np.random.RandomState(seed)
    coords = np.column_stack([rs.uniform(0, 10, n), rs.uniform(0, 10, n)])
    X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
    beta = 1.0 + 0.8 * np.sin(0.9 * coords[:, 0])
    y = X[:, 0] + beta * X[:, 1] + 0.05 * rs.standard_normal(n)
    return X, y, coords


def _constant(n=35, seed=5):
    """Globally constant coefficients: the optimal bandwidth is genuinely infinite."""
    rs = np.random.RandomState(seed)
    coords = np.column_stack([rs.uniform(0, 10, n), rs.uniform(0, 10, n)])
    X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
    y = X @ np.array([1.0, 2.0]) + 0.05 * rs.standard_normal(n)
    return X, y, coords


# --------------------------------------------------------- published output
# spgwr's NY8 example as printed in Bivand, Pebesma & Gomez-Rubio, Applied
# Spatial Data Analysis with R (Use R!), Sec. 10.5.3.
NY8_N, NY8_RSS, NY8_AICC, NY8_AIC = 281, 119.6, 568.0, 561.6


def test_aic_and_aicc_reproduce_spgwrs_published_ny8_output():
    """One unknown, two published numbers -- this pins both formulas at once."""
    sigma2 = NY8_RSS / NY8_N
    base = (2 * NY8_N * np.log(np.sqrt(sigma2)) + NY8_N * np.log(2 * np.pi)
            + NY8_N)
    tr_S = NY8_AIC - base                       # AIC = base + tr(S)
    assert 3.5 < tr_S < 5.0                     # printed as 4.4 effective parameters
    assert aic_from_parts(NY8_N, sigma2, tr_S) == pytest.approx(NY8_AIC)
    assert aicc_from_parts(NY8_N, sigma2, tr_S) == pytest.approx(NY8_AICC, abs=0.5)


def test_aicc_exceeds_aic_and_the_gap_closes_as_n_grows():
    """AICc is AIC plus a small-sample penalty; the two agree asymptotically."""
    small = aicc_from_parts(30, 1.0, 6.0) - aic_from_parts(30, 1.0, 6.0)
    large = aicc_from_parts(30000, 1.0, 6.0) - aic_from_parts(30000, 1.0, 6.0)
    assert small > 0
    assert large > 0
    assert small / 30 > large / 30000


def test_aicc_is_infinite_when_the_model_spends_every_degree_of_freedom():
    assert np.isinf(aicc_from_parts(10, 1.0, 8.0))
    assert np.isinf(aicc_from_parts(10, 1.0, 20.0))


# ------------------------------------------------------------- the criteria
def test_cv_is_leave_one_out_not_in_sample():
    X, y, coords = _varying()
    D = pairwise_distances(coords)
    bw = 2.0
    cv = cv_score(y, X, D, bw)
    manual = 0.0
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        w = kernel_weights(D[i][keep], bw, "gaussian")
        W = np.diag(w)
        b = np.linalg.solve(X[keep].T @ W @ X[keep], X[keep].T @ W @ y[keep])
        manual += float((y[i] - X[i] @ b) ** 2)
    assert cv == pytest.approx(manual, rel=1e-9)


def test_cv_exceeds_the_in_sample_rss():
    X, y, coords = _varying()
    D = pairwise_distances(coords)
    fit = gwr_fit(y, X, D, 2.0)
    assert cv_score(y, X, D, 2.0) > float(np.sum(fit["resid"] ** 2))


def test_hat_matrix_produces_the_fitted_values():
    X, y, coords = _varying()
    fit = gwr_fit(y, X, pairwise_distances(coords), 2.0)
    assert np.allclose(fit["fitted"], fit["S"] @ y)


def test_book_and_ml_variance_estimates_differ_as_the_sources_say():
    """Cressie p. 317 divides by tr{(I-L)(I-L)'}; the AICc wants n."""
    X, y, coords = _varying()
    fit = gwr_fit(y, X, pairwise_distances(coords), 2.0)
    assert fit["sigma2"] == pytest.approx(fit["rss"] / fit["n"])
    assert fit["sigma2_cressie"] > fit["sigma2"]


def test_effective_parameters_is_two_tr_s_minus_tr_sts():
    X, y, coords = _varying()
    fit = gwr_fit(y, X, pairwise_distances(coords), 2.0)
    assert fit["effective_parameters"] == pytest.approx(
        2 * fit["tr_S"] - fit["tr_STS"])


def test_wide_bandwidth_collapses_gwr_onto_global_ols():
    X, y, coords = _constant()
    fit = gwr_fit(y, X, pairwise_distances(coords), 1e7)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(fit["params"], np.tile(ols, (len(y), 1)), atol=1e-5)
    assert fit["tr_S"] == pytest.approx(X.shape[1], abs=1e-6)


def test_narrow_bandwidth_spends_more_degrees_of_freedom():
    X, y, coords = _varying()
    D = pairwise_distances(coords)
    assert gwr_fit(y, X, D, 1.0)["tr_S"] > gwr_fit(y, X, D, 4.0)["tr_S"]


# ---------------------------------------------------------------- selection
def test_selected_bandwidth_is_interior_when_the_process_really_varies():
    X, y, coords = _varying()
    r = select(X, y, coords, criterion="cv")
    lo, hi = r["bounds"]
    assert lo * 1.5 < r["optimal_bandwidth"] < hi * 0.95


def test_selected_bandwidth_runs_to_the_top_when_the_process_does_not_vary():
    """A constant-coefficient process has no local scale; the criterion says so."""
    X, y, coords = _constant()
    r = select(X, y, coords, criterion="cv")
    assert r["optimal_bandwidth"] > 0.9 * r["bounds"][1]


def test_search_interval_is_the_bounding_box_diagonal_over_a_thousand():
    X, y, coords = _varying()
    lo, hi = select(X, y, coords)["bounds"]
    span = coords.max(axis=0) - coords.min(axis=0)
    assert hi == pytest.approx(float(np.sqrt(np.sum(span ** 2))))
    assert lo == pytest.approx(hi / 1000.0)


def test_golden_section_agrees_with_a_brute_force_grid():
    X, y, coords = _varying()
    D = pairwise_distances(coords)
    r = select(X, y, coords, criterion="cv")
    lo, hi = r["bounds"]
    grid = np.linspace(lo, hi, 250)
    brute = grid[int(np.argmin([cv_score(y, X, D, b) for b in grid]))]
    assert abs(r["optimal_bandwidth"] - brute) < 0.05 * (hi - lo)


def test_reported_score_is_the_objective_at_the_reported_bandwidth():
    X, y, coords = _varying()
    r = select(X, y, coords, criterion="cv")
    D = pairwise_distances(coords)
    assert r["score"] == pytest.approx(
        cv_score(y, X, D, r["optimal_bandwidth"]), rel=1e-9)


def test_cv_and_aicc_disagree_but_land_in_the_same_neighbourhood():
    X, y, coords = _varying()
    a = select(X, y, coords, criterion="cv")["optimal_bandwidth"]
    b = select(X, y, coords, criterion="aicc")["optimal_bandwidth"]
    assert a != b
    assert 0.2 < a / b < 5.0


def test_adaptive_selection_returns_an_integer_neighbour_count():
    X, y, coords = _varying()
    r = select(X, y, coords, criterion="aicc", adaptive=True)
    assert isinstance(r["optimal_bandwidth"], int)
    assert 2 <= r["optimal_bandwidth"] < len(y)
    assert r["adaptive"] is True


def test_every_kernel_selects_a_bandwidth():
    X, y, coords = _varying()
    for k in ("gaussian", "bisquare", "tricube", "boxcar"):
        r = select(X, y, coords, kernel=k, criterion="cv")
        assert np.isfinite(r["optimal_bandwidth"]) and r["optimal_bandwidth"] > 0
        assert r["kernel"] == k


def test_explicit_bounds_are_honoured():
    X, y, coords = _varying()
    r = select(X, y, coords, bounds=(0.5, 3.0))
    assert r["bounds"] == (0.5, 3.0)
    assert 0.5 <= r["optimal_bandwidth"] <= 3.0


def test_payload_reports_every_criterion_at_the_chosen_bandwidth():
    X, y, coords = _varying()
    r = select(X, y, coords, criterion="aicc")
    assert r["aicc"] == pytest.approx(
        aicc_from_parts(r["n"], r["sigma2"], r["tr_S"]))
    assert r["aic"] == pytest.approx(
        aic_from_parts(r["n"], r["sigma2"], r["tr_S"]))
    assert r["score"] == pytest.approx(r["aicc"], rel=1e-9)


@pytest.mark.parametrize("bad", ["mse", "gcv", "bic", ""])
def test_unknown_criterion_rejected(bad):
    X, y, coords = _varying()
    with pytest.raises(ValueError):
        select(X, y, coords, criterion=bad)


def test_unknown_kernel_rejected():
    X, y, coords = _varying()
    with pytest.raises(ValueError):
        select(X, y, coords, kernel="epanechnikov")


def test_mismatched_shapes_rejected():
    X, y, coords = _varying()
    with pytest.raises(ValueError):
        select(X, y[:-1], coords)
