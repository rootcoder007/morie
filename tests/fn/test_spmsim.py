"""spmsim -- multiscale GWR.

Algorithm source: mgwr (Oshan, Li, Kang, Wolf & Fotheringham), mgwr/search.py
``multi_bw`` -- the MGWR authors' own implementation.  The paper
(Fotheringham, Yang & Kang 2017) is paywalled and was not read.
"""

import numpy as np
import pytest

from morie.fn._schab_gwr import gwr_fit, pairwise_distances, select_bandwidth
from morie.fn.spgwrb import schabenberger_gwr_bandwidth as select
from morie.fn.spmsim import schabenberger_mgwr_bandwidth as mgwr


def _two_scales(n=35, seed=13):
    """Intercept drifting slowly, slope oscillating fast: two genuine scales."""
    rs = np.random.RandomState(seed)
    coords = np.column_stack([rs.uniform(0, 10, n), rs.uniform(0, 10, n)])
    X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
    b0 = 1.0 + 0.05 * coords[:, 0]
    b1 = np.sin(coords[:, 0])
    y = b0 + b1 * X[:, 1] + 0.05 * rs.standard_normal(n)
    return X, y, coords


def _one_scale(n=35, seed=13):
    rs = np.random.RandomState(seed)
    coords = np.column_stack([rs.uniform(0, 10, n), rs.uniform(0, 10, n)])
    X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
    y = X @ np.array([1.0, 2.0]) + 0.05 * rs.standard_normal(n)
    return X, y, coords


def test_one_bandwidth_per_covariate():
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, max_iter=40)
    assert np.shape(r["bandwidths"]) == (X.shape[1],)
    assert np.all(np.asarray(r["bandwidths"]) > 0)


def test_covariates_at_different_scales_get_different_bandwidths():
    """This is the entire claim MGWR makes over single-bandwidth GWR.

    Stated as a majority over draws, not as a law. Backfitting minimises
    nothing globally and its SOC criterion can stop at the starting point
    (see test_soc_can_stop_before_finding_any_scale_separation), so on an
    individual sample the separation may not appear. Measured over eight
    seeds of this fixture the intercept took the wider kernel in six.
    """
    hits = 0
    for seed in (5, 11, 13, 17, 23, 29, 31, 37):
        X, y, coords = _two_scales(seed=seed)
        bws = mgwr(X, y, coords, max_iter=40)["bandwidths"]
        hits += bws[0] > bws[1]
    assert hits >= 6


def test_soc_can_stop_before_finding_any_scale_separation():
    """The documented failure mode, pinned so it cannot regress silently.

    Both scores measure how much the fit MOVED. When the initial
    single-bandwidth GWR already sits at the wide end of the interval, the
    first sweep leaves it there, the score is tiny and the loop reports
    convergence after two or three sweeps with no separation found. The
    reference implementation shares this; `at_search_boundary` flags it.
    """
    X, y, coords = _two_scales(seed=29)
    r = mgwr(X, y, coords, max_iter=40)
    assert r["converged"] is True
    assert r["n_iter"] <= 3
    assert r["at_search_boundary"] is True
    bws = np.asarray(r["bandwidths"])
    assert bws.max() / bws.min() < 1.05          # no separation at all


def test_a_healthy_fit_is_not_flagged_at_the_boundary():
    X, y, coords = _two_scales(seed=5)
    r = mgwr(X, y, coords, max_iter=40)
    assert r["at_search_boundary"] is False
    assert r["n_iter"] > 3


def test_backfitting_converges_and_reports_it():
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, tol=1e-4, max_iter=40)
    assert r["converged"] is True
    assert r["score_history"][-1] < 1e-4
    assert r["n_iter"] == len(r["score_history"])
    assert "warning" not in r


def test_non_convergence_is_reported_not_hidden():
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, tol=1e-30, max_iter=2)
    assert r["converged"] is False
    assert "warning" in r
    assert r["n_iter"] == 2


def _standardize(X):
    Xs = np.asarray(X, dtype=float).copy()
    nz = Xs.std(axis=0, ddof=0) > 0
    Xs[:, nz] = (Xs[:, nz] - Xs[:, nz].mean(axis=0)) / Xs[:, nz].std(axis=0, ddof=0)
    return Xs


def test_fitted_values_are_the_sum_of_local_terms():
    """With standardization on, the identity holds through the scaling."""
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, max_iter=40)
    inner = np.sum(np.asarray(r["local_coefficients"]) * _standardize(X), axis=1)
    assert np.allclose(r["fitted"], inner * r["y_scale"] + r["y_centre"])
    assert np.allclose(r["resid"], y - r["fitted"])
    assert r["rss"] == pytest.approx(float(np.sum(np.asarray(r["resid"]) ** 2)))


def test_standardization_is_on_by_default():
    """2024 book Sec. 2.3.3.2 and Sec. 6.3: a default that must be turned off."""
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, max_iter=40)
    assert r["standardized"] is True
    assert r["y_scale"] == pytest.approx(y.std(ddof=0))
    assert r["y_centre"] == pytest.approx(y.mean())
    # the intercept column has no variance, so it is left alone
    assert r["x_scale"][0] == 1.0
    assert r["x_centre"][0] == 0.0
    assert r["x_scale"][1] == pytest.approx(X[:, 1].std(ddof=0))


def test_standardization_can_be_turned_off():
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, max_iter=40, standardize=False)
    assert r["standardized"] is False
    assert r["y_scale"] == 1.0 and r["y_centre"] == 0.0
    assert np.allclose(r["fitted"],
                       np.sum(np.asarray(r["local_coefficients"]) * X, axis=1))


def test_fitted_and_resid_stay_in_the_units_of_y_either_way():
    """Whatever standardize does internally, the outputs are comparable."""
    X, y, coords = _two_scales()
    a = mgwr(X, y, coords, max_iter=40, standardize=True)
    b = mgwr(X, y, coords, max_iter=40, standardize=False)
    for r in (a, b):
        assert np.allclose(r["resid"], y - r["fitted"])
        assert float(np.mean(np.abs(r["fitted"]))) > 0.1 * float(np.mean(np.abs(y)))


def test_local_coefficients_have_one_column_per_covariate():
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, max_iter=40)
    assert np.shape(r["local_coefficients"]) == X.shape


def test_mgwr_usually_fits_better_than_the_gwr_it_started_from():
    """Not a theorem -- bandwidths are chosen per covariate by AICc, not to
    minimise total RSS, and backfitting has no global optimality guarantee.
    Measured: MGWR beat single-bandwidth GWR on 7 of 8 seeds, the exception
    being the boundary case above."""
    wins = 0
    for seed in (5, 11, 13, 17, 23, 29, 31, 37):
        X, y, coords = _two_scales(seed=seed)
        r = mgwr(X, y, coords, max_iter=40)
        gwr_rss = float(np.sum(
            gwr_fit(y, X, pairwise_distances(coords), r["bandwidth_gwr"])["resid"] ** 2))
        wins += r["rss"] < gwr_rss
    assert wins >= 6


def test_single_covariate_reduces_to_ordinary_gwr():
    """With k = 1 there is nothing to backfit; the bandwidth must match.

    Compared with standardization off on both sides: centring a lone
    covariate is not a no-op when the inner GWR has no intercept to absorb it.
    """
    X, y, coords = _two_scales()
    x1 = X[:, [1]]
    r = mgwr(x1, y, coords, tol=1e-6, max_iter=40, standardize=False)
    single = select_bandwidth(y, x1, coords, criterion="aicc")["bandwidth"]
    assert r["bandwidths"][0] == pytest.approx(single, rel=1e-9)


def test_a_single_scale_process_gets_similar_bandwidths():
    """The converse check: no artificial scale separation where none exists."""
    X, y, coords = _one_scale()
    bws = np.asarray(mgwr(X, y, coords, max_iter=40)["bandwidths"])
    assert bws.max() / bws.min() < 3.0


def test_soc_rss_is_a_different_score_but_a_comparable_answer():
    X, y, coords = _two_scales()
    f = mgwr(X, y, coords, max_iter=40, rss_score=False)
    g = mgwr(X, y, coords, max_iter=40, rss_score=True)
    assert f["score_type"] == "SOC-f"
    assert g["score_type"] == "SOC-RSS"
    assert f["score_history"] != g["score_history"]
    assert np.allclose(f["bandwidths"], g["bandwidths"], rtol=0.5)


def test_score_history_is_the_convergence_trace():
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, tol=1e-4, max_iter=40)
    assert len(r["bandwidth_history"]) == r["n_iter"]
    assert np.allclose(r["bandwidth_history"][-1], r["bandwidths"])
    assert r["score_history"][-1] < r["score_history"][0]


def test_init_bandwidth_skips_the_initial_search():
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, init_bandwidth=3.0, max_iter=40)
    assert r["bandwidth_gwr"] == 3.0


def test_frozen_search_after_repeated_identical_bandwidths():
    """bws_same_times stops re-searching once the vector settles."""
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, max_iter=40, bws_same_times=1)
    assert r["converged"] is True


def test_every_kernel_runs():
    X, y, coords = _two_scales()
    for k in ("gaussian", "bisquare", "tricube"):
        r = mgwr(X, y, coords, kernel=k, max_iter=25)
        assert np.all(np.isfinite(np.asarray(r["bandwidths"])))
        assert r["kernel"] == k


def test_cv_criterion_also_works_inside_the_backfit():
    X, y, coords = _two_scales()
    r = mgwr(X, y, coords, criterion="cv", max_iter=25)
    assert r["criterion"] == "cv"
    assert np.all(np.asarray(r["bandwidths"]) > 0)


def test_mgwr_beats_gwr_on_a_draw_where_backfitting_separates_the_scales():
    """The comparison that justifies the extra machinery, on a draw where the
    backfitting does not stall at the boundary."""
    X, y, coords = _two_scales(seed=5)
    r = mgwr(X, y, coords, max_iter=40)
    assert r["at_search_boundary"] is False
    gwr_bw = select(X, y, coords, criterion="aicc")["optimal_bandwidth"]
    gwr_rss = float(np.sum(
        gwr_fit(y, X, pairwise_distances(coords), gwr_bw)["resid"] ** 2))
    assert r["rss"] < gwr_rss


def test_mismatched_shapes_rejected():
    X, y, coords = _two_scales()
    with pytest.raises(ValueError):
        mgwr(X, y[:-1], coords)


def test_unknown_kernel_rejected():
    X, y, coords = _two_scales()
    with pytest.raises(ValueError):
        mgwr(X, y, coords, kernel="quartic")
