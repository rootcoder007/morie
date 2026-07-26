"""csphr: cutting plane / normal-vector discriminant for vote classification.

Armstrong et al., Ch 5 (Unfolding Analysis of Binary Choice Data, printed
p.129); Poole (2000) for the nonparametric relative. The module implements a
pooled-covariance linear discriminant:

    w  proportional to  Sigma_pooled^-1 (mu_yea - mu_nay)
    c  =  w' (mu_yea + mu_nay) / 2
"""

import numpy as np
import pytest

from morie.fn.csphr import cutting_plane_sphere as cp


def test_csphr_one_dimension_reduces_to_the_midpoint_between_class_means():
    """The documented degenerate case, and it is exactly checkable."""
    x = np.array([-2.0, -1.0, 1.0, 2.0]).reshape(-1, 1)
    votes = np.array([0, 0, 1, 1])
    r = cp(x, votes)
    assert r["midpoint"] == pytest.approx(0.0)
    assert r["p"] == 1


def test_csphr_separates_two_well_spaced_clouds():
    rng = np.random.default_rng(3401)
    n = 60
    x = np.vstack([rng.normal(-3, 0.5, (n, 2)), rng.normal(3, 0.5, (n, 2))])
    votes = np.r_[np.zeros(n, int), np.ones(n, int)]
    assert cp(x, votes)["correct_class"] >= 2 * n - 2


def test_csphr_normal_vector_points_from_nay_toward_yea():
    """w must have a positive projection onto (mu_yea - mu_nay), or the
    hyperplane is oriented backwards and every classification inverts."""
    rng = np.random.default_rng(3407)
    n = 50
    nay = rng.normal(-2, 0.6, (n, 3))
    yea = rng.normal(2, 0.6, (n, 3))
    r = cp(np.vstack([nay, yea]), np.r_[np.zeros(n, int), np.ones(n, int)])
    w = np.asarray(r["w"]).ravel()
    assert float(w @ (yea.mean(0) - nay.mean(0))) > 0


def test_csphr_cut_sits_between_the_projected_class_means():
    rng = np.random.default_rng(3413)
    n = 40
    nay = rng.normal(-1.5, 0.7, (n, 2))
    yea = rng.normal(1.5, 0.7, (n, 2))
    r = cp(np.vstack([nay, yea]), np.r_[np.zeros(n, int), np.ones(n, int)])
    w = np.asarray(r["w"]).ravel()
    lo, hi = sorted([float(w @ nay.mean(0)), float(w @ yea.mean(0))])
    assert lo <= r["c"] <= hi


def test_csphr_reports_dimensions_and_size():
    rng = np.random.default_rng(3417)
    r = cp(rng.standard_normal((30, 4)), rng.integers(0, 2, 30))
    assert r["n"] == 30 and r["p"] == 4
    assert np.asarray(r["w"]).size == 4


def test_csphr_accepts_a_1d_vector_as_p_equals_one():
    rng = np.random.default_rng(3419)
    r = cp(rng.standard_normal(25), rng.integers(0, 2, 25))
    assert r["p"] == 1
