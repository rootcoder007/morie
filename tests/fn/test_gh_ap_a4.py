"""Tests for gh_ap_a4.ghosal_hellinger_dist."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_ap_a4 import ghosal_hellinger_dist


def test_gh_ap_a4_basic():
    """Test basic functionality: two weight vectors give expected Hellinger^2."""
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    # Manual normalization matching the implementation's semantics.
    sp = float(sum(p))
    sq = float(sum(q))
    pn = [float(a) / sp for a in p]
    qn = [float(b) / sq for b in q]

    rho = sum(math.sqrt(a * b) for a, b in zip(pn, qn))
    expected_h2 = 1.0 - rho
    expected_tv = 0.5 * sum(abs(a - b) for a, b in zip(pn, qn))
    h = math.sqrt(max(expected_h2, 0.0))
    expected_h = h

    result = ghosal_hellinger_dist(p, q)

    # Required keys per docstring/RichResult payload.
    assert "estimate" in result
    assert "affinity" in result
    assert "inequalities_hold" in result
    assert "method" in result

    # Numeric checks (computed independently above).
    assert abs(float(result["estimate"]) - expected_h2) < 1e-12
    assert abs(float(result["affinity"]) - rho) < 1e-12

    # estimate == h^2, so sqrt(max(estimate, 0)) should equal h.
    assert abs(math.sqrt(max(float(result["estimate"]), 0.0)) - expected_h) < 1e-12

    # Documented inequalities: h^2 <= d_TV <= h*sqrt(2 - h^2).
    assert bool(result["inequalities_hold"]) is True
    assert expected_h2 <= expected_tv + 1e-12
    assert expected_tv <= h * math.sqrt(2.0 - expected_h2) + 1e-12


def test_gh_ap_a4_identical():
    """Identical distributions give Hellinger distance of 0 and affinity of 1."""
    x = np.array([0.1, 0.3, 0.6, 0.9, 1.2])
    result = ghosal_hellinger_dist(x, x)

    # rho = sum(sqrt(p*p)) = sum(p) = 1 after normalization -> h^2 = 0
    assert abs(float(result["estimate"])) < 1e-12
    assert abs(float(result["affinity"]) - 1.0) < 1e-12
    assert bool(result["inequalities_hold"]) is True


def test_gh_ap_a4_edge():
    """Single-element distributions: both degenerate -> distance 0."""
    p = np.array([42.0])
    q = np.array([7.0])
    result = ghosal_hellinger_dist(p, q)

    # Normalized: both become [1.0]; sqrt(1*1) = 1; h^2 = 0; affinity = 1.
    assert abs(float(result["estimate"])) < 1e-12
    assert abs(float(result["affinity"]) - 1.0) < 1e-12
    assert bool(result["inequalities_hold"]) is True
    assert "estimate" in result
