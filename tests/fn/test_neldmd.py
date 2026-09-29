"""Tests for neldmd.nelder_mead (Lagarias et al. 1998 rules)."""

import pytest

from morie.fn.neldmd import nelder_mead


def test_nelder_mead_minimises_a_shifted_quadratic():
    f = lambda x: (x[0] - 1.5) ** 2 + 3 * (x[1] + 0.5) ** 2 + x[0] * x[1] / 10  # noqa: E731
    # gradient zero: [[2, 0.1], [0.1, 6]] [x, y]' = [3, -3]'
    det = 2 * 6 - 0.1 * 0.1
    x0 = (6 * 3 - 0.1 * -3) / det
    x1 = (2 * -3 - 0.1 * 3) / det
    r = nelder_mead(f, [0.0, 0.0], xtol=1e-12, ftol=1e-14, max_iter=2000)
    assert r["converged"]
    assert r["x"] == pytest.approx([x0, x1], abs=1e-6)


def test_first_reflection_of_the_initial_simplex():
    """One iteration from x0 with steps 0.1: the worst vertex is reflected
    through the centroid of the other two (then expanded or accepted)."""
    f = lambda x: x[0] + 2 * x[1]  # noqa: E731
    r = nelder_mead(f, [0.0, 0.0], max_iter=1)
    # vertices (0,0), (0.1,0), (0,0.1); worst (0,0.1); centroid (0.05, 0);
    # reflection (0.1,-0.1) f=-0.1 < best 0 -> expansion (0.15,-0.2) f=-0.25 accepted
    assert [0.15, -0.2] in [[round(v, 12) for v in p] for p in r["simplex"]]
