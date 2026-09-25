"""Tests for sobolI.sobol_indices."""

import math

import pytest

from morie.fn.sobolI import ishigami, ishigami_exact, sobol_indices


UNIFORM_PI = [lambda u: -math.pi + 2 * math.pi * u] * 3


def test_sobolI_basic():
    """Ishigami (a = 7, b = 0.1) on U(-pi, pi)^3 has S = (0.3139, 0.4424, 0)
    and ST = (0.5576, 0.4424, 0.2437) in closed form; 4096 Sobol points
    estimate them to about 5e-4."""
    ex = ishigami_exact()
    assert [round(v, 4) for v in ex["S"]] == [0.3139, 0.4424, 0.0]
    assert [round(v, 4) for v in ex["ST"]] == [0.5576, 0.4424, 0.2437]
    r = sobol_indices(ishigami, UNIFORM_PI, N=4096, d=3)
    assert max(abs(a - b) for a, b in zip(r["S"], ex["S"])) < 0.005
    assert max(abs(a - b) for a, b in zip(r["ST"], ex["ST"])) < 0.005


def test_sobolI_edge():
    """x3 acts only through its interaction with x1: S3 = 0 but ST3 > 0."""
    r = sobol_indices(ishigami, UNIFORM_PI, N=4096, d=3)
    assert abs(r["S"][2]) < 0.005 and r["ST"][2] > 0.2
    assert r["interaction"][2] == pytest.approx(r["ST"][2] - r["S"][2], rel=1e-15)


