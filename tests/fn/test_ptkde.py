"""Tests for morie.fn.ptkde: bivariate Gaussian kernel density."""

import math

import pytest

from morie.fn.ptkde import spat, spatial_kde, spatialkde

PTS = [
    (0.12, 0.33),
    (0.47, 0.81),
    (0.55, 0.42),
    (0.91, 0.66),
    (0.31, 0.59),
    (0.72, 0.15),
    (0.05, 0.94),
    (0.66, 0.71),
    (0.38, 0.27),
    (0.83, 0.52),
    (0.24, 0.08),
    (0.59, 0.99),
    (0.97, 0.36),
    (0.44, 0.63),
    (0.18, 0.47),
    (0.76, 0.88),
    (0.29, 0.21),
    (0.63, 0.55),
    (0.02, 0.73),
    (0.88, 0.12),
    (0.51, 0.35),
]


def _density(u, v, hx, hy):
    s = math.fsum(math.exp(-0.5 * ((u - x) / hx) ** 2 - 0.5 * ((v - y) / hy) ** 2) / (2 * math.pi) for x, y in PTS)
    return s / (len(PTS) * hx * hy)


def _sd(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1))


def test_density_formula_on_grid():
    r = spatial_kde(PTS, h=(0.1, 0.15), n=(6, 4), lims=(0, 1, 0, 1))
    assert len(r.z) == 6 and len(r.z[0]) == 4
    for i, u in enumerate(r.x):
        for j, v in enumerate(r.y):
            assert r.z[i][j] == pytest.approx(_density(u, v, 0.1, 0.15), rel=1e-12)


def test_bandwidth_rules():
    xs = sorted(p[0] for p in PTS)
    n = len(xs)
    # type-7 quartiles: index (n-1)p = 5 and 15 are integers for n = 21
    iqr = (xs[15] - xs[5]) / 1.34
    assert spatial_kde(PTS).bandwidth[0] == pytest.approx(1.06 * min(_sd(xs), iqr) * n**-0.2, rel=1e-13)
    ys = [p[1] for p in PTS]
    assert spatial_kde(PTS, method="scott").bandwidth[1] == pytest.approx(_sd(ys) * n ** (-1 / 6), rel=1e-13)
    assert spatial_kde(PTS, method="default").z == spatial_kde(PTS).z


def test_integrates_to_one_on_wide_grid():
    r = spatial_kde(PTS, n=121, lims=(-1, 2, -1, 2))
    dx, dy = r.x[1] - r.x[0], r.y[1] - r.y[0]
    assert math.fsum(math.fsum(row) for row in r.z) * dx * dy == pytest.approx(1.0, abs=1e-6)


def test_aliases_and_errors():
    assert spat is spatial_kde and spatialkde is spatial_kde
    with pytest.raises(ValueError):
        spatial_kde(PTS, method="bogus")
    with pytest.raises(ValueError):
        spatial_kde(PTS[:1])
