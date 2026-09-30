import math

import pytest

from morie.fn.research_concentration import (
    concentration_decompose,
    concentration_dispersion,
    concentration_distinct_growth,
    concentration_gini,
)

X = [int(abs(math.sin(1.3 * i)) * 7 * (i % 3 == 0)) for i in range(61)]


def _gini_double_sum(x):
    n = len(x)
    return math.fsum(abs(a - b) for a in x for b in x) / (2 * n * n * (math.fsum(x) / n))


def test_gini_equals_the_mean_absolute_difference_form():
    for x in (X, [0, 0, 0, 1, 9], [3, 1, 4, 1, 5, 9, 2, 6], [5, 5, 5]):
        assert concentration_gini(x) == pytest.approx(_gini_double_sum(x), abs=1e-12)
    with pytest.raises(ValueError, match="positive total"):
        concentration_gini([0, 0])


def test_zero_decomposition_identity_and_poisson_null():
    d = concentration_decompose(X)
    z = sum(1 for v in X if v == 0) / len(X)
    pos = [v for v in X if v > 0]
    mu = math.fsum(X) / len(X)
    assert d.zero_share == z
    assert d.gini_positive == pytest.approx(_gini_double_sum(pos), abs=1e-12)
    assert d.gini_all == pytest.approx(z + (1 - z) * d.gini_positive, abs=1e-12)
    assert abs(d.identity_check) < 1e-12
    assert d.null_zero_share == pytest.approx(math.exp(-mu), abs=1e-15)
    assert d.excess_zero_share == pytest.approx(z - math.exp(-mu), abs=1e-15)
    assert d.null_gini_same_positive == pytest.approx(math.exp(-mu) + (1 - math.exp(-mu)) * d.gini_positive, abs=1e-15)


def test_dispersion_matches_sample_moments():
    d = concentration_dispersion(X)
    n = len(X)
    mu = math.fsum(X) / n
    var = math.fsum((v - mu) ** 2 for v in X) / (n - 1)
    assert d.variance == pytest.approx(var, abs=1e-12)
    assert d.dispersion_index == pytest.approx(var / mu, abs=1e-12)
    assert d.implied_intensity_sd == pytest.approx(math.sqrt(max(var - mu, 0)), abs=1e-12)
    with pytest.raises(ValueError, match="at least two"):
        concentration_dispersion([3])


def test_distinct_growth_root_envelope_and_slope():
    place = [int(abs(math.sin(0.37 * i * i)) * (3 + i // 10)) for i in range(300)]
    g = concentration_distinct_growth(place)
    seen = []
    for p in place:
        if p not in seen:
            seen.append(p)
    assert g.distinct[-1] == len(seen)
    M = g.M_hat
    K = g.distinct[-1]
    assert M * math.fsum(1 / (M + k) for k in range(300)) == pytest.approx(K, abs=1e-9)
    assert g.expected[-1] == pytest.approx(K, abs=1e-9)
    for i in (1, 50, 300):
        assert g.lower[i - 1] == pytest.approx(M * math.log((M + i) / M), abs=1e-12)
        assert g.lower[i - 1] <= g.expected[i - 1] + 1e-9 <= g.upper[i - 1] + 2e-9
    half = [i for i in range(1, 301) if i >= 150]
    lx = [math.log(i) for i in half]
    ly = [math.log(g.distinct[i - 1]) for i in half]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
    assert g.loglog_slope == pytest.approx(slope, abs=1e-9)
    # a power-law series (every third event a new place) has slope near 1
    assert concentration_distinct_growth([i // 3 for i in range(2100)]).loglog_slope > 0.9
    assert concentration_distinct_growth([1, 2, 3]).M_hat == math.inf
    assert concentration_distinct_growth([1, 1, 1]).M_hat == 0.0
    with pytest.raises(ValueError, match="at least two"):
        concentration_distinct_growth([1])
