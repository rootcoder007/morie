"""compdir: circular statistics against circular 0.5 (mean, rho, var, sd, rayleigh.test, mle.vonmises, dvonmises);
compositional summaries checked against their definitions (Aitchison 1986).
"""

import math

import pytest

from morie.fn._rng import random_uniform
from morie.fn.compdir import (
    aitchison_biplot,
    aitchison_clr_covariance,
    circular_summary,
    compositional_mad,
    compositional_pielou,
    compositional_quantile_dist,
    dirichlet_fit_mom,
    dirichlet_sample,
    vonmises_mle,
)

CIRC = [0.65029443787840968, 0.62685675813122499, 0.37314324186877501, 0.96647526650226123, 2.2645672747835529e-06]
VM = [
    0.65029443787840957,
    1.6237033656887356,
    0.18119917112627371,
    0.38578422618001035,
    1.5826449629808073,
    -41.832293212969851,
]

_u = [float(v) for v in random_uniform(100, seed=73, stream=0)]
TH = [(0.8 + 1.2 * (_u[i] - 0.5) + (math.pi if i % 7 == 0 else 0)) % (2 * math.pi) for i in range(30)]
X = [[1, 2, 7, 3], [2, 2, 6, 1], [3, 1, 6, 2], [1.5, 2.5, 4, 2], [2.2, 0.8, 5, 3], [0.9, 3.1, 4.4, 1.6]]


def test_circular_summary_and_vonmises_equal_circular():
    s = circular_summary(TH)
    assert [s.mean, s.rbar, s.variance, s.sd, s.rayleigh_p] == pytest.approx(CIRC, rel=1e-12, abs=1e-15)
    m = vonmises_mle(TH)
    b = vonmises_mle(TH, bias=True)
    assert [m.mu, m.kappa, m.se_mu, m.se_kappa, b.kappa, m.loglik] == pytest.approx(VM, rel=1e-12)


def _clr(x):
    s = sum(x)
    lg = [math.log(v / s) for v in x]
    m = sum(lg) / len(lg)
    return [v - m for v in lg]


def test_clr_covariance_mad_pielou_quantile_by_definition():
    s = aitchison_clr_covariance(X)
    C = [_clr(r) for r in X]
    mu = [sum(r[j] for r in C) / 6 for j in range(4)]
    assert s.clr_cov[1][2] == pytest.approx(sum((r[1] - mu[1]) * (r[2] - mu[2]) for r in C) / 5, abs=1e-15)
    # total variance = trace of the clr covariance = sum of the variation matrix / (2 D)
    assert s.total_variance == pytest.approx(sum(v for r in s.variation for v in r) / 8, abs=1e-12)
    assert sum(s.center) == pytest.approx(1.0)
    assert compositional_mad([[1, 2, 4], [2, 2, 2], [4, 2, 1]]) == pytest.approx(
        [math.log(2), 0.0, math.log(2)], abs=1e-15
    )
    p = [0.25, 0.25, 0.5]
    assert compositional_pielou([1, 1, 2]) == pytest.approx(-sum(v * math.log(v) for v in p) / math.log(3), abs=1e-15)
    assert compositional_pielou([3, 3, 3]) == pytest.approx(1.0, abs=1e-15)
    q = compositional_quantile_dist([[1, 2, 4], [2, 2, 2], [4, 2, 1], [3, 1, 1]])
    q1, q3 = [1.75, 1.75, 1.0], [3.25, 2.0, 2.5]
    assert q == pytest.approx(math.sqrt(sum((a - b) ** 2 for a, b in zip(_clr(q3), _clr(q1)))), abs=1e-15)


def test_biplot_and_dirichlet():
    b = aitchison_biplot(X)
    assert sum(b.explained) == pytest.approx(1.0) and b.singular_values[-1] == pytest.approx(0.0, abs=1e-12)
    d = dirichlet_fit_mom([[1, 2, 7], [2, 2, 6], [3, 1, 6]])
    assert d.alpha == pytest.approx([3.0, 2.5, 9.5], abs=1e-12)
    draws = dirichlet_sample([1.5, 2.0, 0.7], 200, seed=9)
    assert all(abs(sum(r) - 1) < 1e-14 for r in draws)
    means = [sum(r[i] for r in draws) / 200 for i in range(3)]
    assert means[1] > means[0] > means[2]


def test_validation():
    with pytest.raises(ValueError):
        compositional_pielou([1.0, 0.0])
