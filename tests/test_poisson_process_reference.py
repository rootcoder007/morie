"""Log-linear Poisson process fit (spatstat.model::ppm on the same quadrature)."""

import math

from morie.fn._rng import random_uniform
from morie.fn.poifit import poisson_process_fit


def pattern():
    U = [float(u) for u in random_uniform(120, seed=21, stream=0)]
    return [[2 * U[2 * i], U[2 * i + 1]] for i in range(60)]


def test_matches_ppm_with_the_same_quadrature():
    P = pattern()
    assert abs(math.exp(poisson_process_fit(P, (0, 2, 0, 1), degree=0).value[0]) - 30.0) < 1e-12
    # ppm(quadscheme(X, D, method = "grid", ntile = c(12, 12)) ~ x + y): coef, sqrt(diag(vcov))
    r = poisson_process_fit(P, (0, 2, 0, 1), degree=1)
    assert all(abs(a - b) < 1e-9 for a, b in zip(r.value, [3.51859174463, -0.114096826248, -0.011132311411]))
    assert all(abs(a - b) < 1e-9 for a, b in zip(r.extra["se"], [0.3362050495, 0.223953776, 0.4474899548]))
    r = poisson_process_fit(P, (0, 2, 0, 1), degree=2)
    ref = [4.319475976284, -2.040753170965, -0.7274820847, 0.830874893131, 0.573393831093, 0.159220316807]
    assert all(abs(a - b) < 1e-8 for a, b in zip(r.value, ref))
    # the score equation: fitted mass equals the count
    assert abs(r.extra["expected_count"] - 60) < 1e-9


def test_thinning_simulation():
    r = poisson_process_fit(pattern(), (0, 2, 0, 1), degree=1, simulate=400, seed=3)
    counts = [len(p) for p in r.extra["simulated"]]
    assert abs(sum(counts) / 400 - 60) < 3 * math.sqrt(60 / 400)
    assert all(0 <= x <= 2 and 0 <= y <= 1 for p in r.extra["simulated"] for x, y in p)
    assert counts[:5] == [
        len(p) for p in poisson_process_fit(pattern(), (0, 2, 0, 1), simulate=5, seed=3).extra["simulated"]
    ]
