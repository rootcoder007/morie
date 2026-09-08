"""Tests for mxetA (max-stable simulation by spectral construction).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.mxetA import mxetA


def _spectral(n_t=5, m=3):
    # m spectral functions over n_t sites, all non-negative
    return [[1.0 + 0.5 * ((t + k) % 3) for k in range(m)]
            for t in range(n_t)]


def test_the_fields_have_the_right_shape():
    res = mxetA(_spectral(), n_sim=4, seed=1)
    assert len(res["fields"]) == 4
    assert all(len(f) == 5 for f in res["fields"])
    # n_points is the Poisson point count used per simulation
    assert len(res["n_points"]) == 4
    assert all(v > 0 for v in res["n_points"])


def test_the_values_are_positive():
    res = mxetA(_spectral(), n_sim=3, seed=1)
    assert all(v > 0 for f in res["fields"] for v in f)


def test_the_margins_are_unit_frechet():
    # P(Z <= z) = exp(-1/z), so the median is 1 / log 2 = 1.4427
    import math
    res = mxetA(_spectral(n_t=1, m=1), n_sim=400, seed=3)
    vals = sorted(f[0] for f in res["fields"])
    median = vals[len(vals) // 2]
    assert abs(median - 1.0 / math.log(2.0)) < 0.4


def test_seed_reproducibility():
    a = mxetA(_spectral(), n_sim=3, seed=11)["fields"]
    b = mxetA(_spectral(), n_sim=3, seed=11)["fields"]
    assert a == b


def test_scales_are_reported():
    res = mxetA(_spectral(), n_sim=2, seed=1)
    assert res["scales"]


def test_validation():
    for call in (lambda: mxetA([[1.0, 2.0], [1.0]]),
                 lambda: mxetA([[-1.0, 2.0], [1.0, 1.0]]),
                 lambda: mxetA([[0.0, 0.0], [0.0, 0.0]])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
