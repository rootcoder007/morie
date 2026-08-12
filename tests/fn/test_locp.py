"""Tests for locp (local polynomial smoother, Fan & Gijbels 1996).

Replaces the generated stub, which imported ``local_polynomial``.
"""

import math

from morie.fn.locp import locp


def _grid(n=60):
    return [i / float(n - 1) * 4.0 - 2.0 for i in range(n)]


def test_a_straight_line_is_reproduced_by_a_local_linear_fit():
    x = _grid()
    y = [3.0 + 2.0 * v for v in x]
    res = locp(x, y, degree=1, bandwidth=1.0)
    assert max(abs(res["fitted"][i] - y[i])
               for i in range(len(x))) < 1e-9


def test_local_linear_recovers_the_slope():
    x = _grid()
    y = [3.0 + 2.0 * v for v in x]
    res = locp(x, y, x0=[0.0], degree=1, bandwidth=1.0)
    assert abs(res["slope"][0] - 2.0) < 1e-9


def test_a_quadratic_needs_degree_two():
    x = _grid()
    y = [v * v for v in x]
    deg1 = locp(x, y, x0=[0.0], degree=1, bandwidth=1.0)["fitted"][0]
    deg2 = locp(x, y, x0=[0.0], degree=2, bandwidth=1.0)["fitted"][0]
    assert abs(deg2 - 0.0) < abs(deg1 - 0.0)
    assert abs(deg2) < 1e-9


def test_degree_zero_is_a_kernel_average():
    x = _grid()
    y = [1.0] * len(x)
    res = locp(x, y, degree=0, bandwidth=0.5)
    assert max(abs(v - 1.0) for v in res["fitted"]) < 1e-9


def test_all_three_kernels_run_and_a_wider_band_smooths_more():
    x = _grid()
    y = [math.sin(v) + (0.2 if i % 7 == 0 else 0.0)
         for i, v in enumerate(x)]
    for kern in ("tricube", "epanechnikov", "gaussian"):
        assert len(locp(x, y, kernel=kern, bandwidth=0.8)["fitted"]) == \
            len(x)
    rough = locp(x, y, bandwidth=0.2)["fitted"]
    smooth = locp(x, y, bandwidth=1.5)["fitted"]
    var_rough = sum((rough[i + 1] - rough[i]) ** 2
                    for i in range(len(x) - 1))
    var_smooth = sum((smooth[i + 1] - smooth[i]) ** 2
                     for i in range(len(x) - 1))
    assert var_smooth < var_rough


def test_validation():
    x = _grid()
    y = [1.0] * len(x)
    for call in (lambda: locp([1.0], [1.0]),
                 lambda: locp(x, y[:-1]),
                 lambda: locp(x, y, degree=-1),
                 lambda: locp(x, y, bandwidth=0.0),
                 lambda: locp(x, y, kernel="boxcar")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
