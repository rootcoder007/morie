"""Tests for morie.fn.prediction_improvement: values recomputed from first principles."""

import math

from morie.fn.prediction_improvement import prediction_improvement


def test_residual_fraction_of_a_regression():
    x = [1.0, 2.0, 3.0, 5.0, 8.0]
    y = [2.0, 2.5, 4.0, 4.5, 9.0]
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    r = sxy / math.sqrt(sxx * syy)
    sse = sum((b - my - sxy / sxx * (a - mx)) ** 2 for a, b in zip(x, y))
    assert abs(prediction_improvement(r)["mse_fraction_remaining"] - sse / syy) < 1e-14
