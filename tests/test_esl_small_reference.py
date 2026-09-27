"""Small ESL formulas: median nearest-neighbour radius (2.24), test R^2 (9.23-9.24), hyperplane distance, correlation distance."""

from morie.fn import correlation_dist, hyperplane_side
from morie.fn.eslmnr import esl_median_nn_radius
from morie.fn.esltr2 import esl_test_r2


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_small_formulas():
    r = esl_median_nn_radius(500, 10)["median_radius"]
    assert close(r, (1 - 0.5 ** (1 / 500)) ** 0.1) and round(r, 2) == 0.52  # ESL p. 23: about 0.52
    t = esl_test_r2([1, 2, 3], [1.1, 1.8, 3.3], 2.1)
    assert close(t["mse"], (0.01 + 0.04 + 0.09) / 3) and close(t["mse0"], (1.21 + 0.01 + 0.81) / 3)
    assert close(t["r2"], (t["mse0"] - t["mse"]) / t["mse0"])
    h = hyperplane_side([[1.0, 2.0], [0.0, -1.0]], [3.0, 4.0], -2.0)
    assert h["value"].tolist() == [9.0, -6.0] and h["distance"].tolist() == [1.8, 1.2]
    d = correlation_dist([1, 2, 4, 3], [2, 1, 5, 5])
    assert close(d.estimate, 0.185908421589306)  # 1 - cor() in R
