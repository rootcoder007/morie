import math

from morie.fn.mgwrinfer import bandwidth_confidence_interval, mgwr_local_t
from morie.fn.qt import qt


def test_local_t_and_critical_value():
    r = mgwr_local_t([0.5, -0.2, 1.1, 0.05], [0.2, 0.25, 0.3, 0.1], 3.5, alpha=0.05, df=40)
    assert r.t == [0.5 / 0.2, -0.2 / 0.25, 1.1 / 0.3, 0.05 / 0.1]
    assert abs(r.critical - qt(1 - 0.05 / 3.5 / 2, 40)) < 1e-12
    assert r.significant == [abs(v) > r.critical for v in r.t]


def test_akaike_weight_interval():
    bw, a = [40, 50, 60, 70, 80], [310.0, 302.0, 300.0, 301.0, 306.0]
    r = bandwidth_confidence_interval(bw, a)
    e = [math.exp(-(v - 300.0) / 2) for v in a]
    w = [v / sum(e) for v in e]
    assert all(abs(x - y) < 1e-15 for x, y in zip(r.weights, w))
    assert abs(r.coverage - (w[2] + w[3] + w[1])) < 1e-15 and (r.lower, r.upper) == (50.0, 70.0)
