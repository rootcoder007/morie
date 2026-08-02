"""Front tests: every ACD module called once, headline checked."""

import importlib
import math

import numpy as np
import pytest

PREFIX = "analysis_of_categorical_data_with_r_chapman_hall_crc_christo"


def call(mid, *args):
    mod = importlib.import_module("morie.fn." + PREFIX + mid)
    ch, eq = mid.split("e")
    fn = getattr(mod, "%s_chapter_%s_equation_%s" % (PREFIX, ch, eq))
    res = fn(*args)
    assert isinstance(res, dict) and "value" in res
    assert "eq. (%s.%s)" % (ch, eq) in res["method"]
    assert isinstance(mod.cheatsheet(), str)
    return res


X8 = np.column_stack([np.ones(8), np.arange(8.0)])
Y8 = np.array([0.0, 0, 0, 1, 0, 1, 1, 1])
BASIS = [lambda x: 1.0, lambda x: x, lambda x: x ** 2, lambda x: x ** 3,
         lambda x: (x - 2.0) ** 3 if x > 2.0 else 0.0]


def wilson_fn(w, n):
    z = 1.96
    p_hat = w / n
    p_t = (w + z * z / 2) / (n + z * z)
    h = z * math.sqrt(n) / (n + z * z) \
        * math.sqrt(p_hat * (1 - p_hat) + z * z / (4 * n))
    return p_t - h, p_t + h


CASES = {
    "1e1": ((3, 10, 0.3), math.comb(10, 3) * 0.3 ** 3 * 0.7 ** 7),
    "1e3": ((0.4, 25), 0.24 / 25),
    "1e4": ((4, 10, 1.96), None),
    "1e5": ((1 / 3, 2.0, 3.0), None),
    "1e6": ((10, 0.3, wilson_fn), None),
    "1e7": ((12, 30, 20, 35), None),
    "1e8": ((10, 20, 15, 30), 0.0),
    "1e10": ((20, 50, 10, 50, 1.96), (20 / 30) / (10 / 40)),
    "2e1": (([0.3, 0.7], [0.0, 1.0]), math.log(0.7) + math.log(0.7)),
    "2e2": ((-1.0, [0.5], [2.0]), 0.5),
    "2e3": ((0.5,), 0.0),
    "2e4": (([0.0, 0.0], X8, Y8), 8 * math.log(0.5)),
    "2e5": (([0.0, 0.0], X8, Y8), 8 * math.log(0.5)),
    "2e6": ((-10.0, -8.0), 4.0),
    "2e7": ((-10.0, -8.0), 4.0),
    "2e8": (([0.5] * 8, Y8), -2 * 8 * math.log(0.5)),
    "2e9": (([0.5] * 8, Y8), -2 * 8 * math.log(0.5)),
    "2e11": ((0.5, 0.04, 2.0, 1.96), math.e),
    "2e15": ((0.0, 0.25, 1.96), 0.5),
    "2e16": (([1.0, 3.0], np.array([[0.5, -0.1], [-0.1, 0.05]])),
             0.5 + 0.45 - 0.6),
    "2e22": (([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 1, 0, 1, 0),
             1.3),
    "3e1": (([3, 7], [0.3, 0.7]), math.comb(10, 3) * 0.3 ** 3 * 0.7 ** 7),
    "3e2": ((np.array([[2, 1], [1, 3]]),
             np.array([[0.2, 0.1], [0.3, 0.4]])), None),
    "3e3": ((np.array([[2, 1], [1, 3]]),
             np.array([[2 / 3, 1 / 3], [0.25, 0.75]])), None),
    "3e4": ((0.5, [0.3], [2.0]), 1.1),
    "3e8": ((0.3, 0.0025, 1.96), 0.3 - 1.96 * 0.05),
    "3e10": ((0.5, [0.3, -0.2], [1.0, 0.0]), 0.8),
    "3e11": ((0.5, [0.3], [2.0]), 1.1),
    "3e12": (([0.15, 0.45, 0.80], 2), 0.30),
    "3e13": ((0.5, [-0.3], [2.0]), 1.1),
    "3e16": ((0.5, [0.3], [2.0]), 1.1),
    "3e50": ((0.43, 0.01, 4.0, 1.96), math.exp(1.72)),
    "4e1": ((3.0, 20, 1.96), None),
    "4e2": ((0.1, [1.0], [1.0]), math.exp(1.1)),
    "4e3": (([0.0, math.log(2)],
             np.column_stack([np.ones(4), np.arange(4.0)]),
             [1.0, 2, 4, 8]), None),
    "4e4": ((1.0, 0.9, 0.4), math.exp(2.3)),
    "4e5": ((1.0, 0.9, 0.4), math.exp(2.3)),
    "4e6": ((1.0, 0.9, 0.4, 0.7), math.exp(3.0)),
    "4e7": ((0.0, 0.7, 0.0, 0.0), math.exp(0.7)),
    "4e12": ((0.4, 0.1, 0.2, 3.0, 1.0), math.exp(0.7)),
    "4e15": ((0.1, [1.0], [1.0], 100.0), 100 * math.exp(1.1)),
    "5e2": (([100.0, 102.0],), 0.7311),
    "5e3": (([0.6, 0.4], [1.0, 2.0]), 1.4),
    "5e4": (([0.6, 0.4], [1.0, 2.0], [0.1, 0.2]), None),
    "6e1": ((0.1, 0.95, 0.98), (0.1 + 0.98 - 1) / 0.93),
    "6e3": ((0.3, 0.9, 0.95), (0.3 + 0.95 - 1) / 0.85),
    "6e4": (([0.0, 0.0], X8, Y8), 8 * math.log(0.5)),
    "6e5": (([0.0, 1.0, 2.0], [1, 4, 2], 0.0, 1.0), 4 / 7),
    "6e6": (([0.0, 1.0, 2.0], [1, 4, 2], 0.0, 1.0), 4 / 7),
    "6e7": (([2.0, 3.0, 5.0], ["a", "b", "a"], "a"), 7.0),
    "6e8": (([1.0, 1.2, 0.8, 1.1], 1.0), None),
    "6e9": ((4.0, 9.0, 1.5, 0.3, 100.0), (4 + 0.81 - 0.9) / 1e4),
    "6e10": (([0.3, 0.32, 0.28], 0.3), None),
    "6e11": ((0.3, 0.01, 2.0), None),
    "6e14": ((1.0, 0.2, 0.3), math.exp(1.5)),
    "6e15": ((1.0, 0.2, 0.3), math.exp(1.5)),
    "6e16": ((1.0, 0.2, 0.3, 0.1), math.exp(1.6)),
    "6e17": ((0.5, -0.2), 0.3),
    "6e18": ((0.5, 2.0, 1.5, -0.2), 3.3),
    "6e20": ((0.5, [0.3, -0.2], [1.0, 2.0], -0.1), 0.5 - 0.1 - 0.1),
    "6e22": ((0.99, 0.01, 0.05), 0.0099 / 0.0594),
    "6e23": ((0.4, 7, 20, 1, 1), None),
    "6e24": ((7, 20, 1, 1), 8 / 22),
    "6e25": (([-3.0, -1.0, -2.0], [0.0, 0.0, 0.0]), None),
    "6e26": ((5, 1.0, 1.0, 0.1), 1 + 5 * (1 - 0.9 ** 5)),
    "6e32": ((-1.0, [0.5], [2.0]), 0.5),
    "6e34": ((1.5, 2.0, [1.0, 0.5, -0.2, 0.1], [9.9, 9.9, 9.9, 9.9]),
             1.0 + 0.75 - 0.45 + 0.1 * 1.5 ** 3),
    "6e36": ((3.0, [1.0, 0.5, -0.2, 0.1, 0.3], [2.0]),
             1.0 + 1.5 - 1.8 + 2.7 + 0.3),
    "6e37": (([1.0, 0.5, -0.2, 0.1, 0.3], BASIS, 3.0, 1.0), None),
}


@pytest.mark.parametrize("mid", sorted(CASES))
def test_front(mid):
    args, want = CASES[mid]
    res = call(mid, *args)
    if want is not None:
        assert res["value"] == pytest.approx(want, rel=2e-3, abs=2e-3)
    else:
        assert np.isfinite(res["value"])


def test_all_69_covered():
    assert len(CASES) == 69
    from collections import Counter
    census = Counter(int(m.split("e")[0]) for m in CASES)
    assert census == {1: 8, 2: 13, 3: 11, 4: 9, 5: 3, 6: 25}


def test_cross_module_consistency():
    # eq (2.2) and eq (2.3) are inverses
    p = call("2e2", -1.0, [0.5], [2.0])["value"]
    assert call("2e3", p)["value"] == pytest.approx(0.0, abs=1e-12)
    # eq (6.1) inverts the apparent-probability construction
    se, sp, pt = 0.95, 0.98, 0.12
    pi = se * pt + (1 - sp) * (1 - pt)
    assert call("6e1", pi, se, sp)["value"] == pytest.approx(pt, rel=1e-9)
    # eq (3.13) polr equals eq (3.11) with negated etas
    a = call("3e11", 0.5, [0.3], [2.0])["value"]
    b = call("3e13", 0.5, [-0.3], [2.0])["value"]
    assert a == pytest.approx(b, rel=1e-12)
    # eq (6.36) equals eq (6.37)'s basis evaluation path
    f3 = call("6e36", 3.0, [1.0, 0.5, -0.2, 0.1, 0.3], [2.0])["value"]
    f1 = call("6e36", 1.0, [1.0, 0.5, -0.2, 0.1, 0.3], [2.0])["value"]
    orr = call("6e37", [1.0, 0.5, -0.2, 0.1, 0.3], BASIS, 3.0, 1.0)["value"]
    assert orr == pytest.approx(math.exp(f3 - f1), rel=1e-9)
