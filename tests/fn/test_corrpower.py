import math

from morie.fn.corrpower import cohen_r_magnitude, pwr_r_test
from morie.fn.qt import qt


def _pn(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _power(n, r, a, two_sided):
    t = qt(1 - a / 2 if two_sided else 1 - a, n - 2)
    rc = t / math.sqrt(t * t + n - 2)
    zr = math.atanh(r) + r / (2 * (n - 1))
    p = _pn((zr - math.atanh(rc)) * math.sqrt(n - 3))
    return p + (_pn((-zr - math.atanh(rc)) * math.sqrt(n - 3)) if two_sided else 0.0)


def test_power_formula():
    assert abs(pwr_r_test(n=100, r=0.3).power - _power(100, 0.3, 0.05, True)) < 1e-12
    assert (
        abs(pwr_r_test(n=55, r=0.2, sig_level=0.01, alternative="greater").power - _power(55, 0.2, 0.01, False)) < 1e-12
    )
    assert abs(pwr_r_test(n=37, r=-0.25, alternative="less").power - _power(37, 0.25, 0.05, False)) < 1e-12


def test_solves_round_trip():
    n = pwr_r_test(r=0.3, power=0.8).n
    assert abs(_power(n, 0.3, 0.05, True) - 0.8) < 1e-9
    r = pwr_r_test(n=100, power=0.9).r
    assert abs(_power(100, r, 0.05, True) - 0.9) < 1e-9
    a = pwr_r_test(n=60, r=0.35, power=0.75, sig_level=None).sig_level
    assert abs(_power(60, 0.35, a, True) - 0.75) < 1e-9
    r = pwr_r_test(n=80, power=0.7, alternative="less").r
    assert r < 0 and abs(_power(80, -r, 0.05, False) - 0.7) < 1e-9


def test_cohen_labels():
    assert [cohen_r_magnitude(v) for v in (0.0, 0.1, -0.3, 0.5, -1.0)] == [
        "negligible",
        "small",
        "medium",
        "large",
        "large",
    ]
