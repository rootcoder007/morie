"""Tests for evchitd.evt_chi_tail_dependence."""
from morie.fn import _array_core as np
from morie.fn.evchitd import evt_chi_tail_dependence


def test_bounds_and_ordering():
    rng = np.random.default_rng(3)
    x = [float(v) for v in rng.normal(0, 1, 1000)._flat()]
    e = [float(v) for v in rng.normal(0, 0.3, 1000)._flat()]
    y_dep = [a + b for a, b in zip(x, e)]
    y_ind = [float(v) for v in rng.normal(0, 1, 1000)._flat()]
    c_dep = evt_chi_tail_dependence(x, y_dep, u=0.9)["chi"]
    c_ind = evt_chi_tail_dependence(x, y_ind, u=0.9)["chi"]
    assert 0.0 <= c_ind < c_dep <= 1.0
