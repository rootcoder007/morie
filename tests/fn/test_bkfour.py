"""Tests for bkfour -- the treatment-first front-end over bkmed."""

from morie.fn import _array_core as np

from morie.fn.bkfour import baron_kenny_four_step
from morie.fn.bkmed import baron_kenny


def _data(seed=0, n=500):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    m = 0.8 * x + rng.normal(0, 1, n)
    y = 0.3 * x + 0.6 * m + rng.normal(0, 1, n)
    return x, m, y


def test_front_end_matches_the_canonical_implementation():
    x, m, y = _data(seed=1)
    a = baron_kenny(y, x, m)
    b = baron_kenny_four_step(x, m, y)
    for k in ("c", "a", "b", "c_prime", "indirect", "mediation"):
        assert a[k] == b[k]


def test_all_four_steps_are_reported():
    x, m, y = _data(seed=2)
    steps = baron_kenny_four_step(x, m, y)["steps"]
    assert len(steps) == 4
    assert all(isinstance(v, bool) for v in steps.values())


def test_it_does_not_carry_a_second_implementation():
    import inspect

    from morie.fn import bkfour

    assert "bkmed" in inspect.getsource(bkfour)
    assert len(inspect.getsource(bkfour.baron_kenny_four_step).splitlines()) < 20
