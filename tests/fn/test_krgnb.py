"""Test krgnb."""

from morie.fn import _array_core as np

from morie.fn.krgnb import krgnb


def test_krgnb_basic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = krgnb(x=x, y=y, values=v)
    assert r.value is not None


def test_krgnb_description():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 100, 20)
    y = rng.uniform(0, 100, 20)
    v = rng.standard_normal(20)
    r = krgnb(x=x, y=y, values=v)
    assert r.name
