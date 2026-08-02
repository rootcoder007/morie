"""Test dtcpj."""

from morie.fn import _array_core as np

from morie.fn.dtcpj import dtcpj


def test_dtcpj_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpj(x=x, n=50)
    assert r.value is not None


def test_dtcpj_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtcpj(x=x, n=50)
    assert r.name
