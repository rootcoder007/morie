"""Test mtvul."""

from morie.fn import _array_core as np

from morie.fn.mtvul import mtvul


def test_mtvul_basic():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtvul(trajectory=traj, n=25)
    assert r.value is not None


def test_mtvul_description():
    rng = np.random.default_rng(42)
    traj = np.cumsum(rng.standard_normal((25, 2)), axis=0)
    r = mtvul(trajectory=traj, n=25)
    assert r.name
