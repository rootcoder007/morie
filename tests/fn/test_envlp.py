"""Tests for morie.fn.envlp."""

from morie.fn import _array_core as np

from morie.fn.envlp import envelope


def test_envlp_smoke():
    rng = np.random.default_rng(42)
    result = envelope(signal=np.sin(np.linspace(0, 4 * np.pi, 100)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.envlp import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
