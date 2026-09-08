"""tmpsc: temperature scaling of logits (Hinton et al. 2015).  softmax(z / T)"""

from morie.fn import _array_core as np
import pytest

from morie.fn.tmpsc import temperature_scaling as ts


def _p(x, T=1.0):
    return np.asarray(ts(x, T=T)["tensor"])


def test_tmpsc_output_is_a_probability_distribution():
    rng = np.random.default_rng(1701)
    p = _p(rng.standard_normal((5, 7)))
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.all(p >= 0)


def test_tmpsc_matches_the_closed_form():
    z = np.array([[1.0, 2.0, 3.0]])
    T = 2.0
    e = np.exp(z / T - (z / T).max())
    assert _p(z, T) == pytest.approx(e / e.sum())


def test_tmpsc_temperature_one_is_plain_softmax():
    z = np.array([[0.5, -1.0, 2.0, 0.0]])
    e = np.exp(z - z.max())
    assert _p(z, 1.0) == pytest.approx(e / e.sum())


def test_tmpsc_high_temperature_flattens_toward_uniform():
    z = np.array([[0.0, 1.0, 5.0]])
    hot = _p(z, 1000.0)
    assert hot.ravel() == pytest.approx(np.full(3, 1 / 3), abs=1e-2)


def test_tmpsc_low_temperature_sharpens_toward_the_argmax():
    z = np.array([[0.0, 1.0, 5.0]])
    cold = _p(z, 0.01)
    assert cold[0, 2] > 0.999


def test_tmpsc_entropy_increases_monotonically_with_temperature():
    """The reason temperature exists: it is a knob on output entropy."""
    z = np.array([[0.0, 1.0, 2.0, 3.0]])
    ents = [ts(z, T=T)["entropy"] for T in (0.5, 1.0, 2.0, 5.0, 20.0)]
    ents = [float(np.asarray(e).ravel()[0]) for e in ents]
    assert ents == sorted(ents)


def test_tmpsc_is_invariant_to_adding_a_constant_to_all_logits():
    """Softmax depends only on logit DIFFERENCES."""
    z = np.array([[1.0, 2.0, 3.0]])
    assert _p(z + 17.0, 1.5) == pytest.approx(_p(z, 1.5))


def test_tmpsc_rejects_a_non_positive_temperature():
    with pytest.raises((ValueError, ZeroDivisionError)):
        ts(np.array([[1.0, 2.0]]), T=0.0)
