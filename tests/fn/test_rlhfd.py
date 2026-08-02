"""rlhfd: linear reward head (Ouyang et al. 2022, InstructGPT).  r = w'h + b"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rlhfd import rlhf_reward as rw


def test_rlhfd_matches_the_closed_form():
    rng = np.random.default_rng(3001)
    h = rng.standard_normal((5, 7))
    w = rng.standard_normal(7)
    b = 0.75
    assert np.asarray(rw(h, w=w, b=b)["tensor"]) == pytest.approx(h @ w + b)


def test_rlhfd_is_affine_in_the_hidden_state():
    """r(ah1 + (1-a)h2) = a r(h1) + (1-a) r(h2) for a linear head."""
    rng = np.random.default_rng(3011)
    w = rng.standard_normal(6)
    h1, h2 = rng.standard_normal(6), rng.standard_normal(6)
    a = 0.3
    lhs = float(np.asarray(rw(a * h1 + (1 - a) * h2, w=w, b=1.0)["tensor"]).ravel()[0])
    r1 = float(np.asarray(rw(h1, w=w, b=1.0)["tensor"]).ravel()[0])
    r2 = float(np.asarray(rw(h2, w=w, b=1.0)["tensor"]).ravel()[0])
    assert lhs == pytest.approx(a * r1 + (1 - a) * r2)


def test_rlhfd_bias_shifts_every_reward_equally():
    """A constant bias cannot change any PREFERENCE, only the level -- which
    is why reward models are identified only up to an additive constant."""
    rng = np.random.default_rng(3017)
    h = rng.standard_normal((4, 5))
    w = rng.standard_normal(5)
    a = np.asarray(rw(h, w=w, b=0.0)["tensor"])
    c = np.asarray(rw(h, w=w, b=10.0)["tensor"])
    assert c - a == pytest.approx(np.full(4, 10.0))
    assert np.argsort(a).tolist() == np.argsort(c).tolist()


def test_rlhfd_zero_weight_gives_the_bias_alone():
    rng = np.random.default_rng(3019)
    h = rng.standard_normal((3, 4))
    out = np.asarray(rw(h, w=np.zeros(4), b=2.5)["tensor"])
    assert out == pytest.approx(np.full(3, 2.5))


def test_rlhfd_echoes_its_parameters():
    r = rw(np.zeros((2, 3)), w=np.array([1.0, 2.0, 3.0]), b=0.5)
    assert np.asarray(r["w"]) == pytest.approx([1.0, 2.0, 3.0])
    assert r["b"] == pytest.approx(0.5)


def test_rlhfd_rejects_a_dimension_mismatch():
    with pytest.raises((ValueError, IndexError)):
        rw(np.zeros((2, 4)), w=np.zeros(5))
