"""Tests for morie.fn.a2cv: equals hma2c.geron_a2c with n_steps as max_steps."""

from morie.fn.a2cv import a2c
from morie.fn.hma2c import geron_a2c


def _env():
    return {"reset": lambda: [1.0], "step": lambda a: ([1.0], 1.0 if a == 0 else 0.0, True)}


def test_same_training_path_as_geron_a2c():
    r = a2c(_env(), [[0.0], [0.0]], [0.0], n_steps=5, epochs=60, lr=0.5, seed=3)
    g = geron_a2c(_env(), [[0.0], [0.0]], [0.0], epochs=60, lr=0.5, max_steps=5, seed=3)
    assert list(r["returns"]) == list(g["returns"])
    assert r["policy"]([1.0])[0] > 0.8
