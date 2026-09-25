"""Tests for bkcbow.burkov_cbow."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.bkcbow import burkov_cbow

E3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_bkcbow_basic():
    """The docstring example, with the softmax loss it implies."""
    out = burkov_cbow([[0, 1]], [2], E3, E3)
    assert out["hidden"].shape == (1, 3)
    assert out["n"] == 1
    assert out["context_size"] == 2
    assert out["vocab_size"] == 3
    assert out["dim"] == 3
    # h averages rows 0 and 1 of the identity
    assert list(out["hidden"][0]) == pytest.approx([0.5, 0.5, 0.0],
                                                  abs=1e-15)
    # logits = h @ I, so softmax over (0.5, 0.5, 0)
    z = [0.5, 0.5, 0.0]
    denom = sum(math.exp(v) for v in z)
    prob = [math.exp(v) / denom for v in z]
    assert list(out["logits"][0]) == pytest.approx(z, abs=1e-15)
    assert list(out["probabilities"][0]) == pytest.approx(prob, abs=1e-12)
    assert out["loss"] == pytest.approx(-math.log(prob[2]), abs=1e-12)
    assert out["estimate"] == out["loss"]
    assert out["perplexity"] == pytest.approx(1.0 / prob[2], abs=1e-9)
    assert out["rare_word_dilution"] == 0.5
    # the centre word is not the argmax here, so accuracy is zero
    assert int(out["predicted"][0]) == 0
    assert out["accuracy"] == 0.0


def test_bkcbow_edge():
    """Averaging discards order; the loss falls when the centre wins."""
    a = burkov_cbow([[0, 1]], [2], E3, E3)
    b = burkov_cbow([[1, 0]], [2], E3, E3)
    assert list(b["hidden"][0]) == pytest.approx(list(a["hidden"][0]),
                                                 abs=1e-15)
    assert b["loss"] == pytest.approx(a["loss"], abs=1e-15)
    # a bias that favours the centre word makes it the prediction
    biased = burkov_cbow([[0, 1]], [2], E3, E3, output_bias=[0.0, 0.0, 5.0])
    assert int(biased["predicted"][0]) == 2
    assert biased["accuracy"] == 1.0
    assert biased["loss"] < a["loss"]
    # a wider context dilutes each word further
    wide = burkov_cbow([[0, 1, 2, 0]], [1], E3, E3)
    assert wide["rare_word_dilution"] == 0.25
    assert wide["context_size"] == 4
    assert list(wide["hidden"][0]) == pytest.approx([0.5, 0.25, 0.25],
                                                   abs=1e-15)
    # two examples: accuracy is the mean over rows
    two = burkov_cbow([[0, 0], [1, 1]], [0, 1], E3,
                      [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                       [0.0, 0.0, 10.0]])
    assert list(two["predicted"]) == [0, 1]
    assert two["accuracy"] == 1.0
    assert two["n"] == 2
    with pytest.raises(ValueError):
        burkov_cbow([[0, 1]], [2, 0], E3, E3)
    with pytest.raises(ValueError):
        burkov_cbow([[0, 1]], [2], E3, [[1.0, 0.0], [0.0, 1.0],
                                        [1.0, 1.0]])
    with pytest.raises(ValueError):
        burkov_cbow([[0, 9]], [2], E3, E3)
    with pytest.raises(ValueError):
        burkov_cbow([[0, 1]], [2], E3, E3, output_bias=[0.0, 0.0])
