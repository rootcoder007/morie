"""Tests for layrnm (alias of grln.geron_layer_normalization)."""

from morie.fn.grln import geron_layer_normalization
from morie.fn.layrnm import layer_norm, layrnm


def test_layrnm_anchor_ba2016():
    # Hand anchor, Ba et al. (2016) Sec 3 definition with eps = 0:
    # x = [1, 3]: mu = 2, sigma2 = 1, normalized = [-1, 1].
    r = layrnm([1.0, 3.0], eps=0.0)
    assert r["normalized"] == [-1.0, 1.0]
    assert (r["mean"], r["variance"]) == (2.0, 1.0)
    # affine after normalisation: gamma 2, beta 5 -> [3, 7]
    assert layrnm([1.0, 3.0], gamma=2.0, beta=5.0, eps=0.0)["output"] == [3.0, 7.0]


def test_layrnm_alias_exact_zero():
    X = [[0.5, -1.5, 2.0], [3.0, 0.0, -1.0]]
    a = layrnm(X, gamma=1.3, beta=-0.2, eps=1e-5)
    b = geron_layer_normalization(X, gamma=1.3, beta=-0.2, eps=1e-5)
    assert a["output"] == b["output"]
    assert a["mean"] == b["mean"] and a["variance"] == b["variance"]
    assert layer_norm is layrnm
