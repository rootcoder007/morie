"""Tests for grsnt.geron_sentiment_binary."""

import math

import pytest

from morie.fn.grsig import geron_sigmoid

from morie.fn.grsnt import geron_sentiment_binary


E = [[1.0, -0.5], [2.0, 0.0], [-1.0, 3.0]]
W = [0.8, -0.3]


def test_grsnt_basic():
    """Mean pooling then one logistic unit: p = sigma(w . mean(E[ids]) + b)."""
    r = geron_sentiment_binary([0, 1, 2], E, W, b=0.1)
    pooled = [sum(E[i][j] for i in range(3)) / 3 for j in range(2)]
    logit = sum(a * c for a, c in zip(pooled, W)) + 0.1
    assert r["pooled"] == pytest.approx(pooled, rel=1e-15)
    assert r["logit"] == pytest.approx(logit, rel=1e-15)
    assert r["probability"] == pytest.approx(1 / (1 + math.exp(-logit)), rel=1e-15)
    assert r["label"] == (1 if r["probability"] >= 0.5 else 0)


def test_grsnt_edge():
    """Max pooling keeps each feature's strongest token; the sigmoid of a
    scalar is a scalar."""
    r = geron_sentiment_binary([0, 1, 2], E, W, pooling="max")
    assert r["pooled"] == [2.0, 3.0]
    assert isinstance(geron_sigmoid(1.5)["sigma"], float)
    with pytest.raises(ValueError, match="pooling"):
        geron_sentiment_binary([0], E, W, pooling="median")


