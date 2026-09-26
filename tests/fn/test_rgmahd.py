"""Tests for bsaclass.rangayyan_mahalanobis (Rangayyan 2024 sec. 10.4.3)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_mahalanobis


def test_rgmahd_basic():
    """D^2 = (x - mu)' C^-1 (x - mu): with C = [[2, .5], [.5, 1]],
    C^-1 = [[1, -.5], [-.5, 2]] / 1.75, so x = (1, 2) gives 7 / 1.75 = 4;
    the identity covariance reduces it to the squared Euclidean distance."""
    r = rangayyan_mahalanobis([1.0, 2.0], [0.0, 0.0], [[2.0, 0.5], [0.5, 1.0]])
    assert r["d2"] == pytest.approx(4.0, rel=1e-12)
    assert r["distance"] == pytest.approx(2.0, rel=1e-12)
    assert r["euclidean"] == pytest.approx(math.sqrt(5.0), rel=1e-12)
    e = rangayyan_mahalanobis([1.0, 2.0], [0.5, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    assert e["d2"] == pytest.approx(0.25 + 4.0, rel=1e-12)


def test_rgmahd_edge():
    """A singular covariance cannot define the distance."""
    with pytest.raises((ValueError, ZeroDivisionError)):
        rangayyan_mahalanobis([1.0, 2.0], [0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]])
