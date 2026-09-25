"""Tests for hybRC.hybrid_rec."""

import pytest

from morie.fn.hybRC import cascade, weighted


def test_hybRC_basic():
    """Weighted hybrid: a weighted sum over the components that score an
    item; items only some components score are flagged."""
    r = weighted([{"a": 1.0, "b": 0.5}, {"a": 0.2, "c": 0.9}], [0.7, 0.3])
    assert r["scores"] == pytest.approx({"a": 0.76, "b": 0.35, "c": 0.27})
    assert r["ranking"] == ["a", "b", "c"]
    assert r["partially_scored"] == ["b", "c"]


def test_hybRC_edge():
    """Cascade: the secondary only breaks the primary's ties."""
    r = cascade({"x": 1.0, "y": 1.0, "z": 2.0}, {"x": 0.1, "y": 0.9, "z": 0.0})
    assert r["ranking"] == ["z", "y", "x"]
    assert r["primary_respected"] and r["tie_groups_broken"] == 1
    with pytest.raises(ValueError, match="weight"):
        weighted([{"a": 1.0}, {"a": 2.0}], [1.0])


