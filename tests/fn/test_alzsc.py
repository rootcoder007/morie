"""Tests for alzsc.alammar_zero_shot_classification."""

from morie.fn.alzsc import alammar_zero_shot_classification


def test_alzsc_basic():
    nli = lambda p, h: 5.0 if "sport" in h else 0.0
    out = alammar_zero_shot_classification("text", ["sport", "news"], nli)
    assert out["predicted_label"] == "sport"


def test_alzsc_edge():
    import pytest
    with pytest.raises(ValueError, match="duplicates"):
        alammar_zero_shot_classification("t", ["a", "a"], lambda p, h: 0)
