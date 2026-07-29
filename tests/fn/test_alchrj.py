"""Tests for alchrj.alammar_chosen_rejected_template."""

from morie.fn.alchrj import alammar_chosen_rejected_template


def test_alchrj_basic():
    out = alammar_chosen_rejected_template(["p"], ["good"], ["bad"])
    assert out["records"][0]["chosen"] == "good"


def test_alchrj_edge():
    import pytest
    with pytest.raises(ValueError, match="no preference"):
        alammar_chosen_rejected_template(["p"], ["x"], ["x"])
