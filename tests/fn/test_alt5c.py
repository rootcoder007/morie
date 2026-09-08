"""Tests for alt5c.alammar_t5_text_to_text_classify."""

from morie.fn.alt5c import alammar_t5_text_to_text_classify


def test_alt5c_basic():
    m = lambda inp, lab: 0.0 if lab == "yes" else -5.0
    out = alammar_t5_text_to_text_classify("q", ["yes", "no"], m)
    assert out["predicted_label"] == "yes"


def test_alt5c_edge():
    import pytest
    with pytest.raises(ValueError, match="label tokens"):
        alammar_t5_text_to_text_classify("q", [], lambda i, l: 0)
