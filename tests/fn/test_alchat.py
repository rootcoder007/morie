"""Tests for alchat.alammar_chat_template."""

from morie.fn.alchat import alammar_chat_template


def test_alchat_basic():
    out = alammar_chat_template([("user", "hi")], {"user": ("<u>", "</u>")})
    assert out["prompt"] == "<u>hi</u>"


def test_alchat_edge():
    import pytest
    with pytest.raises(ValueError, match="no template tokens"):
        alammar_chat_template([("robot", "x")], {"user": ("", "")})
