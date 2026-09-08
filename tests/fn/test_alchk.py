"""Tests for alchk.alammar_recursive_chunking."""

from morie.fn.alchk import alammar_recursive_chunking


def test_alchk_basic():
    out = alammar_recursive_chunking("aa bb. cc dd. ee",
        separators=[". ", " "], target_size=6)
    assert out["chunks"] == ["aa bb", "cc dd", "ee"]


def test_alchk_edge():
    import pytest
    with pytest.raises(ValueError, match="overlap"):
        alammar_recursive_chunking("x", target_size=3, overlap=3)
