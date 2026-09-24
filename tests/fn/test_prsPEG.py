"""Tests for prsPEG.peg_parser."""

from morie.fn.prsPEG import peg_parser, lit, seq


def test_prsPEG_basic():
    """Test basic literal matching."""
    grammar = lit("hello")
    result = peg_parser(grammar, "hello")
    assert isinstance(result, dict)
    assert result.get("matched") is True


def test_prsPEG_edge():
    """Test sequence matching."""
    grammar = seq(lit("a"), lit("b"))
    result = peg_parser(grammar, "ab")
    assert isinstance(result, dict)
    assert result.get("matched") is True
