"""Tests for bleuS.bleu."""

from morie.fn import _array_core as np

from morie.fn.bleuS import bleu


def test_bleuS_basic():
    """Test basic functionality."""
    candidate = "the the the the the the the the the the the the"
    references = [
        "the the the the the the the the the the the the",
        "the the the the the the the the the the the the the the",
    ]
    result = bleu(candidate, references, max_n=4)
    assert isinstance(result.summary_lines, list)
    # bleu score must be in [0, 1]
    sc = result.payload["bleu"]
    assert 0.0 <= sc <= 1.0
    # every n-gram precision must be 1.0 because candidate is a strict sub-sequence
    # of every reference, so all clipped counts equal the total candidate counts.
    assert result.payload["p_n"] == [1.0, 1.0, 1.0, 1.0]
    # independent recomputation of the BLEU score from the formula
    import math
    c = result.payload["c"]
    r = result.payload["r"]
    bp = 1.0 if c > r else math.exp(1.0 - (r + 0.0) / c)
    w = 1.0 / result.payload["max_n"]
    logsum = sum(w * math.log(p) for p in result.payload["p_n"])
    expected = bp * math.exp(logsum)
    assert abs(sc - expected) < 1e-12
    assert abs(result.payload["bp"] - bp) < 1e-12
    assert result.payload["clipped"] == result.payload["total"]


def test_bleuS_edge():
    """Test edge cases."""
    # max_n=1 is the smallest valid value of the N parameter; verify it is accepted
    # and that the payload shape reflects a single n-gram order.
    candidate = "a a a a a a"
    references = ["a a a a a a"]
    result = bleu(candidate, references, max_n=1)
    sc = result.payload["bleu"]
    assert 0.0 <= sc <= 1.0
    assert len(result.payload["p_n"]) == 1
    # independence check: p_1 is 1.0 and c == r, so bp == 1.
    import math
    c = result.payload["c"]
    r = result.payload["r"]
    bp = 1.0 if c > r else math.exp(1.0 - (r + 0.0) / c)
    w = 1.0 / result.payload["max_n"]
    logsum = sum(w * math.log(p) for p in result.payload["p_n"])
    expected = bp * math.exp(logsum)
    assert abs(sc - expected) < 1e-12
    assert isinstance(result.summary_lines, list)
