"""Tests for hmmprf.hmm_profile."""

import math

from morie.fn import _array_core as np

from morie.fn.hmmprf import hmm_profile


def _make_profile(rng, L, A):
    """Build a profile HMM with L match positions and alphabet size A."""
    # match: L x A emission probabilities (strictly positive, rows normalised)
    match = []
    for _ in range(L):
        row = [rng.uniform(0.1, 1.0) for _ in range(A)]
        s = sum(row)
        row = [v / s for v in row]
        match.append(row)

    # insert: length-A background emission probabilities
    ins = [rng.uniform(0.1, 1.0) for _ in range(A)]
    s = sum(ins)
    ins = [v / s for v in ins]

    # transition probabilities
    trans = {
        "mm": 0.9, "mi": 0.05, "md": 0.05,
        "im": 0.1, "ii": 0.9,
        "dm": 0.2, "dd": 0.8,
    }

    return {"match": match, "insert": ins, "trans": trans}


def test_hmmprf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)

    A = 4
    L = 5
    T = 20

    seq = rng.integers(0, A, T)
    profile = _make_profile(rng, L, A)

    result = hmm_profile(seq, profile)
    assert isinstance(result, dict)
    assert "forward_logprob" in result
    assert "viterbi_logprob" in result
    assert "log_odds" in result
    assert "background_logprob" in result
    assert "n" in result
    assert math.isfinite(result["forward_logprob"])
    assert math.isfinite(result["viterbi_logprob"])
    assert math.isfinite(result["log_odds"])
    assert math.isfinite(result["background_logprob"])
    assert result["n"] == T
    # The forward log-probability is a log-sum-exp over all paths, so it is
    # always at least as large as the single best path returned by Viterbi.
    assert result["forward_logprob"] >= result["viterbi_logprob"]


def test_hmmprf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)

    # Minimal but valid profile and sequence.
    A = 2
    L = 1
    T = 3

    seq = rng.integers(0, A, T)
    profile = _make_profile(rng, L, A)

    result = hmm_profile(seq, profile)
    assert isinstance(result, dict)
    assert "forward_logprob" in result
    assert "viterbi_logprob" in result
    assert "log_odds" in result
    assert math.isfinite(result["forward_logprob"])
    assert math.isfinite(result["viterbi_logprob"])
    assert result["n"] == T
