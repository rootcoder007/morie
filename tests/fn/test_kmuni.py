"""Tests for kmuni.kamath_unigram_lm_tokenizer."""

from morie.fn import _array_core as np

from morie.fn.kmuni import kamath_unigram_lm_tokenizer

import math
import pytest


def test_kmuni_basic():
    """Test basic functionality."""
    corpus = ["a", "b", "ab", "aa", "bb"]
    vocab = ["a", "b", "ab"]
    result = kamath_unigram_lm_tokenizer(corpus, vocab)
    # The result is a RichResult (dict-like); check structure
    assert isinstance(result, dict)
    expected_keys = {
        "probs",
        "log_likelihood",
        "log_likelihood_history",
        "n_iterations",
        "segmentations",
        "vocab_size",
    }
    for key in expected_keys:
        assert key in result
    # probabilities sum to 1
    total = sum(result["probs"].values())
    assert abs(total - 1.0) < 1e-12
    # log-likelihood is finite
    assert math.isfinite(result["log_likelihood"])
    # n_iterations is a positive integer
    assert isinstance(result["n_iterations"], int)
    assert result["n_iterations"] >= 1
    # log-likelihood history is non-decreasing (EM property)
    hist = result["log_likelihood_history"]
    assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))
    # segmentations is a list of segmentations
    assert isinstance(result["segmentations"], list)
    assert all(isinstance(seg, list) for seg in result["segmentations"])
    # vocab_size matches the vocabulary size
    assert result["vocab_size"] == len(vocab)


def test_kmuni_edge():
    """Test edge cases."""
    # Valid minimal input
    corpus = ["a"]
    vocab = ["a"]
    result = kamath_unigram_lm_tokenizer(corpus, vocab)
    assert isinstance(result, dict)
    assert result["vocab_size"] == 1
    # n_iterations should be bounded by max_iter (default 100)
    assert result["n_iterations"] <= 100

    # Empty corpus should raise ValueError
    with pytest.raises(ValueError):
        kamath_unigram_lm_tokenizer([], ["a"])
    # Empty vocabulary should raise ValueError
    with pytest.raises(ValueError):
        kamath_unigram_lm_tokenizer(["a"], [])
    # Empty string in vocabulary should raise ValueError
    with pytest.raises(ValueError):
        kamath_unigram_lm_tokenizer(["a"], ["a", ""])


def test_every_printed_value_in_the_worked_example_reproduces():
    """Replicate the worked example from the docstring."""
    out = kamath_unigram_lm_tokenizer(["ab"], ["a", "b"])
    # probabilities are equal and round to 0.5
    assert [round(out["probs"][w], 12) for w in ["a", "b"]] == [0.5, 0.5]
    # log-likelihood equals log(0.25)
    assert abs(out["log_likelihood"] - math.log(0.25)) < 1e-12
    # log-likelihood history is non-decreasing
    hist = out["log_likelihood_history"]
    assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))
    # With repeated corpus and "ab" in vocab, the probability of "ab" exceeds 0.9
    em = kamath_unigram_lm_tokenizer(["ab", "ab"], ["a", "b", "ab"])
    assert em["probs"]["ab"] > 0.9
    # The segmentation of the first corpus string is ["ab"]
    assert em["segmentations"][0] == ["ab"]
