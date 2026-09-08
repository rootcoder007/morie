"""Tests for tknbp.bpe_tokenizer."""

import pytest

from morie.fn.tknbp import bpe_tokenizer


def test_tknbp_learns_the_most_frequent_pair_first():
    """Corpus dominated by 'ab': the first merge must be ('a', 'b') --
    the greedy rule IS the algorithm (Sennrich et al. 2016)."""
    r = bpe_tokenizer(["abab", "abc", "ab"], num_merges=1)
    assert len(r["merges"]) == 1
    assert tuple(r["merges"][0]) == ("a", "b")
    assert "ab" in r["vocab"]


def test_tknbp_merges_cascade_into_longer_units():
    """With enough merges on repeats of one word, the whole word becomes a
    single vocabulary unit."""
    r = bpe_tokenizer(["low"] * 5 + ["lower"] * 2, num_merges=6)
    # Words carry an end-of-word marker, so the whole-word unit is
    # "low</w>" (the Sennrich et al. convention).
    assert any(v.startswith("low") for v in r["vocab"])
    assert r["n_merges"] <= 6


def test_tknbp_merge_count_is_capped_by_available_pairs():
    r = bpe_tokenizer(["ab"], num_merges=50)
    # Two merges exist -- ('a','b') then ('ab','</w>') -- and then the
    # learner must stop, not loop to 50.
    assert r["n_merges"] == 2


def test_tknbp_empty_corpus_is_a_clean_empty_result():
    r = bpe_tokenizer([], num_merges=5)
    assert r["merges"] == [] and r["n_merges"] == 0
