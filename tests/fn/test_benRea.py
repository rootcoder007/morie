"""Tests for benRea. Full anchor: ledger/wave3/anchor_nlp_family.py."""
import pytest
from morie.fn.benRea import (bio_labels, extract_spans, greedy_decode,
                             is_valid_bio, ner_decode, span_f1,
                             valid_transitions, viterbi_decode)

LABS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
# token 0 most wants O and token 1 most wants I-PER, so the greedy
# reading is invalid
EM = [[5.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 4.0, 0.0, 0.0],
      [3.0, 0.0, 0.0, 0.0, 0.0]]


def test_the_label_set_and_transition_rules():
    assert bio_labels(["PER", "LOC"]) == LABS
    T = valid_transitions(LABS)
    assert not T[LABS.index("O")][LABS.index("I-PER")]
    assert not T[LABS.index("B-LOC")][LABS.index("I-PER")]
    assert T[LABS.index("B-PER")][LABS.index("I-PER")]
    assert T[LABS.index("I-PER")][LABS.index("I-PER")]
    with pytest.raises(ValueError):
        bio_labels(["PER", "PER"])


def test_viterbi_is_valid_where_greedy_is_not():
    """And it never outscores greedy on emissions -- that gap IS the
    constraint. A Viterbi that beat greedy would mean it was not
    applied."""
    g = greedy_decode(EM, LABS)
    v, score = viterbi_decode(EM, LABS)
    gscore = sum(EM[t][LABS.index(g[t])] for t in range(3))
    assert not is_valid_bio(g)
    assert is_valid_bio(v)
    assert score <= gscore + 1e-12


def test_spans_need_the_type_and_both_boundaries():
    assert extract_spans(["B-PER", "I-PER", "O", "B-LOC"]) == [
        ("PER", 0, 1), ("LOC", 3, 3)]
    f = span_f1(["B-PER", "I-PER", "O"], ["B-PER", "O", "O"])
    assert f["f1"] == 0.0
    assert f["true_positives"] == 0
    assert span_f1(["B-PER", "I-PER"], ["B-PER", "I-PER"])["f1"] == 1.0


def test_the_wrapper_reports_validity_and_scores():
    r = ner_decode(EM, ["PER", "LOC"], gold=["O", "O", "O"])
    assert r["valid"]
    assert r["decoder"] == "viterbi"
    assert "f1" in r
    with pytest.raises(ValueError):
        ner_decode(EM, ["PER"], decoder="nope")
    with pytest.raises(ValueError):
        viterbi_decode([[1.0, 2.0], [1.0]], LABS)
