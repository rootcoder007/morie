"""Tests for glove -- Pennington, Socher & Manning (2014).

Replaces a generated test that called a stub returning mean(corpus).
Full anchor: ledger/wave3/anchor_embed.py.
"""

import math

import pytest

from morie.fn.glove import cooccurrence, glove, glove_loss, glove_weight

DOCS = [["cat", "sat", "mat"] * 8, ["dog", "ran", "far"] * 8]


def cosine(a, b):
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    return sum(a[i] * b[i] for i in range(len(a))) / (na * nb)


def test_weighting_function_eq9():
    """Eq. (9) with the paper's x_max = 100 and alpha = 3/4."""
    assert glove_weight(0.0) == 0.0                      # property 1
    assert glove_weight(50.0) == pytest.approx(0.5 ** 0.75, abs=1e-15)
    assert glove_weight(100.0) == 1.0
    assert glove_weight(1e6) == 1.0
    # property 2: non-decreasing
    assert all(glove_weight(x) <= glove_weight(x + 1.0)
               for x in range(0, 200))


def test_harmonic_context_window():
    """Sec. 4.2: a token d positions away contributes 1/d."""
    X, _, idx = cooccurrence([["a", "b", "c"]], window=2)
    assert X[(idx["a"], idx["b"])] == pytest.approx(1.0)
    assert X[(idx["a"], idx["c"])] == pytest.approx(0.5)
    # symmetric by construction
    assert all(X[(i, j)] == pytest.approx(X[(j, i)]) for (i, j) in X)
    # a flat window is a different model
    Xf, _, idf = cooccurrence([["a", "b", "c"]], window=2,
                              harmonic=False)
    assert Xf[(idf["a"], idf["c"])] == pytest.approx(1.0)


def test_objective_falls_and_is_recomputable():
    """The reported loss must BE eq. (8), not an SGD running total."""
    g = glove(DOCS, dim=12, epochs=40, window=3, seed=1)
    assert g["final_loss"] < g["loss_history"][0]
    recomputed = glove_loss(g["cooccurrence"], g["W"], g["W_tilde"],
                            g["b"], g["b_tilde"])
    assert recomputed == pytest.approx(g["final_loss"], abs=1e-9)
    # the running total is kept separately and is a different number
    assert g["running_loss"][-1] != g["loss_history"][-1]


def test_cooccurring_words_end_up_closer():
    g = glove(DOCS, dim=12, epochs=40, window=3, seed=1)
    i = g["index"]
    assert (cosine(g["vectors"][i["cat"]], g["vectors"][i["sat"]])
            > cosine(g["vectors"][i["cat"]], g["vectors"][i["dog"]]))


def test_combine_choices_differ():
    a = glove(DOCS, dim=12, epochs=20, window=3, seed=1)
    b = glove(DOCS, dim=12, epochs=20, window=3, seed=1, combine="w")
    assert max(abs(a["vectors"][r][t] - b["vectors"][r][t])
               for r in range(len(a["vocab"]))
               for t in range(12)) > 1e-9


def test_argument_checks():
    with pytest.raises(ValueError):
        glove([["only"]], dim=4)
    with pytest.raises(ValueError):
        glove(DOCS, dim=0)
    with pytest.raises(ValueError):
        glove(DOCS, combine="nope")
    with pytest.raises(ValueError):
        cooccurrence(DOCS, window=0)
