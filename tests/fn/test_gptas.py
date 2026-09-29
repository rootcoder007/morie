"""Tests for morie.fn.gptas: beam search against exhaustive enumeration."""

import itertools
import math

from morie.fn.gptas import gpt_assistant_decode


def _model(src, prefix):
    return [math.log(0.6), math.log(0.4)] if len(prefix) % 2 == 0 else [math.log(0.3), math.log(0.7)]


def test_beam_equals_exhaustive_best_for_a_wide_beam():
    r = gpt_assistant_decode(_model, "hi", k=8, max_len=3)
    best = max(itertools.product([0, 1], repeat=3), key=lambda s: sum(_model("hi", s[:i])[t] for i, t in enumerate(s)))
    assert tuple(r["sequence"]) == best
    assert abs(r["score"] - sum(_model("hi", best[:i])[t] for i, t in enumerate(best))) < 1e-12
