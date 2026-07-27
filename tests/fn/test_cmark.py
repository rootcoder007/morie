"""Tests for cmark.causal_markov_condition."""

import numpy as np
import pytest

from morie.fn.cmark import causal_markov_condition


def test_cmark_basic():
    out = causal_markov_condition({"X": ["Y"], "Y": ["Z"]})
    assert out["implied"] == [("Z", "X", ("Y",))]
    rng = np.random.default_rng(42)
    x = rng.normal(size=2000)
    y = x + rng.normal(scale=0.7, size=2000)
    z = y + rng.normal(scale=0.7, size=2000)
    assert causal_markov_condition({"X": ["Y"], "Y": ["Z"]}, {"X": x, "Y": y, "Z": z})["holds"] is True


def test_cmark_edge():
    with pytest.raises(ValueError):
        causal_markov_condition({"A": ["B"], "B": ["A"]})  # cycle
    with pytest.raises(ValueError):
        causal_markov_condition({"X": ["Y"], "Y": ["Z"]}, {"X": [1.0, 2.0]})  # missing nodes
