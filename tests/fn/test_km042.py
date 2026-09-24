"""Tests for km042.kamath_ch3_prompt_label_mapping."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.km042 import kamath_ch3_prompt_label_mapping


def test_km042_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    answer_words = ["great", "terrible", "okay", "bad", "amazing"]
    raw = rng.uniform(0, 1, len(answer_words))
    total = float(np.sum(raw))
    probs = [float(p) / total for p in raw]
    x = {w: p for w, p in zip(answer_words, probs)}
    M = {"positive": "great", "negative": "terrible", "neutral": "okay"}
    y = "positive"
    result = kamath_ch3_prompt_label_mapping(x, y, M)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "label_probs" in result
    assert result["label"] == y
    assert result["answer_word"] == M[y]
    assert result["estimate"] == x[M[y]]
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["label_mass"])
    assert result["n"] == len(x)
    assert set(result["label_probs"].keys()) == set(M.keys())


def test_km042_edge():
    """Test edge cases: full coverage of x by M so label_mass == 1."""
    rng = np.random.default_rng(43)
    answer_words = ["yes", "no", "maybe"]
    raw = rng.uniform(0, 1, len(answer_words))
    total = float(np.sum(raw))
    probs = [float(p) / total for p in raw]
    x = {w: p for w, p in zip(answer_words, probs)}
    M = {"yes": "yes", "no": "no", "maybe": "maybe"}
    y = "no"
    result = kamath_ch3_prompt_label_mapping(x, y, M)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "label_probs" in result
    assert result["label_mass"] == pytest.approx(1.0)
    assert result["estimate"] == x["no"]
    assert result["answer_word"] == "no"
    assert math.isfinite(result["estimate"])
