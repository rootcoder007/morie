"""Tests for km043.kamath_ch3_prompt_softmax_label."""

import math

from morie.fn import _array_core as np

from morie.fn.km043 import kamath_ch3_prompt_softmax_label


def test_km043_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    words = ["great", "terrible", "okay", "meh"]
    w = {word: list(rng.normal(0, 1, 3)) for word in words}
    M = {
        "pos": "great",
        "neg": "terrible",
        "neu": "okay",
        "mid": "meh",
    }
    h_z = list(rng.normal(0, 1, 3))
    result = kamath_ch3_prompt_softmax_label(w, h_z, M)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "label" in result
    assert "label_probs" in result
    assert "logits" in result
    assert "n" in result
    assert "method" in result
    # Probabilities should sum to 1.
    probs_sum = sum(result["label_probs"].values())
    assert math.isfinite(probs_sum)
    assert abs(probs_sum - 1.0) < 1e-9
    # Each label probability lies in [0, 1].
    for p in result["label_probs"].values():
        assert 0.0 <= p <= 1.0
    # The reported estimate is in [0, 1] and finite.
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    # Number of labels matches the size of M.
    assert result["n"] == len(M)
    # The reported label is the argmax of the probabilities.
    best = max(result["label_probs"], key=result["label_probs"].get)
    assert result["label"] == best
    # All labels from M appear in label_probs and logits.
    for lab in M:
        assert lab in result["label_probs"]
        assert lab in result["logits"]


def test_km043_edge():
    """Test with the minimal 2-label case from the docstring example."""
    w = {"great": [1.0, 0.0], "terrible": [0.0, 1.0]}
    M = {"pos": "great", "neg": "terrible"}
    h_z = [1.0, 0.0]
    result = kamath_ch3_prompt_softmax_label(w, h_z, M)
    assert isinstance(result, dict)
    assert result["n"] == 2
    assert result["label"] == "pos"
    # Derivable from the formula: p(pos) = 1 / (1 + exp(-1)).
    expected_pos = 1.0 / (1.0 + math.exp(-1.0))
    assert abs(result["label_probs"]["pos"] - expected_pos) < 1e-9
    assert abs(result["label_probs"]["neg"] - (1.0 - expected_pos)) < 1e-9
    # Logit for pos should be 1.0 (1.0 * 1.0 + 0.0 * 0.0) and for neg 0.0.
    assert abs(result["logits"]["pos"] - 1.0) < 1e-12
    assert abs(result["logits"]["neg"] - 0.0) < 1e-12
