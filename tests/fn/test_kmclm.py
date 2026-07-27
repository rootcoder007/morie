"""Tests for kmclm.kamath_causal_lm_loss."""

import numpy as np
import pytest

from morie.fn.kmclm import kamath_causal_lm_loss


def test_kmclm_basic():
    V, T = 8, 5
    uniform = np.zeros((T, V))
    result = kamath_causal_lm_loss(uniform, np.arange(T) % V)
    assert result["loss"] == pytest.approx(np.log(V))
    assert result["perplexity"] == pytest.approx(V)


def test_kmclm_edge():
    V, T = 4, 3
    logits = np.full((T, V), -50.0)
    tgt = np.arange(T) % V
    logits[np.arange(T), tgt] = 50.0
    assert kamath_causal_lm_loss(logits, tgt)["loss"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        kamath_causal_lm_loss(logits, np.full(T, V))  # id out of range
