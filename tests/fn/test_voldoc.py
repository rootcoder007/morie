"""Tests for voldoc."""

import numpy as np
import pytest

from morie.fn.voldoc import vol_decomposed_realised


def test_voldoc_basic():
    out = vol_decomposed_realised(2.0, 1.5)
    assert out["jump"] == pytest.approx(0.5)
    assert out["continuous"] == pytest.approx(1.5)


def test_voldoc_edge():
    out = vol_decomposed_realised([1.0], [1.4])  # BPV above RV -> J = 0
    assert out["jump"] == pytest.approx([0.0])
    with pytest.raises(ValueError):
        vol_decomposed_realised([-1.0], [1.0])
