"""Tests for copfra."""

import numpy as np
import pytest

from morie.fn.copfra import frank_copula

def test_copfra_basic():
    pos = frank_copula(0.5, 0.5, 5.0)
    neg = frank_copula(0.5, 0.5, -5.0)
    assert pos["tau"] > 0 > neg["tau"]
    assert pos["tau"] == pytest.approx(-neg["tau"])


def test_copfra_edge():
    with pytest.raises(ValueError):
        frank_copula(0.5, 0.5, 0.0)
