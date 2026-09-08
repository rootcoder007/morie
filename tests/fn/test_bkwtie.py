"""Tests for bkwtie.burkov_weight_tying."""

from morie.fn.bkwtie import burkov_weight_tying


def test_bkwtie_basic():
    out = burkov_weight_tying([1.0, 0.0], [[2.0, 0.0], [0.0, 3.0]])
    assert out["logits"] == [2.0, 0.0]


def test_bkwtie_edge():
    import pytest
    with pytest.raises(ValueError, match="columns"):
        burkov_weight_tying([1.0, 2.0, 3.0], [[1.0, 2.0]])
