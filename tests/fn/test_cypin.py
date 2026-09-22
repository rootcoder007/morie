"""Tests for cypin.cyp450_inhibition."""

from morie.fn import _array_core as np

from morie.fn.cypin import cyp450_inhibition


def test_cypin_basic():
    """Test basic functionality."""
    smiles = "CCO"
    isozyme = "1A2"
    result = cyp450_inhibition(smiles, isozyme)
    assert isinstance(result, dict)
    assert "descriptors" in result
    assert "names" in result
    assert "named" in result
    assert "predicted" in result
    assert "inhibits" in result
    assert "reason" in result
    assert "isozyme" in result
    assert "n_descriptors" in result
    assert "has_model" in result
    assert "method" in result
    assert result["predicted"] is None
    assert result["has_model"] is False
    assert result["inhibits"] is None
    assert result["isozyme"] == "1A2"
    assert result["n_descriptors"] == len(result["descriptors"])
    assert len(result["names"]) == result["n_descriptors"]
    for k in result["names"]:
        assert k in result["named"]
        assert result["named"][k] == result["descriptors"][result["names"].index(k)]
    assert result["reason"]


def test_cypin_edge():
    """Test edge cases."""
    smiles = "CCO"
    isozyme = "3A4"
    result = cyp450_inhibition(smiles, isozyme)
    assert isinstance(result, dict)
    assert result["isozyme"] == "3A4"
    assert result["predicted"] is None
    assert result["has_model"] is False

    isozyme2 = "2D6"
    result2 = cyp450_inhibition("c1ccccc1O", isozyme2)
    assert isinstance(result2, dict)
    assert result2["isozyme"] == "2D6"
    assert result2["predicted"] is None
    assert result2["has_model"] is False
