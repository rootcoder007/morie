"""Tests for scfhop.scaffold_hop."""

import math

from morie.fn import _array_core as np

from morie.fn.scfhop import scaffold_hop


def test_scfhop_basic():
    """Test basic functionality."""
    lead_smiles = "CC"
    scaffold_db = ["CC", "C", "CCC", "CCCC", "CCCCC"]
    result = scaffold_hop(lead_smiles, scaffold_db)
    assert isinstance(result, dict)
    # Check that all expected keys are present
    expected_keys = [
        "lead", "lead_scaffold", "lead_scaffold_size", "lead_signature",
        "ranked", "similarity", "is_hop", "order", "n_candidates",
        "n_hops", "n_dim", "maxdist", "scaling", "metric", "rounds",
        "threshold", "method"
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
    # Basic type and length checks
    assert result["n_candidates"] == len(scaffold_db)
    assert len(result["ranked"]) == len(scaffold_db)
    assert len(result["similarity"]) == len(scaffold_db)
    assert len(result["is_hop"]) == len(scaffold_db)
    assert len(result["order"]) == len(scaffold_db)
    # Similarities should be finite floats in [0,1]
    for sim in result["similarity"]:
        assert isinstance(sim, (int, float))
        assert math.isfinite(sim)
        assert 0.0 <= sim <= 1.0
    # is_hop should be booleans
    for hop in result["is_hop"]:
        assert isinstance(hop, bool)
    # Ranking should be sorted by similarity descending (ties broken by original order)
    sims = result["similarity"]
    for i in range(len(sims) - 1):
        assert sims[i] >= sims[i+1], f"Similarities not non-increasing: {sims[i]} vs {sims[i+1]}"
    # Each ranked entry should have the expected keys
    for row in result["ranked"]:
        assert "index" in row
        assert "smiles" in row
        assert "similarity" in row
        assert "scaffold_differs" in row
        assert "scaffold_size" in row
        assert "is_hop" in row
        # Also check types
        assert isinstance(row["similarity"], (int, float))
        assert isinstance(row["is_hop"], bool)
        assert isinstance(row["index"], int)
        assert isinstance(row["smiles"], str)
        assert isinstance(row["scaffold_size"], int)
    # Lead descriptor and scaffold info
    assert isinstance(result["lead"], list)
    assert isinstance(result["lead_scaffold"], list)
    # n_hops should equal sum of is_hop
    assert result["n_hops"] == sum(result["is_hop"])
    assert 0 <= result["n_hops"] <= result["n_candidates"]
    # Method is a string
    assert isinstance(result["method"], str)


def test_scfhop_edge():
    """Test edge cases."""
    lead_smiles = "C"
    scaffold_db = []
    result = scaffold_hop(lead_smiles, scaffold_db)
    assert isinstance(result, dict)
    # With no candidates, these should be empty
    assert result["n_candidates"] == 0
    assert result["ranked"] == []
    assert result["similarity"] == []
    assert result["is_hop"] == []
    assert result["order"] == []
    # Lead info should still be present
    assert "lead" in result
    assert "lead_scaffold" in result
    assert "lead_signature" in result
    assert "method" in result
