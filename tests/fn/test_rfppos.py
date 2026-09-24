"""Tests for rfppos.reactive_pose_filter."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.rfppos import reactive_pose_filter


def test_rfppos_basic():
    """Test basic functionality."""
    # A simple ligand with a carbonyl warhead: acetaldehyde (CC=O)
    pose = {
        "smiles": "CC=O",
        "coords": [
            [0.0, 0.0, 0.0],   # methyl carbon
            [1.5, 0.0, 0.0],   # carbonyl carbon
            [2.0, 1.0, 0.0],   # oxygen
        ],
    }
    # Cysteine residue: provide SG and optional CB
    cys_residue = {
        "SG": [2.5, 0.5, 0.0],
        "CB": [1.0, 1.0, 0.0],
    }
    # Provide warhead explicitly (electrophile, reference, torsion)
    warhead = (1, 2, 0)
    result = reactive_pose_filter(pose, cys_residue, warhead=warhead)
    # The result is a RichResult (dict-like)
    assert isinstance(result, dict)
    # Check that essential keys are present
    assert "passes" in result
    assert isinstance(result["passes"], bool)
    assert "distance" in result
    assert math.isfinite(result["distance"])
    assert "angle" in result
    assert math.isfinite(result["angle"])
    # Ensure mode is recorded
    assert result["mode"] == "burgi_dunitz"
    # Warhead indices should match what we provided
    assert result["electrophile"] == 1
    assert result["reference"] == 2
    assert result["torsion_atom"] == 0


def test_rfppos_edge():
    """Test edge cases: invalid mode raises ValueError."""
    pose = {
        "smiles": "CC=O",
        "coords": [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
    }
    cys_residue = {
        "SG": [2.5, 0.5, 0.0],
        "CB": [1.0, 1.0, 0.0],
    }
    with pytest.raises(ValueError):
        reactive_pose_filter(pose, cys_residue, mode="not_a_mode")
