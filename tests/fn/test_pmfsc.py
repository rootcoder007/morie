"""Tests for pmfsc.pmf_potential."""

import math
import pytest

from morie.fn import _array_core as np
import morie.fn.pmfsc as pmfsc


def test_pmfsc_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    types = ["C", "N", "O"]
    # Build receptor: 5 atoms
    receptor = []
    for i in range(5):
        x = rng.normal(0, 1)
        y = rng.normal(0, 1)
        z = rng.normal(0, 1)
        t = types[i % len(types)]
        receptor.append([x, y, z, t])
    # Build ligand: 5 atoms
    ligand = []
    for i in range(5):
        x = rng.normal(5, 1)
        y = rng.normal(5, 1)
        z = rng.normal(5, 1)
        t = types[(i + 1) % len(types)]
        ligand.append([x, y, z, t])
    # Build observations: list of (rt, lt, dist) tuples
    observations = []
    for rt in types:
        for lt in types:
            for _ in range(10):
                dist = rng.uniform(1.0, 10.0)
                observations.append((rt, lt, dist))
    # Derive potential from observations
    potential = pmfsc.derive_potential(observations)
    # Score the pose with the derived potential
    result = pmfsc.pmf_potential(receptor, ligand, potential=potential)
    # Result is a RichResult (dict-like)
    assert isinstance(result, dict)
    # Check presence of key fields
    assert "score" in result
    assert "estimate" in result
    # Score should be finite
    assert math.isfinite(result["score"])
    # Check number of pairs
    assert result["n_pairs"] == len(receptor) * len(ligand)


def test_pmfsc_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    receptor = [[0.0, 0.0, 0.0, "C"]]
    ligand = [[1.0, 1.0, 1.0, "N"]]
    # Calling without potential or observations should raise ValueError
    with pytest.raises(ValueError):
        pmfsc.pmf_potential(receptor, ligand)
