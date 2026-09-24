"""Tests for goldsc.gold_score."""

import math

from morie.fn import _array_core as np

from morie.fn.goldsc import gold_score


def test_goldsc_basic():
    """Test basic functionality with a small receptor-ligand complex."""
    rng = np.random.default_rng(42)

    # Each atom row: [x, y, z, type]
    receptor = [
        [float(rng.normal(0, 1)),
         float(rng.normal(0, 1)),
         float(rng.normal(0, 1)),
         "C"]
        for _ in range(5)
    ]
    ligand = [
        [float(rng.normal(2, 1)),
         float(rng.normal(2, 1)),
         float(rng.normal(2, 1)),
         "N"]
        for _ in range(4)
    ]

    radii = [("C", 1.7), ("N", 1.55)]
    depths = [("C", 0.1), ("N", 0.1)]

    result = gold_score(receptor, ligand, radii=radii, depths=depths)

    assert isinstance(result, dict)
    assert "fitness" in result
    assert "energy" in result
    assert "estimate" in result
    assert "n_receptor" in result
    assert "n_ligand" in result
    assert result["n_receptor"] == 5
    assert result["n_ligand"] == 4
    assert math.isfinite(result["fitness"])
    assert math.isfinite(result["energy"])


def test_goldsc_edge():
    """Test edge case with the minimal valid receptor and ligand."""
    # One carbon atom in the receptor and one in the ligand, well separated.
    receptor = [[0.0, 0.0, 0.0, "C"]]
    ligand = [[5.0, 0.0, 0.0, "C"]]

    radii = [("C", 1.7)]
    depths = [("C", 0.1)]

    result = gold_score(receptor, ligand, radii=radii, depths=depths)

    assert isinstance(result, dict)
    assert "fitness" in result
    assert "energy" in result
    assert result["n_receptor"] == 1
    assert result["n_ligand"] == 1
    assert math.isfinite(result["fitness"])
    assert math.isfinite(result["energy"])
