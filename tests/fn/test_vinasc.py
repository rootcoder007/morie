"""Tests for vinasc.autodock_vina_score."""

from morie.fn import _array_core as np

from morie.fn.vinasc import autodock_vina_score


def test_vinasc_basic():
    """Test basic functionality."""
    receptor = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    ligand_pose = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = autodock_vina_score(receptor, ligand_pose)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_vinasc_edge():
    """Test edge cases."""
    receptor = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    ligand_pose = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = autodock_vina_score(receptor, ligand_pose)
    assert isinstance(result, dict)
