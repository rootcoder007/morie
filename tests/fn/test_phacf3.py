"""Tests for phacf3.pharmacophore_3d."""

from morie.fn import _array_core as np
from morie.fn.phacf3 import pharmacophore_3d


def test_phacf3_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    feature_set = ["A", "B", "C"]
    edges = [0.0, 3.0, 6.0, 10.0]
    mol_3d = []
    for _ in range(15):
        xyz = list(rng.normal(0, 1, 3))
        t = feature_set[int(rng.integers(0, len(feature_set)))]
        mol_3d.append([float(xyz[0]), float(xyz[1]), float(xyz[2]), t])
    result = pharmacophore_3d(mol_3d, feature_set, edges)
    assert isinstance(result, dict)
    assert "fingerprint" in result
    assert isinstance(result["fingerprint"], list)


def test_phacf3_edge():
    """Test edge cases."""
    feature_set = ["A", "B", "C"]
    edges = [0.0, 3.0, 6.0, 10.0]
    # Minimal molecule: exactly 3 points, which yields one triangle.
    mol_3d = [
        [0.0, 0.0, 0.0, "A"],
        [1.0, 0.0, 0.0, "B"],
        [0.0, 1.0, 0.0, "C"],
    ]
    result = pharmacophore_3d(mol_3d, feature_set, edges)
    assert isinstance(result, dict)
    assert "fingerprint" in result
    assert isinstance(result["fingerprint"], list)
