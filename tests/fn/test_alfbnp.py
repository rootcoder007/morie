"""Tests for alfbnp.af3_protein_ligand."""

from morie.fn import _array_core as np

from morie.fn.alfbnp import af3_protein_ligand


def test_alfbnp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 8
    clean = [rng.normal(0, 1, (n, 3)) for _ in range(2)]
    result = af3_protein_ligand(
        n_atoms=n,
        clean=clean,
        steps=4,
        s_max=80.0,
        s_min=4e-4,
        seed=2,
    )
    assert isinstance(result.coords, list)
    assert len(result.coords) == n
    assert all(len(p) == 3 for p in result.coords)
    assert len(result.sigmas) == 5  # steps + 1 (initial sigma included)
    assert len(result.trace) == 4   # one trace entry per step
    assert result.denoiser_coefs is not None


def test_alfbnp_edge():
    """Test deterministic reproduction with the same seed."""
    rng = np.random.default_rng(42)
    n = 5
    clean = [rng.normal(0, 1, (n, 3))]
    r1 = af3_protein_ligand(n_atoms=n, clean=clean, steps=3, seed=2)
    r2 = af3_protein_ligand(n_atoms=n, clean=clean, steps=3, seed=2)
    assert r1.sigmas == r2.sigmas
    assert all(r1.sigmas[i] > r1.sigmas[i + 1] for i in range(len(r1.sigmas) - 1))
