"""Tests for alfrf2.rfdiffusion_protein."""

from morie.fn import _array_core as np

from morie.fn.alfrf2 import rfdiffusion_protein


def _build_motif(seed=42):
    """Build a target_motif of (residue_index, [x, y, z]) pairs.

    Uses 4 motif residues at positions 5, 12, 20, 30 in a design of
    40 residues, with coordinates on a sphere of radius 5.0 centred
    at the origin. Four non-coplanar motif points are enough that
    `rmsd` has a unique superposition.
    """
    rng = np.random.default_rng(seed)
    n_motif = 4
    indices = [5, 12, 20, 30]
    coords = []
    for _ in range(n_motif):
        v = rng.normal(0.0, 1.0, 3)
        norm = float(np.sqrt(np.sum(v * v)))
        coords.append([5.0 * float(v[k]) / norm for k in range(3)])
    return list(zip(indices, coords))


def test_alfrf2_basic():
    """Test basic functionality with a well-formed motif and integer scaffold."""
    target_motif = _build_motif(seed=42)
    scaffold = 40
    result = rfdiffusion_protein(target_motif, scaffold, T=20, seed=42)
    assert isinstance(result, dict)
    # The function returns a RichResult whose documented keys must be present.
    for key in ("backbone", "motif_index", "motif_target", "motif_placed",
                "motif_max_deviation", "motif_rmsd", "spacing",
                "mean_spacing", "radius_of_gyration", "trace",
                "n", "n_motif", "T", "denoise", "noise_scale", "seed",
                "method"):
        assert key in result, f"missing documented key: {key}"
    # Shape: backbone must have one entry per residue, each a 3-vector.
    assert len(result["backbone"]) == 40
    for row in result["backbone"]:
        assert len(row) == 3
    # n and n_motif must reflect what we asked for.
    assert result["n"] == 40
    assert result["n_motif"] == 4
    # The motif must land exactly: the placed coordinates equal the targets,
    # so the maximum deviation is zero. This is a direct consequence of the
    # closed-form noising of the motif to abar[0], which is zero, so the
    # formula `nxt[i] = forward_noise(m0, abar[t-1], ...)` collapses to m0[i]
    # at t=1. For t>1 the loop only overwrites the fixed residues with the
    # noised motif, so the final state has the motif at its targets.
    assert result["motif_max_deviation"] == 0.0
    # Same length of trace as diffusion steps.
    assert len(result["trace"]) == 20
    # Spacing list has one fewer entry than the backbone.
    assert len(result["spacing"]) == 39


def test_alfrf2_edge():
    """Test edge cases: scaffold given as a starting structure."""
    target_motif = _build_motif(seed=42)
    n = 40
    rng = np.random.default_rng(7)
    # Build a starting scaffold of shape (n, 3) so scaffold is treated as
    # a starting structure (not an integer count).
    scaffold = [[float(rng.normal()) for _ in range(3)] for _ in range(n)]
    result = rfdiffusion_protein(target_motif, scaffold, T=20, seed=42)
    assert isinstance(result, dict)
    # The motif lands exactly regardless of whether the scaffold was given
    # as an int or a starting structure.
    assert result["motif_max_deviation"] == 0.0
    assert result["n"] == n
    assert result["n_motif"] == 4
