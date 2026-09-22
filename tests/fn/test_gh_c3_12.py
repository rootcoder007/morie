"""Tests for gh_c3_12.ghosal_polya_tree_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_12 import ghosal_polya_tree_def


def test_gh_c3_12_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_polya_tree_def(x)
    assert "estimate" in result
    assert "cell_mass" in result
    assert "depth" in result
    assert "method" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert result["depth"] == 8
    assert np.isfinite(result["cell_mass"])
    # density = cell_mass * 2**depth
    assert np.isclose(result["estimate"], result["cell_mass"] * 2.0 ** 8)


def test_gh_c3_12_edge():
    """Test edge cases."""
    result = ghosal_polya_tree_def(np.array([42.0]))
    assert "estimate" in result
    assert "cell_mass" in result
    assert "depth" in result
    # With a single-element input, the function uses xs[0] (42.0) and proceeds
    assert np.isfinite(result["estimate"])
    assert np.isclose(result["estimate"], result["cell_mass"] * 2.0 ** 8)


def test_gh_c3_12_reproducible():
    """Test reproducibility with a fixed seed."""
    x = np.array([0.7, 0.3])
    r1 = ghosal_polya_tree_def(x, depth=4, a_scale=1.0, seed=42)
    r2 = ghosal_polya_tree_def(x, depth=4, a_scale=1.0, seed=42)
    assert r1["estimate"] == r2["estimate"]
    assert r1["cell_mass"] == r2["cell_mass"]


def test_gh_c3_12_independent_formula():
    """Cross-check the documented formula with an independent computation."""
    depth = 6
    a_scale = 0.5
    seed = 7
    x = np.array([0.25])

    # Independent draw using the documented beta(alpha, alpha) splitting variables
    rng = np.random.default_rng(seed)
    # Convert x[0] to bits up to `depth` using the same flattening scheme
    # (we only need a deterministic bit string; 0.25 in [0,1) maps to 0.01 in binary)
    bits = []
    v = 0.25
    for _ in range(depth):
        v *= 2.0
        if v >= 1.0:
            bits.append(1)
            v -= 1.0
        else:
            bits.append(0)

    expected_mass = 1.0
    for m, b in enumerate(bits, start=1):
        a = a_scale * m * m
        V0 = float(rng.beta(a, a))
        expected_mass *= V0 if b == 0 else (1.0 - V0)
    expected_density = expected_mass * 2.0 ** depth

    res = ghosal_polya_tree_def(x, depth=depth, a_scale=a_scale, seed=seed)
    assert np.isclose(res["cell_mass"], expected_mass)
    assert np.isclose(res["estimate"], expected_density)
