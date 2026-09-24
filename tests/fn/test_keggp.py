"""Tests for keggp.kegg_pathway."""

import math

from morie.fn import _array_core as np

from morie.fn.keggp import kegg_pathway


def test_keggp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    k = 5
    # 0/1 indicator over the gene universe
    genes = [float(v) for v in rng.integers(0, 2, n)]
    # ensure at least one selected gene so the test is valid
    if sum(genes) == 0:
        genes[0] = 1.0
    # membership matrix: one row per gene, one column per pathway
    kegg_pathways = [[float(rng.integers(0, 2)) for _ in range(k)] for _ in range(n)]
    result = kegg_pathway(genes, kegg_pathways)
    assert isinstance(result, dict)
    for key in ("estimate", "pvalue", "qvalue", "overlap", "pathway_size",
                "top_pathway", "n_significant", "significant", "n_selected",
                "n_pathways", "alpha", "n", "method"):
        assert key in result
    # p-values and q-values must be valid probabilities
    assert all(0.0 <= v <= 1.0 for v in result["pvalue"])
    assert all(0.0 <= v <= 1.0 for v in result["qvalue"])
    # per-pathway arrays must have one entry per pathway
    assert len(result["pvalue"]) == k
    assert len(result["qvalue"]) == k
    assert len(result["overlap"]) == k
    assert len(result["pathway_size"]) == k
    assert len(result["significant"]) == k
    # significant entries are coded as 0/1
    assert all(s in (0.0, 1.0) for s in result["significant"])
    # estimate is the minimum raw p-value
    assert math.isfinite(result["estimate"])


def test_keggp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    k = 3
    genes = [float(v) for v in rng.integers(0, 2, n)]
    if sum(genes) == 0:
        genes[0] = 1.0
    kegg_pathways = [[float(rng.integers(0, 2)) for _ in range(k)] for _ in range(n)]
    # call with a custom FDR level
    result = kegg_pathway(genes, kegg_pathways, alpha=0.1)
    assert isinstance(result, dict)
    assert result["alpha"] == 0.1
    assert result["n"] == float(n)
    assert result["n_pathways"] == float(k)
    assert result["n_selected"] > 0
    assert result["n_significant"] >= 0
    assert all(s in (0.0, 1.0) for s in result["significant"])
    assert len(result["pvalue"]) == k
    assert len(result["significant"]) == k
