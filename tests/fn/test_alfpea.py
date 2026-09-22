"""Tests for alfpea.alphafold_pae_predict."""

from morie.fn import _array_core as np

from morie.fn.alfpea import alphafold_pae_predict


def test_alfpea_basic():
    """Test basic functionality with uniform pair representation."""
    n = 4
    cz = 3
    nbins = 4
    bins = [1.0, 3.0, 5.0, 7.0]

    # All-zero pair representation z (n x n x cz): uniform logits -> uniform p
    z = [[[0.0] * cz for _ in range(n)] for _ in range(n)]

    # Weight matrix w: nbins x cz, also all zeros -> uniform distribution.
    w = [[0.0] * cz for _ in range(nbins)]

    result = alphafold_pae_predict(z, w, bins=bins)

    # The function returns a RichResult which behaves like a dict.
    assert isinstance(result, dict)

    # Documented keys must all be present.
    for key in ("pae", "p", "estimate", "n", "method"):
        assert key in result

    # Shape checks.
    assert result["n"] == n
    assert len(result["pae"]) == n
    assert len(result["p"]) == n
    for row in result["pae"]:
        assert len(row) == n
    for row in result["p"]:
        assert len(row) == n
    for row in result["p"]:
        for dist in row:
            assert len(dist) == nbins

    # With zero weights everywhere, softmax is uniform: p[i][j][b] = 1/nbins.
    # The docstring guarantees "zero weights give exactly the mean of the bins".
    expected_mean_of_bins = sum(bins) / nbins

    for i in range(n):
        for j in range(n):
            for b in range(nbins):
                assert result["p"][i][j][b] == 1.0 / nbins
            assert result["pae"][i][j] == expected_mean_of_bins

    # Estimate is the mean of the full flat pae matrix (n*n entries).
    assert result["estimate"] == expected_mean_of_bins
    assert result["method"] == "AlphaFold predicted aligned error"


def test_alfpea_edge():
    """Test edge case: n=1 (single residue)."""
    n = 1
    cz = 2
    nbins = 3
    bins = [0.5, 1.5, 2.5]

    # Single pair entry with known values.
    z = [[[1.0, -1.0]]]

    # Construct w so that we can hand-verify the softmax for i=j=0.
    # lin(z[0][0], w) = z[0][0][0]*w[b][0] + z[0][0][1]*w[b][1]
    w = [[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]

    # Independent computation of the expected distribution.
    logits = [z[0][0][0] * w[b][0] + z[0][0][1] * w[b][1] for b in range(nbins)]
    # logits = [2.0, -2.0, 0.0]
    m = max(logits)
    exps = [pow(2.718281828459045, l - m) for l in logits]
    s_exp = sum(exps)
    expected_p = [e / s_exp for e in exps]
    expected_pae = sum(expected_p[b] * bins[b] for b in range(nbins))

    result = alphafold_pae_predict(z, w, bins=bins)

    assert isinstance(result, dict)
    assert result["n"] == n
    assert len(result["pae"]) == 1
    assert len(result["pae"][0]) == 1
    assert len(result["p"]) == 1
    assert len(result["p"][0]) == 1
    assert len(result["p"][0][0]) == nbins

    # Each distribution sums to 1.
    assert abs(sum(result["p"][0][0]) - 1.0) < 1e-9
    # PAE is the convex combination of bin centres.
    assert abs(result["pae"][0][0] - expected_pae) < 1e-9
    # Estimate equals the single PAE value for n=1.
    assert abs(result["estimate"] - expected_pae) < 1e-9
