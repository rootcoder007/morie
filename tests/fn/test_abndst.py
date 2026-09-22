"""Tests for abndst.abundance_estimation."""

from morie.fn import _array_core as np

from morie.fn.abndst import abundance_estimation


def test_abndst_basic():
    """Test basic functionality."""
    # kmer_distribution: shape (n_nodes, n_species); columns sum to 1
    # P(i|j): row=node, column=species
    P = np.array([[0.5, 0.0],
                  [0.0, 0.5],
                  [0.5, 0.5]])
    reads = np.array([100.0, 300.0, 400.0])
    result = abundance_estimation(reads, P)

    # Documented return keys
    assert isinstance(result, dict)
    for key in ("estimate", "fractions", "reads_reassigned",
                "iterations", "converged", "identifiable",
                "naive_species_reads", "log_likelihood"):
        assert key in result, f"missing key {key!r}"

    # Independent recomputation from the documented formula.
    # Two species, columns sum to 1, all rows have at least one
    # positive entry so no reads are stranded (usable = total = 800).
    # Closed form for this P: theta -> (200/800, 600/800) = (0.25, 0.75).
    est = result["estimate"]
    assert est.shape == (2,)
    assert np.allclose(est, np.array([200.0, 600.0]))
    assert np.allclose(result["fractions"], np.array([0.25, 0.75]))
    # Reads reassigned = usable - naive_species_reads
    # Naive: node 0 -> sp 0 (100 reads), node 1 -> sp 1 (300 reads).
    # Node 2 is shared (unique_node=False), so naive = [100, 300].
    # reassigned = 800 - 400 = 400.
    assert np.allclose(result["naive_species_reads"], np.array([100.0, 300.0]))
    assert result["reads_reassigned"] == 400.0
    assert result["converged"] is True
    assert result["identifiable"] is True


def test_abndst_edge():
    """Test edge cases: threshold filtering of low-abundance species."""
    # Same two-species setup but with a tiny extra species that has
    # a very small column to exercise the threshold path. Use a
    # three-species matrix whose columns each sum to 1.
    P = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    reads = np.array([50.0, 50.0, 50.0])
    result = abundance_estimation(reads, P, threshold=10.0)

    assert isinstance(result, dict)
    # Columns are unit vectors and reads match them exactly, so each
    # species gets 50 reads after redistribution. With threshold=10,
    # none are filtered (50 >= 10).
    assert np.allclose(result["estimate"], np.array([50.0, 50.0, 50.0]))
    # n_filtered is not a documented key, but we can check that the
    # raw estimate is preserved (no reads dropped silently):
    assert np.sum(result["estimate"] >= 10.0) == 3
    assert result["converged"] is True
    # No stranded reads, so reassigned == 0.
    assert result["reads_reassigned"] == 0.0
