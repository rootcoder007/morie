"""Tests for morie.fn.hetlc: values recomputed from the definition."""

from morie.fn.hetlc import heterozygosity_locus


def test_per_marker_and_per_individual():
    G = [[0, 1, 2, 1], [1, 1, 2, 0], [2, 0, 1, 1], [1, 1, 0, 1], [0, 2, 1, 1]]
    r = heterozygosity_locus(G)
    for j in range(4):
        col = [row[j] for row in G]
        p = sum(col) / 10
        assert abs(r["allele_freq"][j] - p) < 1e-15
        assert abs(r["H_exp"][j] - 2 * p * (1 - p)) < 1e-15
        assert r["H_obs"][j] == col.count(1) / 5
    for i, row in enumerate(G):
        assert r["het_freq_individual"][i] == row.count(1) / 4


def test_missing_calls_skipped_and_bad_codes_rejected():
    import pytest

    r = heterozygosity_locus([[0, None], [1, 2], [2, 1]])
    assert r["H_obs"][1] == 0.5
    with pytest.raises(ValueError):
        heterozygosity_locus([[0, 3]])
