"""Tests for mafcl.maf_calculation."""

import pytest

from morie.fn.mafcl import maf_calculation


def test_mafcl_basic():
    """MAF per locus recomputed from allele counts."""
    G = [[0, 2, 1], [1, 2, 1], [2, 1, 0], [0, 2, 1], [1, 0, 2]]
    r = maf_calculation(G)
    n = len(G)
    for j in range(3):
        p = sum(row[j] for row in G) / (2 * n)
        assert r["p"][j] == pytest.approx(p, rel=1e-15)
        assert r["maf"][j] == pytest.approx(min(p, 1 - p), rel=1e-15)
    assert r["estimate"] == pytest.approx(sum(r["maf"]) / 3, rel=1e-15)
    assert r["n_loci"] == 3 and r["n"] == 5


def test_mafcl_missing_and_centred_coding():
    G = [[-1, 1], [0, None], [1, 1], [-1, 0]]
    r = maf_calculation(G, coding="-101")
    assert r["p"][0] == pytest.approx((0 + 1 + 2 + 0) / 8, rel=1e-15)
    assert r["p"][1] == pytest.approx((2 + 2 + 1) / 6, rel=1e-15)
    assert r["n_genotyped"] == [4, 3]
    assert r["maf"][1] == pytest.approx(1 - 5 / 6, rel=1e-12)


def test_mafcl_edge():
    with pytest.raises(ValueError, match="not a valid"):
        maf_calculation([[0, 3]])
    assert maf_calculation([2, 2, 2])["maf"] == [0.0]
