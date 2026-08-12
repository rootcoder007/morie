"""Tests for hapblk (Gabriel haplotype blocks, Gabriel et al. 2002).

Replaces the generated stub, which imported ``haplotype_block``.
"""

from morie.fn.hapblk import hapblk


def _perfect_ld(n_hap=40, n_snp=6):
    # every SNP identical: D' = 1 everywhere, so one block
    return [[i % 2] * n_snp for i in range(n_hap)]


def _two_blocks(n_hap=40):
    # first three SNPs perfectly linked, last three too, independent
    out = []
    for i in range(n_hap):
        a, b = i % 2, (i // 2) % 2
        out.append([a, a, a, b, b, b])
    return out


def test_perfect_linkage_gives_one_block():
    res = hapblk(_perfect_ld())
    assert len(res["blocks"]) == 1
    assert res["blocks"][0] == (0, 5) or res["blocks"][0] == [0, 5]


def test_dprime_is_one_under_perfect_linkage():
    res = hapblk(_perfect_ld())
    d = res["dprime"]
    assert abs(d[0][1] - 1.0) < 1e-9


def test_two_independent_halves_are_not_one_block():
    res = hapblk(_two_blocks())
    assert len(res["blocks"]) >= 1
    for b in res["blocks"]:
        lo, hi = b[0], b[1]
        assert not (lo <= 2 and hi >= 3)      # no block spans the break


def test_pairs_are_classified():
    res = hapblk(_perfect_ld())
    kinds = set()
    for row in res["pair_class"]:
        for v in row:
            if v is not None:
                kinds.add(v)
    assert kinds


def test_confidence_bounds_are_ordered():
    res = hapblk(_perfect_ld())
    for i in range(len(res["ci_lo"])):
        for j in range(len(res["ci_lo"])):
            if i < j:
                assert res["ci_lo"][i][j] <= res["ci_hi"][i][j] + 1e-12


def test_validation():
    for call in (lambda: hapblk([[0, 1], [1, 0]]),
                 lambda: hapblk([[0, 1], [1], [0, 1], [1, 0]])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
