"""Tests for sv_dl (Rausch et al. 2012, DELLY)."""

import math

from morie.fn.sv_dl import (classify_pair, deletion_type_reference,
                            gotoh_score_vectors, insert_size_stats,
                            kmer_diagonals, maximal_clique, optimal_split,
                            refine_breakpoint, split_read_consensus,
                            structural_variant, sv_delly)

RL, MED, SD = 60, 400, 25
BP1, BP2 = 1200, 1500
DELSIZE = BP2 - BP1


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


_rnd = _lcg(20120918)
REF = "".join("ACGT"[int(_rnd() * 4)] for _ in range(3000))
DONOR = REF[:BP1] + REF[BP2:]


def _simulate(n_frag=600, seed=7):
    r = _lcg(seed)
    pairs, splits = [], []
    for i in range(n_frag):
        L = int(MED + SD * _gauss(r))
        if L < 2 * RL + 10 or L > 700:
            continue
        s = int(r() * (len(DONOR) - L))
        d1a, d1b, d2a, d2b = s, s + RL, s + L - RL, s + L
        for a, b in ((d1a, d1b), (d2a, d2b)):
            if a < BP1 < b:
                splits.append(DONOR[a:b])
        if d1a < BP1 < d1b or d2a < BP1 < d2b:
            continue
        to_ref = (lambda d: d if d < BP1 else d + DELSIZE)
        pairs.append({"chrom1": "chr1", "pos1": to_ref(d1a), "strand1": "+",
                      "len1": RL, "chrom2": "chr1", "pos2": to_ref(d2a),
                      "strand2": "-", "len2": RL})
    return pairs, splits


PAIRS, SPLITS = _simulate()


def _mk(p1, s1, p2, s2, c2="chr1"):
    return {"chrom1": "chr1", "pos1": p1, "strand1": s1, "len1": RL,
            "chrom2": c2, "pos2": p2, "strand2": s2, "len2": RL}


def test_the_library_parameters_are_recovered():
    st = insert_size_stats(PAIRS)
    assert abs(st["median"] - MED) < 15
    assert abs(st["sd"] - SD) < 15
    assert st["orientation"] == ("+", "-")


def test_the_literal_sd_is_inflated_by_the_deletions_own_pairs():
    robust = insert_size_stats(PAIRS)["sd"]
    literal = insert_size_stats(PAIRS, spread="sd")["sd"]
    assert literal > 2 * robust


def test_every_signature_is_recognised():
    assert classify_pair(_mk(1000, "+", 1340, "-"), MED, SD) is None
    assert classify_pair(_mk(1000, "+", 1900, "-"), MED, SD) == ("DEL", "")
    assert classify_pair(_mk(1000, "-", 1340, "+"), MED, SD) == ("DUP", "")
    assert classify_pair(_mk(1000, "+", 1340, "+"), MED, SD) == ("INV", "left")
    assert classify_pair(_mk(1000, "-", 1340, "-"), MED, SD) == \
        ("INV", "right")
    tra = set()
    for s1 in ("+", "-"):
        for s2 in ("+", "-"):
            tra.add(classify_pair(_mk(1000, s1, 1340, s2, c2="chr2"),
                                  MED, SD)[1])
    assert tra == {"0", "1", "2", "3"}


def test_the_clique_is_a_clique_grown_from_the_lightest_edge():
    edges = [(0.5, 0, 1), (0.9, 0, 2), (0.7, 1, 2), (0.1, 2, 3)]
    cl = maximal_clique([0, 1, 2, 3], edges)
    adj = set()
    for _, i, j in edges:
        adj.add((i, j))
        adj.add((j, i))
    assert {2, 3} <= set(cl)
    assert all((a, b) in adj for a in cl for b in cl if a != b)
    assert maximal_clique([5, 6], []) == []


def test_a_deletion_is_called_and_bracketed():
    res = structural_variant(PAIRS)
    dels = [c for c in res["calls"] if c["type"] == "DEL"]
    assert len(dels) == 1
    d = dels[0]
    assert d["start"] <= BP1 and d["end"] >= BP2
    assert d["precise"] is False
    assert abs(d["size"] - DELSIZE) < 60


def test_a_quiet_genome_stays_quiet():
    r = _lcg(99)
    flat = []
    for _ in range(300):
        L = int(MED + SD * _gauss(r))
        if L < 2 * RL + 10 or L > 700:
            continue
        s = int(r() * (len(REF) - L))
        flat.append({"chrom1": "chr1", "pos1": s, "strand1": "+", "len1": RL,
                     "chrom2": "chr1", "pos2": s + L - RL, "strand2": "-",
                     "len2": RL})
    assert structural_variant(flat)["n_calls"] == 0


def test_a_junction_read_gives_two_diagonals_a_deletion_apart():
    region = REF[800:1900]
    diags = kmer_diagonals(SPLITS[0], region, k=7, k_min=3)
    assert diags is not None and len(diags) == 2
    assert diags[1][0] - diags[0][0] == DELSIZE
    assert kmer_diagonals(REF[300:360], REF[200:600], k=7) is None


def test_the_consensus_is_a_majority_vote():
    assert split_read_consensus(["ACGTAC", "ACGTAC", "ACGAAC"]) == \
        ("ACGTAC", 0)
    assert split_read_consensus(["ACGTT", "GTTGG"], starts=[10, 12]) == \
        ("ACGTTGG", 10)


def test_the_split_takes_argmax_f_plus_r():
    assert optimal_split([1.0, 5.0, 2.0, 0.0], [0.0, 1.0, 4.0, 9.0])[:2] == \
        (2, 4)
    f, f_at, r, r_at = gotoh_score_vectors("ACGTACGT", REF[100:200])
    assert len(f) == len(r) == len(f_at) == len(r_at) == 8


def test_split_reads_rebuild_the_sequenced_haplotype():
    res = structural_variant(PAIRS, reference=REF, split_reads=SPLITS)
    d = [c for c in res["calls"] if c["type"] == "DEL"][0]
    assert d["precise"] is True
    assert REF[:d["start"]] + REF[d["end"]:] == DONOR
    assert d["end"] - d["start"] == DELSIZE
    assert d["microinsertion"] == ""
    assert d["microhomology"] == 3


def test_refinement_declines_rather_than_invents():
    region = REF[800:1900]
    assert refine_breakpoint({"type": "DEL", "size": float(DELSIZE)}, region,
                             [REF[200:260], REF[400:460]]) is None
    assert refine_breakpoint({"type": "DEL", "size": 900.0}, region,
                             SPLITS) is None


def test_the_reference_rewriting_of_figure_4():
    s = "AAAACCCC"
    assert deletion_type_reference(s, "DEL") == s
    assert deletion_type_reference(s, "DUP") == "CCCCAAAA"
    assert deletion_type_reference(s, "INV") == "AAAAGGGG"
    assert deletion_type_reference(s, "TRA") == "GGGGAAAA"


def test_validation():
    for call in (lambda: structural_variant([]),
                 lambda: structural_variant(PAIRS, n_sd=-1.0),
                 lambda: structural_variant(PAIRS, min_support=0),
                 lambda: structural_variant(PAIRS, spread="iqr"),
                 lambda: insert_size_stats([_mk(10, "+", 20, "-",
                                                c2="chr2")]),
                 lambda: classify_pair({"chrom1": "a"}, 1, 1),
                 lambda: classify_pair(_mk(10, "x", 20, "-"), MED, SD),
                 lambda: deletion_type_reference("ACGT", "INS"),
                 lambda: kmer_diagonals("ACGT", "ACGT", k=0),
                 lambda: split_read_consensus([]),
                 lambda: split_read_consensus(["AC"], starts=[1, 2]),
                 lambda: gotoh_score_vectors("", "ACGT"),
                 lambda: optimal_split([1.0, 2.0], [1.0])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert sv_delly is structural_variant
