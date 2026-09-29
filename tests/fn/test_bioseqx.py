import pytest

from morie.fn.bioseqx import olc_assembly, protein_disorder

KD = {"A": 1.8, "R": -4.5, "D": -3.5, "E": -3.5, "G": -0.4, "K": -3.9, "L": 3.8, "S": -0.8, "V": 4.2, "I": 4.5}


def test_olc_exact_reads_reconstruct_genome():
    g = "ATGCGTACGTTAGCCGATCGATTGCAAGCTTGACCTAGGCATCGAT"
    reads = [g[s : s + 15] for s in range(0, len(g) - 14, 5)]
    if not g.endswith(reads[-1]):
        reads.append(g[-15:])
    r = olc_assembly(reads[::-1] + [g[3:9]], min_overlap=5)
    assert r.contigs == [g]
    assert r.n_reads_used == len(reads)


def test_olc_consensus_fixes_minority_error():
    g = "ACGTTGCAAGGCTTACCGAT"
    bad = g[3:8] + "T" + g[9:14]
    assert bad != g[3:14]
    reads = [g[0:11], bad, g[6:17], g[9:20]]
    r = olc_assembly(reads, min_overlap=6, max_error=0.2)
    assert r.contigs == [g]


def test_olc_doc_example():
    r = olc_assembly(["ATGGCGT", "GCGTGCA", "TGCAATG", "CAATGGA"])
    assert r.contigs == ["ATGGCGTGCAATGGA"]
    assert [L for _, _, L, _ in r.overlaps] == [5, 4, 4]


def test_foldindex_formula():
    s = "KKEEGALLVIDRSS"
    H = sum((KD[a] + 4.5) / 9 for a in s) / len(s)
    R = (s.count("K") + s.count("R") - s.count("D") - s.count("E")) / len(s)
    r = protein_disorder(s, window=5)
    assert r.fold_index == pytest.approx(2.785 * H - abs(R) - 1.151, abs=1e-12)
    w = s[0:5]
    Hw = sum((KD[a] + 4.5) / 9 for a in w) / 5
    Rw = (w.count("K") - w.count("E")) / 5
    assert r.profile[0] == pytest.approx(2.785 * Hw - abs(Rw) - 1.151, abs=1e-12)
    assert r.profile[2] == r.profile[0]


def test_disorder_regions_and_errors():
    r = protein_disorder("EEKKEESKDEKKSEE" + "LLIVAVLLIVAVLLV", window=7, min_region=3)
    assert r.regions and r.regions[0][0] == 1
    assert not r.disordered[-1]
    with pytest.raises(ValueError):
        protein_disorder("MKX")
    with pytest.raises(ValueError):
        protein_disorder("MKKL", window=4)
