"""Tests for vepan (McLaren et al. 2016, Ensembl VEP)."""

from morie.fn.snpeff import translate
from morie.fn.vepan import (CONSEQUENCE_IMPACT, CONSEQUENCE_RANK,
                            CONSEQUENCE_SO, annotate, consequence_impact,
                            consequence_rank, most_severe_consequence, pick,
                            transcript_sequence, vep_annotation,
                            vepannotation)

PROT = "MKWAYFGHLSTVNDQEREPICM" + "KWAYFGHLSTVNDQ" + "*"
CODON = {"M": "ATG", "K": "AAA", "W": "TGG", "A": "GCA", "Y": "TAC",
         "F": "TTT", "G": "GGA", "H": "CAC", "L": "CTG", "S": "AGC",
         "T": "ACA", "V": "GTG", "N": "AAC", "D": "GAC", "Q": "CAG",
         "E": "GAA", "R": "CGG", "P": "CCA", "I": "ATC", "C": "TGC",
         "*": "TAA"}
CDS = "".join(CODON[a] for a in PROT)
EXONS = [(101, 130), (201, 260), (331, 380)]

_g = ["ACGT"[(k * 7 + 3) % 4] for k in range(500)]
_coding = (list(range(110, 131)) + list(range(201, 261)) +
           list(range(331, 361)))
for _b, _p in zip(CDS, _coding):
    _g[_p - 1] = _b
for _p, _b in ((131, "G"), (132, "T"), (199, "A"), (200, "G"),
               (261, "G"), (262, "T"), (329, "A"), (330, "G")):
    _g[_p - 1] = _b
GENOME = "".join(_g)

TR = {"id": "T1", "gene": "G1", "chrom": "chr1", "strand": "+",
      "exons": EXONS, "cds_start": 110, "cds_end": 360,
      "biotype": "protein_coding", "canonical": True}
ALT_TR = {"id": "T2", "gene": "G1", "chrom": "chr1", "strand": "+",
          "exons": [(101, 380)], "cds_start": 109, "cds_end": 360,
          "biotype": "protein_coding", "canonical": False}


def _call(pos, ref, alt, tr=TR):
    return annotate({"chrom": "chr1", "pos": pos, "ref": ref, "alt": alt},
                    [tr], GENOME)[0]


def test_the_published_severity_table():
    for rank, term, impact, acc in (
            (2, "splice_acceptor_variant", "HIGH", "SO:0001574"),
            (4, "stop_gained", "HIGH", "SO:0001587"),
            (13, "missense_variant", "MODERATE", "SO:0001583"),
            (22, "synonymous_variant", "LOW", "SO:0001819"),
            (40, "intergenic_variant", "MODIFIER", "SO:0001628")):
        assert CONSEQUENCE_RANK[term] == rank
        assert CONSEQUENCE_IMPACT[term] == impact
        assert CONSEQUENCE_SO[term] == acc
    assert sorted(CONSEQUENCE_RANK.values()) == list(range(1, 42))


def test_most_severe_and_impact():
    assert most_severe_consequence(
        ["intron_variant", "stop_gained", "missense_variant"]) == \
        "stop_gained"
    assert most_severe_consequence(["nope", "synonymous_variant"]) == \
        "synonymous_variant"
    assert consequence_rank("nope") > consequence_rank("intergenic_variant")
    assert consequence_impact("nope") == "MODIFIER"


def test_the_spliced_cds_translates_to_the_protein():
    seq, gpos = transcript_sequence(TR, GENOME)
    cds = "".join(seq[k] for k, g in enumerate(gpos) if 110 <= g <= 360)
    assert translate(cds) == translate(CDS)
    assert translate(cds)[0] == "M"


def test_the_coding_consequences():
    assert _call(113, "A", "T")["most_severe"] == "stop_gained"
    assert _call(113, "A", "T")["impact"] == "HIGH"
    r = _call(115, "A", "C")
    assert r["most_severe"] == "missense_variant"
    assert (r["ref_aa"], r["alt_aa"]) == ("K", "N")
    assert r["protein_position"] == 2
    assert r["hgvs_p"] == "p.Lys2Asn"
    assert r["hgvs_c"] == "c.6A>C"
    assert _call(121, "A", "G")["most_severe"] == "synonymous_variant"
    assert _call(121, "A", "G")["hgvs_p"] == "p.Ala4="
    assert _call(110, "A", "G")["most_severe"] == "start_lost"
    assert _call(358, "T", "G")["most_severe"] == "stop_lost"


def test_indels_in_and_out_of_frame():
    assert _call(120, "C", "CA")["most_severe"] == "frameshift_variant"
    assert _call(120, "C", "CAAA")["most_severe"] == "inframe_insertion"
    assert _call(120, GENOME[119:122], GENOME[119])["most_severe"] == \
        "frameshift_variant"
    assert _call(120, GENOME[119:123], GENOME[119])["most_severe"] == \
        "inframe_deletion"
    assert _call(120, "C", "CA")["hgvs_p"].endswith("fs")


def test_the_splice_terms():
    assert _call(131, GENOME[130], "C")["most_severe"] == \
        "splice_donor_variant"
    assert _call(200, GENOME[199], "C")["most_severe"] == \
        "splice_acceptor_variant"
    assert _call(135, GENOME[134], "C")["most_severe"] == \
        "splice_donor_5th_base_variant"
    assert _call(165, GENOME[164], "C")["most_severe"] == "intron_variant"
    assert "splice_polypyrimidine_tract_variant" in \
        _call(190, GENOME[189], "C")["consequences"]
    assert "splice_region_variant" in \
        _call(129, GENOME[128], "C")["consequences"]


def test_utrs_and_flanks():
    assert _call(105, GENOME[104], "C")["most_severe"] == \
        "5_prime_UTR_variant"
    assert _call(370, GENOME[369], "C")["most_severe"] == \
        "3_prime_UTR_variant"
    assert _call(95, GENOME[94], "C")["most_severe"] == \
        "upstream_gene_variant"
    assert _call(390, GENOME[389], "C")["most_severe"] == \
        "downstream_gene_variant"
    near = annotate({"chrom": "chr1", "pos": 90, "ref": "A", "alt": "C"},
                    [TR], GENOME, upstream=11, downstream=11)
    assert near[0]["most_severe"] == "upstream_gene_variant"
    out = annotate({"chrom": "chr1", "pos": 89, "ref": "A", "alt": "C"},
                   [TR], GENOME, upstream=11, downstream=11)
    assert out[0]["most_severe"] == "intergenic_variant"


def test_the_reverse_strand():
    rev = dict(TR, id="T1R", strand="-")
    fwd_seq, _ = transcript_sequence(TR, GENOME)
    rev_seq, rev_pos = transcript_sequence(rev, GENOME)
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    assert rev_seq == "".join(comp[b] for b in reversed(fwd_seq))
    assert rev_pos[0] > rev_pos[-1]
    up = annotate({"chrom": "chr1", "pos": 95, "ref": GENOME[94],
                   "alt": "C"}, [rev], GENOME)
    assert up[0]["most_severe"] == "downstream_gene_variant"


def test_pick_puts_the_canonical_transcript_first():
    recs = annotate({"chrom": "chr1", "pos": 165, "ref": GENOME[164],
                     "alt": "C"}, [TR, ALT_TR], GENOME)
    by = dict((r["transcript"], r) for r in recs)
    assert consequence_rank(by["T2"]["most_severe"]) < \
        consequence_rank(by["T1"]["most_severe"])
    assert pick(recs)[0]["transcript"] == "T1"
    plain = [dict(r, canonical=False) for r in recs]
    assert pick(plain)[0]["transcript"] == "T2"


def test_per_gene_and_non_coding():
    nc = {"id": "T3", "gene": "G2", "chrom": "chr1", "strand": "+",
          "exons": [(101, 380)], "biotype": "lncRNA"}
    recs = annotate({"chrom": "chr1", "pos": 165, "ref": GENOME[164],
                     "alt": "C"}, [ALT_TR, nc], GENOME)
    assert pick(recs)[0]["transcript"] == "T2"
    per = pick(recs, per_gene=True)
    assert sorted(r["gene"] for r in per) == ["G1", "G2"]
    t3 = [r for r in recs if r["transcript"] == "T3"][0]
    assert "non_coding_transcript_variant" in t3["consequences"]


def test_an_insertion_in_a_repeat_is_shifted_three_prime():
    rep = list(GENOME)
    for k in range(150, 162):
        rep[k] = "A"
    rep = "".join(rep)
    tr = {"id": "TR2", "gene": "GR", "chrom": "chr1", "strand": "+",
          "exons": [(101, 380)], "cds_start": 109, "cds_end": 360,
          "biotype": "protein_coding", "canonical": True}
    a = annotate({"chrom": "chr1", "pos": 151, "ref": "A", "alt": "AA"},
                 [tr], rep)[0]
    b = annotate({"chrom": "chr1", "pos": 151, "ref": "A", "alt": "AA"},
                 [tr], GENOME)[0]
    assert a["hgvs_c"] != b["hgvs_c"]


def test_the_driver_and_its_modes():
    vs = [{"chrom": "chr1", "pos": 113, "ref": "A", "alt": "T"},
          {"chrom": "chr1", "pos": 165, "ref": GENOME[164], "alt": "C"}]
    res = vep_annotation(vs, [TR, ALT_TR], GENOME)
    assert res["n_variants"] == 2 and res["n_annotations"] == 4
    want = {}
    for r in res["annotations"]:
        for t in r["consequences"]:
            want[t] = want.get(t, 0) + 1
    assert res["consequence_counts"] == want
    assert vep_annotation(vs, [TR, ALT_TR], GENOME,
                          mode="pick")["n_annotations"] == 2
    assert vep_annotation(vs, [TR, ALT_TR], GENOME,
                          mode="per_gene")["n_annotations"] == 2
    off = vep_annotation([{"chrom": "chr1", "pos": 5, "ref": GENOME[4],
                           "alt": "C"}], [TR], GENOME, upstream=11,
                         downstream=11, no_intergenic=True)
    assert off["n_annotations"] == 0


def test_validation():
    v = {"pos": 1, "ref": "A", "alt": "C"}
    for call in (lambda: vep_annotation([], [TR], GENOME),
                 lambda: vep_annotation([v], [TR], GENOME, mode="worst"),
                 lambda: annotate({"pos": 0, "ref": "A", "alt": "C"}, [TR],
                                  GENOME),
                 lambda: annotate({"pos": 1, "ref": "AC", "alt": "GT"},
                                  [TR], GENOME),
                 lambda: annotate({"pos": 1, "ref": "A", "alt": "Z"}, [TR],
                                  GENOME),
                 lambda: annotate(v, [TR], GENOME, upstream=-1),
                 lambda: annotate(v, [{"exons": [], "strand": "+"}],
                                  GENOME),
                 lambda: annotate(v, [{"exons": [(10, 5)], "strand": "+"}],
                                  GENOME),
                 lambda: annotate(v, [{"exons": [(1, 20), (10, 30)],
                                       "strand": "+"}], GENOME),
                 lambda: annotate(v, [{"exons": [(1, 20)], "strand": "?"}],
                                  GENOME),
                 lambda: annotate(v, [{"exons": [(1, 20)],
                                       "biotype": "protein_coding"}],
                                  GENOME),
                 lambda: most_severe_consequence([]),
                 lambda: pick([])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert vepannotation is vep_annotation
