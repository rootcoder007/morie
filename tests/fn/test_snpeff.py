"""Tests for snpeff (Cingolani et al. 2012)."""

from morie.fn.snpeff import (annotate_variant, codon_table, snpeff,
                             translate)

CDS = "ATGGAAGTTTAA"          # M E V *


def test_the_standard_genetic_code():
    tab = codon_table()
    assert len(tab) == 64
    assert tab["ATG"] == "M"
    assert tab["TAA"] == tab["TAG"] == tab["TGA"] == "*"
    assert len(set(v for v in tab.values() if v != "*")) == 20
    for codon, aa in (("TTT", "F"), ("TGG", "W"), ("ATT", "I"),
                      ("GGG", "G"), ("CAT", "H"), ("AGA", "R")):
        assert tab[codon] == aa


def test_translation():
    assert translate(CDS) == "MEV*"
    assert translate(CDS, to_stop=True) == "MEV"
    assert translate("AUGGAA") == "ME"          # U is read as T


def test_every_coding_class():
    for pos, ref, alt, eff, imp, hgvs in (
            (3, "G", "T", "stop_gained", "HIGH", "p.E2*"),
            (4, "A", "T", "missense_variant", "MODERATE", "p.E2V"),
            (5, "A", "G", "synonymous_variant", "LOW", "p.E2="),
            (2, "G", "A", "start_lost", "HIGH", "p.M1I"),
            (9, "T", "A", "stop_lost", "HIGH", "p.*4K")):
        a = annotate_variant(CDS, pos, ref, alt)
        assert a["effect"] == eff
        assert a["impact"] == imp
        assert a["hgvs_p"] == hgvs


def test_the_codon_change_is_reported():
    a = annotate_variant(CDS, 4, "A", "T")
    assert a["ref_codon"] == "GAA" and a["alt_codon"] == "GTA"
    assert a["ref_aa"] == "E" and a["alt_aa"] == "V"
    assert a["codon_index"] == 1


def test_indels_are_classified_by_length_modulo_three():
    for pos, ref, alt, eff in ((4, "A", "AG", "frameshift_variant"),
                               (4, "AA", "A", "frameshift_variant"),
                               (3, "G", "GAAA", "inframe_insertion"),
                               (3, "GAAG", "G", "inframe_deletion")):
        assert annotate_variant(CDS, pos, ref, alt)["effect"] == eff
    assert annotate_variant(CDS, 4, "A", "AG")["impact"] == "HIGH"
    assert annotate_variant(CDS, 3, "G", "GAAA")["impact"] == "MODERATE"


def test_position_decides_the_non_coding_classes():
    big = "A" * 100 + CDS + "A" * 100
    up = annotate_variant(big, 50, "A", "G", cds_start=100,
                          transcript_len=12)
    down = annotate_variant(big, 150, "A", "G", cds_start=100,
                            transcript_len=12)
    far = annotate_variant(big, 50, "A", "G", cds_start=100,
                           transcript_len=12, upstream=10)
    assert up["effect"] == "upstream_gene_variant"
    assert up["impact"] == "MODIFIER"
    assert down["effect"] == "downstream_gene_variant"
    assert far["effect"] == "intergenic_variant"


def test_the_batch_call_tallies():
    res = snpeff(CDS, [(3, "G", "T"), (5, "A", "G"), (4, "A", "T")])
    assert res["n_variants"] == 3
    assert res["effect_counts"] == {"stop_gained": 1,
                                    "synonymous_variant": 1,
                                    "missense_variant": 1}
    assert res["impact_counts"] == {"HIGH": 1, "LOW": 1, "MODERATE": 1}
    assert res["protein"] == "MEV*"
    assert res["annotations"][0]["hgvs_c"] == "c.4G>T"
    assert res["annotations"][0]["pos"] == 3


def test_a_wrong_reference_allele_is_refused():
    try:
        annotate_variant(CDS, 4, "C", "T")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_validation():
    for call in (lambda: annotate_variant("", 0, "A", "T"),
                 lambda: annotate_variant(CDS, 99, "A", "T"),
                 lambda: annotate_variant(CDS, 4, "", "T"),
                 lambda: translate("ATGXYZ"),
                 lambda: snpeff(CDS, [(1, "T")])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
