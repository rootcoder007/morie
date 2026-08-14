"""Variant calling by pileup image classification."""
import importlib

import pytest

V = importlib.import_module("morie.fn.varcal")
REF = "ACGTACGTACGTACGTACGTACGTACGTACGT"


def mkreads(vaf=0.5, n=20, bq=35, site=10):
    reads = []
    for k in range(n):
        seq = list(REF[5:25])
        if k < int(round(vaf * n)):
            seq[site - 5] = "T" if REF[site] != "T" else "A"
        reads.append({"pos": 5, "seq": "".join(seq), "bq": [bq] * 20,
                      "mq": 60, "reverse": k % 3 == 0})
    return reads


READS = mkreads()


def test_pileup_column_depth_and_reference():
    c = V.pileup_column(READS, 10, REF)
    assert c["depth"] == 20
    assert c["reference"] == REF[10]


def test_permissive_thresholds_keep_low_fraction_variants():
    low = V.find_candidates(mkreads(vaf=0.10), REF,
                            min_alt_fraction=0.05)
    strict = V.find_candidates(mkreads(vaf=0.10), REF,
                               min_alt_fraction=0.30)
    assert len(low) == 1
    assert strict == []


def test_candidates_carry_their_evidence():
    c = V.find_candidates(mkreads(vaf=0.10), REF,
                          min_alt_fraction=0.05)[0]
    assert c["alt_count"] == 2
    assert c["depth"] == 20
    assert c["alt_fraction"] == pytest.approx(0.1)


def test_low_quality_bases_do_not_count():
    assert V.find_candidates(mkreads(vaf=0.10, bq=5), REF,
                             min_bq=10) == []


@pytest.mark.parametrize("kw", [{"min_alt_fraction": 1.5},
                                {"min_alt_fraction": -0.1},
                                {"min_alt_count": 0}])
def test_invalid_candidate_thresholds_are_refused(kw):
    with pytest.raises(ValueError):
        V.find_candidates(READS, REF, **kw)


def test_the_image_is_centred_on_the_candidate():
    img = V.encode_pileup(READS, REF, {"position": 10}, width=21)
    assert img["centre"] == 10
    assert img["width"] == 21
    assert len(img["reference_row"]) == 21
    assert all(len(r) == 21 for r in img["read_rows"])


def test_every_cell_has_four_channels():
    img = V.encode_pileup(READS, REF, {"position": 10})
    assert all(len(c) == 4 for r in img["read_rows"] for c in r)


def test_the_matches_reference_channel_marks_the_alt_reads():
    img = V.encode_pileup(READS, REF, {"position": 10})
    mid = img["width"] // 2
    assert sum(1 for r in img["read_rows"] if r[mid][3] < 0.5) == 10


def test_an_even_width_is_refused():
    with pytest.raises(ValueError):
        V.encode_pileup(READS, REF, {"position": 10}, width=20)


def test_an_unknown_channel_set_is_refused():
    with pytest.raises(ValueError):
        V.encode_pileup(READS, REF, {"position": 10},
                        channels="rgb_secret")


@pytest.mark.parametrize("vaf,want", [(0.02, "hom_ref"),
                                      (0.50, "het"),
                                      (0.98, "hom_alt")])
def test_the_genotype_follows_the_allele_fraction(vaf, want):
    img = V.encode_pileup(mkreads(vaf=vaf), REF, {"position": 10})
    assert V.genotype_posterior(img)["call"] == want


def test_the_posterior_is_a_probability_vector():
    img = V.encode_pileup(READS, REF, {"position": 10})
    p = V.genotype_posterior(img)["posterior"]
    assert set(p) == set(V.GENOTYPES)
    assert sum(p.values()) == pytest.approx(1.0)


def test_the_fallback_declares_itself():
    img = V.encode_pileup(READS, REF, {"position": 10})
    assert "NOT a trained" in V.genotype_posterior(img)["source"]


def test_a_supplied_scorer_is_used():
    img = V.encode_pileup(READS, REF, {"position": 10})
    g = V.genotype_posterior(img, scorer=lambda im: (0.0, 0.0, 1.0))
    assert g["call"] == "hom_alt"
    assert g["source"] == "supplied scorer"


@pytest.mark.parametrize("kw", [
    {"scorer": lambda im: (1.0, -1.0, 0.0)},
    {"scorer": lambda im: (1.0, 0.0)},
    {"prior": (0.5, 0.4, 0.2)},
])
def test_invalid_scorers_and_priors_are_refused(kw):
    img = V.encode_pileup(READS, REF, {"position": 10})
    with pytest.raises(ValueError):
        V.genotype_posterior(img, **kw)


def test_evaluate_reports_both_stages():
    truth = [{"position": 10, "alternate": "T"}]
    cands = truth + [{"position": p, "alternate": "A"}
                     for p in range(11, 23)]
    ev = V.evaluate(truth, truth, cands)
    assert ev["ppv"] == 1.0
    assert ev["candidate_ppv"] == pytest.approx(1.0 / 13.0)
    assert ev["sensitivity_loss"] == 0.0


def test_the_ion_torrent_arithmetic():
    truth = [{"position": i, "alternate": "T"} for i in range(1000)]
    cands = truth + [{"position": 10000 + i, "alternate": "A"}
                     for i in range(11346)]
    called = [{"position": i, "alternate": "T"} for i in range(997)] \
        + [{"position": 50000 + i, "alternate": "A"} for i in range(3)]
    ev = V.evaluate(called, truth, cands)
    assert ev["candidate_ppv"] == pytest.approx(0.081, abs=0.001)
    assert ev["ppv"] == pytest.approx(0.997, abs=0.001)
    assert ev["sensitivity_loss"] == pytest.approx(0.003, abs=0.001)


def test_end_to_end_call():
    r = V.call_variants(READS, REF)
    assert r["n_candidates"] == 1
    assert r["n_called"] == 1
    assert r["calls"][0]["call"] == "het"
