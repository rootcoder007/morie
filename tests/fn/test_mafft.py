"""Tests for mafft (Katoh et al. 2002)."""

from morie.fn.mafft import (GRANTHAM_POLARITY, GRANTHAM_VOLUME, jtt_matrix,
                            arrange_segments, correlation,
                            find_homologous_segments, group_align,
                            guide_tree, iterative_refine, mafft_alignment,
                            mafftalignment, normalized_similarity_matrix,
                            residue_vectors, sixtuple_distance, wsp_score)

AA = "ARNDCQEGHILKMFPSTWYV"
SC = normalized_similarity_matrix(s_a=0.06)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _rand(r, n):
    return "".join(AA[int(r() * 20)] for _ in range(n))


def test_the_grantham_tables():
    assert sorted(GRANTHAM_POLARITY) == sorted(AA)
    assert sorted(GRANTHAM_VOLUME) == sorted(AA)
    assert GRANTHAM_POLARITY["A"] == 8.1 and GRANTHAM_VOLUME["A"] == 31.0
    assert GRANTHAM_VOLUME["G"] == 3.0 and GRANTHAM_VOLUME["W"] == 170.0
    assert GRANTHAM_POLARITY["D"] == 13.0 and GRANTHAM_POLARITY["L"] == 4.9


def test_the_jtt_200_default_matrix():
    from morie.fn.mafft import _JTT_COUNTS, _JTT_FREQ
    assert len(_JTT_COUNTS) == 190 and len(_JTT_FREQ) == 20
    assert abs(sum(_JTT_FREQ) - 1.0) < 1e-9
    j = jtt_matrix(200)
    P, f = j["P"], j["freqs"]
    order = AA
    assert max(abs(sum(row) - 1.0) for row in P) < 1e-9
    assert max(abs(f[order[i]] * P[i][k] - f[order[k]] * P[k][i])
               for i in range(20) for k in range(20)) < 1e-12
    assert abs(j["rate"] - 0.01) < 1e-12
    M = j["matrix"]
    assert max(abs(M[(a, b)] - M[(b, a)]) for a in AA for b in AA) < 1e-9
    assert all(M[(a, a)] > max(M[(a, b)] for b in AA if b != a)
               for a in AA)
    assert max(AA, key=lambda a: M[(a, a)]) == "W"


def test_the_all_positive_s_a_matches_the_printed_value():
    ap = normalized_similarity_matrix(mode="all_positive")
    assert abs(ap["s_a"] - 0.82) < 0.005


def test_the_default_scoring_uses_jtt_frequencies():
    assert abs(SC["freqs"]["W"] - 0.014) < 1e-9
    assert abs(SC["freqs"]["L"] - 0.091) < 1e-9


def test_the_fft_reproduces_equation_2():
    r = _lcg(20020714)
    a, b = _rand(r, 50), _rand(r, 60)
    lf, cf = correlation([a], [b], method="fft")
    ld, cd = correlation([a], [b], method="direct")
    assert lf == ld
    assert max(abs(x - y) for x, y in zip(cf, cd)) < 1e-9


def test_the_top_peak_sits_at_the_planted_lag():
    pad = 17
    s = _rand(_lcg(5), 80)
    t = _rand(_lcg(9), pad) + s
    lags, c = correlation([s], [t])
    assert lags[max(range(len(c)), key=lambda i: c[i])] == pad


def test_nucleotides_use_four_components():
    comps = residue_vectors(["ACGT"], seq_type="nt")
    assert len(comps) == 4
    lags, c = correlation(["ACGTACGTTTGA"], ["ACGTACGTTTGA"], seq_type="nt")
    assert lags[max(range(len(c)), key=lambda i: c[i])] == 0


def test_equation_7_pins_random_and_identical_scores():
    M, alpha, f = SC["matrix"], SC["alphabet"], SC["freqs"]
    rand = sum(f[a] * f[b] * M[(a, b)] for a in alpha for b in alpha)
    ident = sum(f[a] * M[(a, a)] for a in alpha)
    assert abs(rand - 0.06) < 1e-12
    assert abs(ident - 1.06) < 1e-12
    assert min(M.values()) < 0 < max(M.values())


def test_the_all_positive_control():
    ap = normalized_similarity_matrix(mode="all_positive")
    assert min(ap["matrix"].values()) >= -1e-12
    assert abs(min(ap["matrix"].values())) < 1e-12


def test_the_gap_penalty_vanishes_on_an_existing_gap():
    from morie.fn.mafft import _gap_profiles
    s_op = 2.4
    gs, ge = _gap_profiles(["AC--GT", "AC--GT"], [0.5, 0.5])
    assert abs(s_op * (1.0 - (gs[1] + ge[4]) / 2.0)) < 1e-12
    gs, ge = _gap_profiles(["AC--GT", "ACTTGT"], [0.5, 0.5])
    assert abs(s_op * (1.0 - (gs[1] + ge[4]) / 2.0) - s_op * 0.5) < 1e-12
    gs, ge = _gap_profiles(["ACTTGT", "ACTTGT"], [0.5, 0.5])
    assert abs(s_op * (1.0 - (gs[1] + ge[4]) / 2.0) - s_op) < 1e-12


def test_segments_cover_a_planted_block():
    block = "".join(AA[(i * 7 + 3) % 20] for i in range(60))
    a = _rand(_lcg(101), 25) + block + _rand(_lcg(202), 25)
    b = _rand(_lcg(303), 40) + block + _rand(_lcg(404), 10)
    segs = find_homologous_segments([a], [b], SC, window=30, threshold=0.7)
    assert any(s[4] == 15 and s[0] <= 25 and s[0] + s[2] >= 85
               for s in segs)
    u1, u2 = _rand(_lcg(7), 90), _rand(_lcg(8), 90)
    assert find_homologous_segments([u1], [u2], SC, window=30) == []
    chain = arrange_segments(segs)
    assert all(chain[i][0] + chain[i][2] <= chain[i + 1][0]
               for i in range(len(chain) - 1))


def test_max_len_cuts_long_segments():
    long_block = "".join(AA[(i * 3) % 20] for i in range(400))
    segs = find_homologous_segments([long_block], [long_block], SC,
                                    window=30, max_len=150)
    assert segs and max(s[2] for s in segs) <= 150


def test_a_single_deletion_is_a_single_gap():
    a, b = group_align(["ACDEFGHIK"], ["ACDEGHIK"], SC)
    assert len(a[0]) == len(b[0])
    assert a[0] == "ACDEFGHIK"
    assert b[0].count("-") == 1
    assert b[0].replace("-", "") == "ACDEGHIK"


def test_anchors_do_not_change_a_clean_answer():
    plain = group_align(["ACDEFGHIK"], ["ACDEFGHIK"], SC)
    anch = group_align(["ACDEFGHIK"], ["ACDEFGHIK"], SC, anchors=[(4, 4)])
    assert plain == anch


def test_the_guide_tree_recovers_planted_families():
    base = _rand(_lcg(31337), 70)
    other = _rand(_lcg(555), 70)

    def mut(s, k, seed):
        rr = _lcg(seed)
        t = list(s)
        for _ in range(k):
            t[int(rr() * len(t))] = AA[int(rr() * 20)]
        return "".join(t)

    fam = [mut(base, 3, 1), mut(base, 4, 2), mut(other, 3, 3),
           mut(other, 4, 4)]
    D = sixtuple_distance(fam)
    assert max(D[0][1], D[2][3]) < min(D[0][2], D[0][3], D[1][2], D[1][3])
    assert all(D[i][i] == 0.0 for i in range(4))
    tree = guide_tree(D)
    assert len(tree) == 3
    assert set(tree[0][3]) in ({0, 1}, {2, 3})


def test_every_named_method_aligns():
    seqs = ["ACDEFGHIKLMNPQRSTVWY",
            "ACDEFGHIKLMNPQRSTVWY".replace("F", ""),
            "ACDEFGHIKLMNPQRSTVWY".replace("MN", "M"),
            "ACDEFGHIKLMNPQRSTVWY"]
    out = {}
    for meth in ("FFT-NS-1", "FFT-NS-2", "FFT-NS-i", "NW-NS-2"):
        res = mafft_alignment(seqs, method=meth)
        aln = res["alignment"]
        assert len(aln) == 4
        assert len(set(len(s) for s in aln)) == 1
        assert all(a.replace("-", "") == s for a, s in zip(aln, seqs))
        out[meth] = res
    assert out["FFT-NS-i"]["score"] >= out["FFT-NS-2"]["score"] - 1e-9
    assert out["FFT-NS-1"]["refine_rounds"] == 0


def test_identical_and_nucleotide_input():
    assert mafft_alignment(["ACDEFGHIK"] * 3)["alignment"] == \
        ["ACDEFGHIK"] * 3
    nt = mafft_alignment(["ACGTACGTAC", "ACGTCGTAC", "ACGTACGTAC"])
    assert nt["seq_type"] == "nt"
    assert len(set(len(s) for s in nt["alignment"])) == 1


def test_the_wsp_score_reacts_to_damage():
    good = mafft_alignment(["ACDEFGHIK", "ACDEFGHIK", "ACDEGHIK"],
                           method="FFT-NS-2")["alignment"]
    bad = [good[0]] + ["---" + s[:-3] for s in good[1:]]
    assert wsp_score(bad, SC) < wsp_score(good, SC)


def test_iterative_refine_never_worsens():
    aln = mafft_alignment(["ACDEFGHIK", "ACDEFGHIK", "ACDEGHIK"],
                          method="FFT-NS-2")["alignment"]
    before = wsp_score(aln, SC)
    out, score, rounds = iterative_refine(aln, SC)
    assert score >= before - 1e-9
    assert rounds >= 1
    assert len(set(len(s) for s in out)) == 1


def test_validation():
    for call in (lambda: mafft_alignment(["ACDEF"]),
                 lambda: mafft_alignment(["ACDEF", "ACDEF"],
                                         method="T-COFFEE"),
                 lambda: mafft_alignment(["ACDEF", "ACDEF"], matrix="raw"),
                 lambda: mafft_alignment(["ACDEF", ""]),
                 lambda: mafft_alignment(["ACDEF", "ACDEF"],
                                         seq_type="rna"),
                 lambda: correlation(["ACDEF"], ["ACDEF"], method="dft"),
                 lambda: residue_vectors([]),
                 lambda: residue_vectors(["ACD", "AC"]),
                 lambda: residue_vectors(["ACD"], weights=[1.0, 1.0]),
                 lambda: normalized_similarity_matrix(mode="positive"),
                 lambda: normalized_similarity_matrix(default="blosum62"),
                 lambda: jtt_matrix(pam=0),
                 lambda: normalized_similarity_matrix(
                     dict(((a, b), 1.0) for a in AA for b in AA)),
                 lambda: normalized_similarity_matrix({("A", "A"): 1.0}),
                 lambda: group_align([], ["AC"], SC),
                 lambda: group_align(["AC", "ACG"], ["AC"], SC),
                 lambda: group_align(["AC"], ["AC"], SC,
                                     weights1=[1.0, 1.0]),
                 lambda: group_align(["ACGT"], ["ACGT"], SC,
                                     anchors=[(1, 3), (3, 1)]),
                 lambda: group_align(["ACGT"], ["ACGT"], SC,
                                     anchors=[(9, 1)]),
                 lambda: find_homologous_segments(["AC"], ["AC"], SC,
                                                  window=0),
                 lambda: guide_tree([[0.0]]),
                 lambda: wsp_score(["ACD", "AC"], SC),
                 lambda: iterative_refine(["ACD", "ACD"], SC,
                                          max_iterate=0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert mafftalignment is mafft_alignment
