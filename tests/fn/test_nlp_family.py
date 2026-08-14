# -*- coding: utf-8 -*-
"""Tests for the NLP batch: lda, plsa, sacrb, sentpc, sbert.

Each test pins a published value, a closed form, an exact identity, or
a brute-force cross-check -- never a self-comparison.
"""
import importlib
import math

import pytest

lda = importlib.import_module("morie.fn.lda")
plsa = importlib.import_module("morie.fn.plsa")
sacrb = importlib.import_module("morie.fn.sacrb")
sentpc = importlib.import_module("morie.fn.sentpc")
sbert = importlib.import_module("morie.fn.sbert")

DOC = [0, 1, 1, 2, 0, 3, 3, 3]
BETA1 = [[0.4, 0.3, 0.2, 0.1]]
BETA2 = [[0.55, 0.25, 0.15, 0.05], [0.05, 0.15, 0.25, 0.55]]
N_DW = [[3.0, 1.0, 0.0, 2.0],
        [0.0, 4.0, 1.0, 1.0],
        [2.0, 0.0, 3.0, 0.0]]


# ---------------------------------------------------------------- lda
def test_lda_single_topic_closed_form():
    """K=1 collapses eqs (6)-(7): gamma = alpha + N, every phi = 1."""
    r = lda.variational_inference(DOC, 0.7, BETA1, iters=50)
    assert abs(r["gamma"][0] - (0.7 + len(DOC))) < 1e-12
    assert all(abs(p[0] - 1.0) < 1e-12 for p in r["phi"])
    assert r["converged"]


def test_lda_elbo_is_monotone():
    """Blockwise maximisation of one bound -- it cannot fall."""
    vals = []
    for it in range(1, 12):
        ri = lda.variational_inference(DOC, 0.3, BETA2, iters=it,
                                       tol=0.0)
        vals.append(lda.elbo(DOC, 0.3, BETA2, ri["phi"], ri["gamma"]))
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i - 1] - 1e-12
    assert vals[-1] > vals[0]


def test_lda_jensen_gap_is_strict():
    """Eq. (6) uses exp(E[log theta]), strictly below E[theta]."""
    g = [1.4, 3.1, 0.6]
    el = lda.e_log_theta(g)
    S = sum(g)
    for i in range(3):
        assert math.exp(el[i]) < g[i] / S - 1e-6


def test_lda_recovers_disjoint_topics():
    docs = [[0, 1, 0, 1, 0, 1]] * 4 + [[2, 3, 2, 3, 2, 3]] * 4
    em = lda.variational_em(docs, 2, 4, alpha=0.1, iters=40, seed=3)
    mass = sorted(b[0] + b[1] for b in em["beta"])
    assert mass[0] < 0.02
    assert mass[1] > 0.98
    h = em["elbo_history"]
    for i in range(1, len(h)):
        assert h[i] >= h[i - 1] - 1e-8


def test_lda_rejects_bad_input():
    with pytest.raises(ValueError):
        lda.variational_inference([], 0.1, BETA1)
    with pytest.raises(ValueError):
        lda.variational_inference([9], 0.1, BETA1)
    with pytest.raises(ValueError):
        lda.e_log_theta([1.0, 0.0])


def test_lda_topic_words_are_ordered():
    top = lda.topic_words(BETA2, n_top=2, vocab=["a", "b", "c", "d"])
    assert top[0][0][0] == "a"
    assert top[1][0][0] == "d"


# --------------------------------------------------------------- plsa
def test_plsa_single_aspect_is_the_empirical_marginal():
    """K=1 makes the posterior 1, so one M step is exact."""
    tot = sum(sum(r) for r in N_DW)
    f = plsa.fit_plsa(N_DW, 1, iters=3)
    ew = [sum(N_DW[d][w] for d in range(3)) / tot for w in range(4)]
    ed = [sum(N_DW[d]) / tot for d in range(3)]
    assert max(abs(f["P_w_given_z"][0][w] - ew[w])
               for w in range(4)) < 1e-9
    assert max(abs(f["P_d_given_z"][0][d] - ed[d])
               for d in range(3)) < 1e-9


def test_plsa_single_aspect_loglik_closed_form():
    tot = sum(sum(r) for r in N_DW)
    ew = [sum(N_DW[d][w] for d in range(3)) / tot for w in range(4)]
    ed = [sum(N_DW[d]) / tot for d in range(3)]
    closed = sum(N_DW[d][w] * math.log(ed[d] * ew[w])
                 for d in range(3) for w in range(4) if N_DW[d][w] > 0)
    f = plsa.fit_plsa(N_DW, 1, iters=3)
    assert abs(f["final_loglik"] - closed) < 1e-9
    px = plsa.perplexity(N_DW, f["P_z"], f["P_d_given_z"],
                         f["P_w_given_z"])
    assert abs(px - math.exp(-closed / tot)) < 1e-9


def test_plsa_em_never_decreases_the_likelihood():
    f = plsa.fit_plsa(N_DW, 3, iters=60, seed=11)
    h = f["loglik_history"]
    for i in range(1, len(h)):
        assert h[i] >= h[i - 1] - 1e-9


def test_plsa_more_aspects_fit_at_least_as_well():
    f1 = plsa.fit_plsa(N_DW, 1, iters=3)
    f3 = plsa.fit_plsa(N_DW, 3, iters=60, seed=11)
    assert f3["final_loglik"] >= f1["final_loglik"] - 1e-9


def test_plsa_posterior_rows_are_distributions():
    f = plsa.fit_plsa(N_DW, 2, iters=20, seed=5)
    post = plsa.e_step(N_DW, f["P_z"], f["P_d_given_z"],
                       f["P_w_given_z"])
    for d in range(3):
        for w in range(4):
            if N_DW[d][w] > 0:
                assert abs(sum(post[d][w]) - 1.0) < 1e-12


def test_plsa_joint_is_a_distribution():
    f = plsa.fit_plsa(N_DW, 2, iters=20, seed=5)
    P = plsa.joint_probability(f["P_z"], f["P_d_given_z"],
                               f["P_w_given_z"])
    assert abs(sum(sum(r) for r in P) - 1.0) < 1e-9


def test_plsa_rejects_bad_input():
    with pytest.raises(ValueError):
        plsa.fit_plsa([[0.0, 0.0]], 1)
    with pytest.raises(ValueError):
        plsa.fit_plsa([[1.0, -1.0]], 1)
    with pytest.raises(ValueError):
        plsa.fit_plsa(N_DW, 0)


# -------------------------------------------------------------- sacrb
def test_sacrb_papineni_example_1():
    """The printed 2/7 of Papineni et al. (2002) Example 1."""
    cand = "the the the the the the the"
    refs = ["The cat is on the mat", "There is a cat on the mat"]
    mp = sacrb.modified_precision(
        sacrb.tokenize_13a(cand, True),
        [sacrb.tokenize_13a(x, True) for x in refs], 1)
    assert mp["numerator"] == 2
    assert mp["denominator"] == 7
    assert abs(mp["precision"] - 2.0 / 7.0) < 1e-12


def test_sacrb_clipping_is_what_caps_the_score():
    cand = sacrb.tokenize_13a("the the the the the the the", True)
    assert sum(sacrb.ngram_counts(cand, 1).values()) == 7
    mp = sacrb.modified_precision(
        cand, [sacrb.tokenize_13a("The cat is on the mat", True)], 1)
    assert mp["numerator"] == 2


def test_sacrb_identical_candidate_scores_one():
    b = sacrb.bleu(["the cat sat on the mat today"],
                   [["the cat sat on the mat today"]])
    assert abs(b["bleu"] - 1.0) < 1e-12
    assert b["bp"] == 1.0


def test_sacrb_brevity_penalty_closed_form():
    assert abs(sacrb.brevity_penalty(6, 7)
               - math.exp(1.0 - 7.0 / 6.0)) < 1e-15
    assert sacrb.brevity_penalty(8, 7) == 1.0
    assert sacrb.brevity_penalty(7, 7) == 1.0


def test_sacrb_best_match_is_closest_not_shortest():
    """Candidate 16, references 12 and 17: r must be 17."""
    c16 = " ".join("w%d" % i for i in range(16))
    r12 = " ".join("x%d" % i for i in range(12))
    r17 = " ".join("y%d" % i for i in range(17))
    b = sacrb.bleu([c16], [[r12, r17]], max_n=1)
    assert b["reference_length"] == 17
    assert abs(b["bp"] - math.exp(1.0 - 17.0 / 16.0)) < 1e-12


def test_sacrb_penalty_is_corpus_level_not_per_sentence():
    """A short sentence may be offset by a long one."""
    cands = ["a b", "c d e f g h i j"]
    refs = [["a b c d e"], ["c d e f g"]]
    b = sacrb.bleu(cands, refs, max_n=1)
    assert b["candidate_length"] == 10
    assert b["reference_length"] == 10
    assert b["bp"] == 1.0


def test_sacrb_tokenisation_changes_the_number():
    h = ["The cat, quite happily, sat on the mat."]
    r = [["The cat sat happily on the mat."]]
    a = sacrb.bleu(h, r, tokenizer="13a")["bleu"]
    b = sacrb.bleu(h, r, tokenizer="none")["bleu"]
    assert abs(a - b) > 1e-6


def test_sacrb_signature_records_every_choice():
    s = sacrb.signature("intl", True, 4, 1)
    assert s == ("nrefs:1|case:lc|tok:intl|ngram:4|"
                 "version:morie-sacrb-1")


def test_sacrb_rejects_bad_input():
    with pytest.raises(ValueError):
        sacrb.bleu(["a"], [])
    with pytest.raises(ValueError):
        sacrb.bleu(["a"], [[]])
    with pytest.raises(ValueError):
        sacrb.bleu(["a"], [["a"]], max_n=2, weights=[0.5])
    with pytest.raises(ValueError):
        sacrb.bleu(["a"], [["a"]], tokenizer="moses")


# ------------------------------------------------------------- sentpc
CASES = ["hello world", " leading", "trailing ", "a  double  space",
         "   ", "nospaces", u"日本語 テキスト", "mixed  中文 text"]


@pytest.mark.parametrize("s", CASES)
def test_sentpc_escape_is_exactly_invertible(s):
    assert sentpc.unescape_whitespace(sentpc.escape_whitespace(s)) == s


@pytest.mark.parametrize("s", CASES)
def test_sentpc_bpe_round_trip_is_lossless(s):
    m = sentpc.train_bpe(["hello world", "hello there world",
                          "a  double  space", "world of words"], 40)
    assert sentpc.decode(sentpc.encode_bpe(s, m)) == s


def test_sentpc_first_merge_is_the_modal_pair():
    """(a,a) occurs 4 times in aaabdaaabac, more than any other."""
    m = sentpc.train_bpe(["aaabdaaabac"], 5, add_prefix=False)
    assert m["merges"][0] == ("a", "a")


def test_sentpc_viterbi_matches_brute_force():
    lp = {"a": math.log(0.2), "b": math.log(0.15),
          "ab": math.log(0.5), "ba": math.log(0.05),
          "aba": math.log(0.3)}

    def brute(s):
        if not s:
            return (0.0, [])
        best = (-math.inf, None)
        for L in range(1, len(s) + 1):
            if s[:L] not in lp:
                continue
            sub = brute(s[L:])
            if sub[1] is None:
                continue
            cand = (lp[s[:L]] + sub[0], [s[:L]] + sub[1])
            if cand[0] > best[0]:
                best = cand
        return best

    for txt in ("abababa", "ababab", "aba", "ab"):
        v = sentpc.viterbi_segment(txt, lp, add_prefix=False)
        assert abs(v["logp"] - brute(txt)[0]) < 1e-12
        assert "".join(v["pieces"]) == txt


def test_sentpc_viterbi_beats_greedy_longest_match():
    lp = {"a": math.log(0.2), "b": math.log(0.15),
          "ab": math.log(0.5), "ba": math.log(0.05),
          "aba": math.log(0.3)}
    txt = "abababa"
    greedy, i = [], 0
    while i < len(txt):
        for L in range(min(3, len(txt) - i), 0, -1):
            if txt[i:i + L] in lp:
                greedy.append(txt[i:i + L])
                i += L
                break
        else:  # pragma: no cover - every character is a piece here
            i += 1
    gl = sum(lp[p] for p in greedy)
    assert sentpc.viterbi_segment(txt, lp,
                                  add_prefix=False)["logp"] > gl + 1e-9


def test_sentpc_rejects_uncoverable_input():
    with pytest.raises(ValueError):
        sentpc.viterbi_segment("xyz", {"a": -1.0}, add_prefix=False)
    with pytest.raises(ValueError):
        sentpc.train_bpe(["abc"], 0)
    with pytest.raises(ValueError):
        sentpc.train_bpe([], 10)


# -------------------------------------------------------------- sbert
def test_sbert_pair_cost_is_the_papers_50_million():
    pc = sbert.pair_cost(10000)
    assert pc["cross_encoder"] == 49995000
    assert pc["bi_encoder"] == 10000
    assert abs(pc["speedup"] - 4999.5) < 1e-9


def test_sbert_cosine_identities():
    assert abs(sbert.cosine_similarity([3.0, -1.0, 2.0],
                                       [3.0, -1.0, 2.0]) - 1.0) < 1e-12
    assert abs(sbert.cosine_similarity([1.0, 0.0], [0.0, 5.0])) < 1e-12
    assert abs(sbert.cosine_similarity([1.0, 0.0], [-1.0, 0.0])
               + 1.0) < 1e-12


def test_sbert_cosine_is_blind_where_features_are_not():
    """Cosine discards magnitude; (u, v, |u-v|) keeps it."""
    c1 = sbert.cosine_similarity([1.0, 0.0], [0.0, 1.0])
    c2 = sbert.cosine_similarity([2.0, 0.0], [0.0, 2.0])
    assert abs(c1 - c2) < 1e-12
    f1 = sbert.classification_features([1.0, 0.0], [0.0, 1.0])
    f2 = sbert.classification_features([2.0, 0.0], [0.0, 2.0])
    assert f1["features"] != f2["features"]
    assert f1["dim"] == 6


def test_sbert_abs_diff_is_elementwise():
    f = sbert.classification_features([1.0, 5.0], [4.0, 1.0])
    assert f["abs_diff"] == [3.0, 4.0]


def test_sbert_pooling_modes_disagree():
    T = [[1.0, 4.0], [3.0, 0.0], [-1.0, 2.0], [0.0, 0.0]]
    assert sbert.pool(T, "mean") == [0.75, 1.5]
    assert sbert.pool(T, "cls") == [1.0, 4.0]
    assert sbert.pool(T, "max") == [3.0, 4.0]


def test_sbert_mask_excludes_padding_from_the_mean():
    T = [[1.0, 4.0], [3.0, 0.0], [-1.0, 2.0], [0.0, 0.0]]
    assert sbert.pool(T, "mean", mask=[1, 1, 1, 0]) == [1.0, 2.0]


def test_sbert_embeds_each_sentence_once():
    calls = {"n": 0}

    def emb(s):
        calls["n"] += 1
        return [float(len(s)), float(s.count("a")), 1.0]

    pairs = [("aa", "bb"), ("aa", "cc"), ("bb", "cc"), ("aa", "bb")]
    st = sbert.sts_score(pairs, emb)
    assert st["embed_calls"] == 3
    assert st["cross_encoder_calls"] == 4
    assert calls["n"] == 3
    assert abs(st["scores"][0] - st["scores"][3]) < 1e-15


def test_sbert_ranking_needs_no_forward_passes():
    E = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    rk = sbert.rank_by_similarity([1.0, 0.0], E, top_k=4)
    assert [i for i, _ in rk["ranking"]] == [0, 1, 2, 3]
    assert rk["forward_passes"] == 0


def test_sbert_rejects_bad_input():
    with pytest.raises(ValueError):
        sbert.pool([[1.0]], "sum")
    with pytest.raises(ValueError):
        sbert.pool([[1.0], [2.0]], "mean", mask=[0, 0])
    with pytest.raises(ValueError):
        sbert.cosine_similarity([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        sbert.cosine_similarity([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        sbert.pair_cost(1)


def test_nlp_cheatsheets_are_present():
    for mod in (lda, plsa, sacrb, sentpc, sbert):
        assert len(mod.cheatsheet()) > 80
