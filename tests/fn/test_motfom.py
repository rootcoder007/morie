"""Anchored tests for motfom (Grant et al. 2011 FIMO; Staden 1994 DP).

Independent anchor: brute-force enumeration of all 4^w words under
the zero-order background reproduces the DP p-values exactly (same
integer discretization, different algorithm).
"""

import itertools
import math

from morie.fn.motfom import motfom, motif_fimo

PWM = [[0.5, 0.2, 0.2, 0.1],
       [0.1, 0.6, 0.2, 0.1],
       [0.25, 0.25, 0.25, 0.25]]
BG = [0.3, 0.2, 0.2, 0.3]


def _brute_pvalue(word_score_int, pwm, bg, scale):
    w = len(pwm)
    illr = []
    for j in range(w):
        row = []
        tot = sum(pwm[j])
        for a in range(4):
            row.append(int(round(math.log2((pwm[j][a] / tot) / bg[a]) * scale)))
        illr.append(row)
    p = 0.0
    for word in itertools.product(range(4), repeat=w):
        s = sum(illr[j][word[j]] for j in range(w))
        if s >= word_score_int:
            pr = 1.0
            for j in range(w):
                pr *= bg[word[j]]
            p += pr
    return p


def test_motfom_scores_by_hand():
    res = motfom("ACA", PWM, background=BG)
    # score of window ACA = log2(.5/.3) + log2(.6/.2) + log2(.25/.3)
    hand = (math.log2(0.5 / 0.3) + math.log2(0.6 / 0.2)
            + math.log2(0.25 / 0.3))
    assert abs(res["scores"][0] - hand) < 1e-12
    assert res["n_windows"] == 1


def test_motfom_pvalues_match_bruteforce():
    seq = "ACAGTCA"
    res = motfom(seq, PWM, background=BG, scale=1000)
    w = 3
    illr = []
    for j in range(w):
        tot = sum(PWM[j])
        illr.append([int(round(math.log2((PWM[j][a] / tot) / BG[a]) * 1000))
                     for a in range(4)])
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i in range(len(seq) - w + 1):
        s_int = sum(illr[j][idx[seq[i + j]]] for j in range(w))
        brute = _brute_pvalue(s_int, PWM, BG, 1000)
        assert abs(res["pvalues"][i] - brute) < 1e-12


def test_motfom_uniform_pwm_null():
    # PWM equal to the background: every score 0, every p-value 1.
    pwm = [[0.25] * 4, [0.25] * 4]
    res = motfom("ACGT", pwm)
    for s, p in zip(res["scores"], res["pvalues"]):
        assert abs(s) < 1e-12
        assert abs(p - 1.0) < 1e-12


def test_motfom_best_hit_and_alias():
    res = motfom("TTACAT", PWM, background=BG)
    assert res["best_position"] == 2  # window ACA maximizes the score
    assert motif_fimo is motfom
