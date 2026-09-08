"""Tests for motfsr (Bailey & Elkan 1994 MM/MEME motif discovery)."""

import math

from morie.fn.motfsr import (bayes_threshold, mm_fit, motfsr, motif_meme,
                             score_sequence)

ALPHA = "ACGT"


def _planted(n=6, w=8, motif="TGACGTCA", seed=7):
    """Sequences with ``motif`` planted at a known position in each."""
    state = seed
    out, truth = [], []
    for i in range(n):
        letters = []
        for _ in range(40):
            state = (1103515245 * state + 12345) % (1 << 31)
            letters.append(ALPHA[int(state / (1 << 31) * 4)])
        state = (1103515245 * state + 12345) % (1 << 31)
        p = 4 + int(state / (1 << 31) * 25)
        s = "".join(letters)
        out.append(s[:p] + motif + s[p + w:])
        truth.append((i, p))
    return out, truth


def test_recovers_a_planted_motif():
    seqs, truth = _planted()
    res = motfsr(seqs, 8, ALPHA, max_starts=40)
    m = res["motifs"][0]
    assert m["consensus"] == "TGACGTCA"
    top = set((i, j) for i, j, _ in
              sorted(m["sites"], key=lambda r: -r[2])[:len(truth)])
    assert top == set(truth)


def test_e_step_and_m_step_match_the_printed_equations():
    seqs = ["ACGT", "TTTT"]
    theta0 = [[0.4, 0.1, 0.1, 0.4], [0.7, 0.1, 0.1, 0.1],
              [0.1, 0.7, 0.1, 0.1]]
    fit = mm_fit(seqs, 2, ALPHA, theta0, 0.2, beta=0.0, max_iter=1,
                 normalize_overlaps=False)
    # equation (4) for the first subsequence, "AC"
    lp1 = math.log(0.7) + math.log(0.7)
    lp2 = math.log(0.4) + math.log(0.1)
    want = (0.2 * math.exp(lp1)) / (0.2 * math.exp(lp1) +
                                    0.8 * math.exp(lp2))
    assert abs(fit["z"][0][0] - want) < 1e-12
    # equation (5)
    n = sum(len(r) for r in fit["z"])
    assert abs(fit["lambda1"] - sum(sum(r) for r in fit["z"]) / n) < 1e-12
    # every reestimated row is a distribution
    for row in fit["theta"]:
        assert abs(sum(row) - 1.0) < 1e-12


def test_pseudo_counts_release_a_zero_frequency():
    """Equation 13 exists because a zero is absorbing under equation 12."""
    theta0 = [[0.25] * 4, [0.0, 0.5, 0.25, 0.25], [0.25] * 4, [0.25] * 4]
    seqs = ["ACGT", "AAGT", "ACGA"]
    stuck = mm_fit(seqs, 3, ALPHA, theta0, 0.3, beta=0.0, max_iter=50,
                   normalize_overlaps=False)
    free = mm_fit(seqs, 3, ALPHA, theta0, 0.3, beta=0.5, max_iter=50,
                  normalize_overlaps=False)
    assert stuck["theta"][1][0] == 0.0
    assert free["theta"][1][0] > 1e-3


def test_em_does_not_decrease_the_log_likelihood():
    fit = mm_fit(["ACGTTGCAACGTA", "TTACGTGGCCAAT", "GACGTTTACGTAC"], 5,
                 ALPHA, None, 0.1, max_iter=60, normalize_overlaps=False)
    tr = fit["log_likelihood_trace"]
    assert all(tr[t] >= tr[t - 1] - 1e-9 for t in range(1, len(tr)))
    assert tr[-1] > tr[0]


def test_window_constraint_bounds_the_overlapping_z():
    poly = ["AAAAAAAAAAGATTCAAAAAAAAAA", "AAAAAAAAAAAGATTCAAAAAAAAA",
            "AAAAAAAAAGATTCAAAAAAAAAAA"]
    start = [[0.7, 0.1, 0.1, 0.1]] + [[0.85, 0.05, 0.05, 0.05]] * 5
    free = mm_fit(poly, 5, ALPHA, start, 0.1, max_iter=300,
                  normalize_overlaps=False)
    held = mm_fit(poly, 5, ALPHA, start, 0.1, max_iter=300,
                  normalize_overlaps=True)

    def worst(z):
        return max(sum(r[j:j + 5]) for r in z for j in range(len(r) - 4))

    assert worst(free["z"]) > 4.0
    assert worst(held["z"]) <= 1.0 + 1e-9


def test_the_score_is_the_log_likelihood_ratio():
    seqs, _ = _planted()
    m = motfsr(seqs, 8, ALPHA, max_starts=40)["motifs"][0]
    word = "TGACGTCA"
    got = score_sequence(m["log_odds"], word, ALPHA)[0]
    want = sum(math.log(m["motif"][j][ALPHA.index(ch)] /
                        m["background"][ALPHA.index(ch)])
               for j, ch in enumerate(word))
    assert abs(got - want) < 1e-9
    assert abs(m["threshold"] -
               math.log((1 - m["lambda1"]) / m["lambda1"])) < 1e-12


def test_bayes_threshold_scales_with_the_loss_matrix():
    assert bayes_threshold(0.5) == 0.0
    assert abs(bayes_threshold(0.5, [[0.0, 4.0], [2.0, 0.0]]) -
               math.log(2.0)) < 1e-12


def test_a_second_pass_finds_a_second_motif():
    state = 999
    seqs = []
    for _ in range(12):
        letters = []
        for _ in range(60):
            state = (1103515245 * state + 12345) % (1 << 31)
            letters.append(ALPHA[int(state / (1 << 31) * 4)])
        bg = "".join(letters)
        seqs.append(bg[:6] + "CACGTGAC" + bg[14:36] + "TTTATAGG" + bg[44:])
    res = motfsr(seqs, 8, ALPHA, n_motifs=2, max_starts=60)
    assert len(res["motifs"]) == 2
    assert (res["motifs"][0]["consensus"] !=
            res["motifs"][1]["consensus"])
    # the erasing factors collapsed where the first motif was found
    i, j, _ = res["motifs"][0]["sites"][0]
    assert res["erasing"][i][j] < 0.1


def test_validation():
    for bad in (dict(w=0), dict(w=99)):
        try:
            motfsr(["ACGT"], **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        motfsr(["ACGX"], 3, ALPHA)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_alias():
    assert motif_meme is motfsr
