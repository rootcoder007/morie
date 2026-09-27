"""Bilder & Loughin eqs 6.13, 6.26-6.33: binGroup2, MRCV and exact-enumeration checks."""

import itertools
import math

from morie.fn.gtdorfman import gtdorfman
from morie.fn.gtprev import gtprev
from morie.fn.gtregem import gtregem
from morie.fn.spmi import spmi


def lcg(seed=12345):
    xs = seed
    while True:
        xs = xs * 16807 % 2147483647
        yield xs / 2147483647


def test_dorfman_matches_bingroup2_and_enumeration():
    d = gtdorfman(0.05, 0.95, 0.97, 0.9, 0.99, size=6)
    assert abs(d["expected_tests"] - 2.64229276375) < 1e-10  # binGroup2::opChar1 D2
    o = d["overall"]
    assert (
        abs(o["PSp"] - 0.997618784625) < 1e-12
        and abs(o["PPPV"] - 0.94974347204575) < 1e-12
        and abs(o["PNPV"] - 0.992408280340981) < 1e-12
    )
    h = gtdorfman(0.0041, 0.967, 0.976, size=20)["overall"]
    assert abs(h["PSp"] - 0.997724536131) < 1e-12 and abs(h["PPPV"] - 0.628502507799) < 1e-12
    # heterogeneous risks: enumerate all true-status patterns and test outcomes
    p = [0.01, 0.02, 0.05, 0.1, 0.2]
    se, sp, ser, spr = 0.95, 0.97, 0.9, 0.99
    r = gtdorfman(p, se, sp, ser, spr)
    et, py = 0.0, [0.0] * 5
    for s in itertools.product((0, 1), repeat=5):
        pr = math.prod(pi if v else 1 - pi for pi, v in zip(p, s))
        pz = se if any(s) else 1 - sp
        et += pr * (1 + 5 * pz)
        for i in range(5):
            py[i] += pr * pz * (ser if s[i] else 1 - spr)
    assert abs(r["expected_tests"] - et) < 1e-14
    for i in range(5):
        assert abs(r["individual"][i]["P(Y=1)"] - py[i]) < 1e-15


def test_group_testing_prevalence_matches_bingroup2():
    ref = {
        "CP": (0.00383809926017364, 0.0543240825507736),
        "Wald": (0.0, 0.0400890177398339),
        "score": (0.00632494931890608, 0.0516362362808093),
    }
    for m, (lo, hi) in ref.items():
        r = gtprev(3, 7, 24, ci=m)
        assert abs(r["ci"][0] - lo) < 1e-12 and abs(r["ci"][1] - hi) < 1e-12
    r = gtprev(3, 7, 24)
    assert (
        abs(r["estimate"] - 0.0188951194267189) < 1e-15 and abs(r["estimate"] - (1 - (1 - 3 / 24) ** (1 / 7))) < 1e-16
    )
    assert abs(r["b_hat"] - 52.4026315766191) < 1e-6  # the book's optimize() code
    assert abs(r["eb1"] - 0.0188574984247741) < 1e-12 and abs(r["eb2"] - 0.0185957956040668) < 1e-12


def test_xie_em_reaches_the_mle():
    u = lcg()
    x, y = [], []
    for _ in range(150):
        x.append(4 * next(u) - 2)
        y.append(1 if next(u) < 1 / (1 + math.exp(2.5 - 1.5 * x[-1])) else 0)
    grp = [i // 5 for i in range(150)]
    z = []
    for k in range(30):
        pos = any(y[i] for i in range(150) if grp[i] == k)
        r = next(u)
        z.append(int(r < 0.95) if pos else int(r > 0.98))
    fit = gtregem(z, grp, [[1, v] for v in x], se=0.95, sp=0.98)
    assert fit["converged"]
    assert abs(fit["beta"][0] + 1.1479358561534339) < 1e-7 and abs(fit["beta"][1] - 0.9762824948165424) < 1e-7  # R arm
    assert abs(fit["beta"][0] + 1.1474498) < 2e-3  # binGroup2::gtReg(method = "Xie"), looser stopping rule

    # the score of the observed log-likelihood vanishes at the MLE
    def ll(b):
        tot = 0.0
        for k in range(30):
            q = math.prod(1 - 1 / (1 + math.exp(-(b[0] + b[1] * x[i]))) for i in range(5 * k, 5 * k + 5))
            pz = 0.95 - 0.93 * q
            tot += math.log(pz) if z[k] else math.log(1 - pz)
        return tot

    b = fit["beta"]
    for a in range(2):
        e = [1e-6 * (c == a) for c in range(2)]
        assert abs((ll([b[0] + e[0], b[1] + e[1]]) - ll([b[0] - e[0], b[1] - e[1]])) / 2e-6) < 1e-5


def test_spmi_matches_mrcv():
    u = lcg(777)
    W, Y = [], []
    for _ in range(120):
        a = next(u)
        W.append([int(next(u) < 0.3 + 0.3 * a), int(next(u) < 0.5), int(next(u) < 0.2 + 0.5 * a)])
        Y.append([int(next(u) < 0.25 + 0.5 * a), int(next(u) < 0.4)])
    s = spmi(W, Y)
    # X2_S equals the Pearson form (6.13) when no cell is empty
    n = 120
    tot = 0.0
    for i in range(3):
        for j in range(2):
            gi = sum(w[i] for w in W) / n
            gj = sum(y[j] for y in Y) / n
            gij = sum(w[i] * y[j] for w, y in zip(W, Y)) / n
            tot += n * (gij - gi * gj) ** 2 / (gi * gj * (1 - gi) * (1 - gj))
    assert abs(s["statistic"] - tot) < 1e-10 and abs(s["statistic"] - 12.050532970495411) < 1e-11
    assert (
        abs(s["rs2_statistic"] - 12.223014793644012) < 1e-9 and abs(s["rs2_df"] - 6.0858792670353621) < 1e-9
    )  # MRCV::MI.test
