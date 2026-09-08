"""Tests for snpqc1 (Marees et al. 2018 GWAS QC)."""

import math

from morie.fn.snpqc1 import (call_rates, heterozygosity, hwe_pvalue,
                             ibd_moments, ibs_given_ibd, kinship_matrix,
                             ld_prune, maf, pihat_matrix, sex_check,
                             snp_quality_control, snpqc1)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _hwe_panel(n=60, m=20, seed=13):
    rnd = _lcg(seed)
    out = []
    for _ in range(n):
        row = []
        for _ in range(m):
            u = rnd()
            row.append(0 if u < 0.49 else (1 if u < 0.91 else 2))
        out.append(row)
    return out


def test_thresholds_are_the_tutorials():
    t = snpqc1(_hwe_panel())["thresholds"]
    assert t["geno_relaxed"] == 0.2 and t["mind_relaxed"] == 0.2
    assert t["geno"] == 0.02 and t["mind"] == 0.02
    assert t["maf"] == 0.01
    assert t["hwe_case"] == 1e-10 and t["hwe_control"] == 1e-6
    assert t["hwe_quantitative"] == 1e-6
    assert t["het_sd"] == 3.0 and t["pihat"] == 0.2


def test_call_rates_maf_and_heterozygosity_by_hand():
    small = [[0, 1, None, 2], [0, None, None, 2], [1, 1, 0, None]]
    snp, ind = call_rates(small)
    assert abs(snp[1] - 2.0 / 3.0) < 1e-12
    assert abs(ind[1] - 0.5) < 1e-12
    f = maf(small)
    assert abs(f[0] - 1.0 / 6.0) < 1e-12
    assert f[3] == 0.0
    assert abs(heterozygosity(small)[0] - 1.0 / 3.0) < 1e-12


def test_hwe_tests():
    assert hwe_pvalue(25, 50, 25, "exact") > 0.6
    assert hwe_pvalue(0, 100, 0, "exact") < 1e-20
    assert hwe_pvalue(50, 0, 50, "exact") < 1e-20
    a, h, b = 12, 40, 48
    n = a + h + b
    p = (2 * a + h) / (2.0 * n)
    exp = [n * p * p, 2 * n * p * (1 - p), n * (1 - p) ** 2]
    chi = sum((o - e) ** 2 / e for o, e in zip([a, h, b], exp))
    assert abs(hwe_pvalue(a, h, b, "chisq") -
               math.erfc(math.sqrt(chi / 2.0))) < 1e-12
    # the chi-square approximation is anti-conservative on a rare variant
    assert hwe_pvalue(1, 2, 97, "exact") > 100 * hwe_pvalue(1, 2, 97,
                                                            "chisq")


def test_missingness_is_two_passes_snps_first():
    base = [[0 if (i + j) % 3 else 1 for j in range(20)]
            for i in range(100)]
    for i in range(30):
        base[i][0] = None
    for i in range(5):
        base[i][1] = None
    r = snpqc1(base, maf_threshold=0.0, het_sd=1e9, pihat=1e9)
    assert 0 in r["removed"]["geno_relaxed"]
    assert 1 not in r["removed"]["geno_relaxed"]
    assert 1 in r["removed"]["geno"]


def test_maf_and_heterozygosity_filters():
    rnd = _lcg(5)
    rare = [[0 if rnd() < 0.5 else (1 if rnd() < 0.8 else 2)
             for _ in range(10)] for _ in range(100)]
    for i in range(100):
        rare[i][3] = 0
    rare[0][3] = 1
    r = snpqc1(rare, maf_threshold=0.01, het_sd=1e9, pihat=1e9)
    assert 3 in r["removed"]["maf"]
    panel = _hwe_panel(m=40)
    panel[7] = [1] * 40
    r = snpqc1(panel, maf_threshold=0.0, pihat=1e9)
    assert 7 in r["removed"]["heterozygosity"]


def test_plink_ibd_moments_on_known_relationships():
    """Purcell et al. 2007: pi-hat = P(Z=2) + P(Z=1)/2."""
    rnd = _lcg(21)
    ns = 400
    fr = [0.15 + 0.6 * rnd() for _ in range(ns)]

    def draw(p):
        return sum(1 for _ in range(2) if rnd() < p)

    founders = [[draw(fr[j]) for j in range(ns)] for _ in range(20)]
    panel = [list(r) for r in founders]
    panel.append(list(founders[3]))                 # duplicate
    child = []
    for j in range(ns):
        par = founders[5][j]
        a = 1 if par == 2 else (0 if par == 0 else (1 if rnd() < 0.5
                                                    else 0))
        child.append(a + (1 if rnd() < fr[j] else 0))
    panel.append(child)                             # offspring of 5
    Z, P = ibd_moments(panel)
    assert abs(P[3][20] - 1.0) < 0.02 and Z[3][20][2] > 0.97
    assert abs(P[5][21] - 0.5) < 0.05 and Z[5][21][1] > 0.9
    un = [P[i][k] for i in range(20) for k in range(i + 1, 20)]
    assert sum(un) / len(un) < 0.1 and max(un) < 0.25
    assert all(abs(P[i][k] - (Z[i][k][2] + 0.5 * Z[i][k][1])) < 1e-12
               for i in range(22) for k in range(22))
    assert all(abs(sum(Z[i][k]) - 1.0) < 1e-9
               for i in range(22) for k in range(22))
    assert pihat_matrix(panel)[3][20] == P[3][20]


def test_ibs_given_ibd_table():
    tab = ibs_given_ibd(4000, 4000, correction=False)
    p = q = 0.5
    assert abs(tab[0][0] - 2 * p ** 2 * q ** 2) < 1e-12
    assert abs(tab[0][1] - (4 * p ** 3 * q + 4 * p * q ** 3)) < 1e-12
    assert abs(tab[0][2] - (p ** 4 + q ** 4 + 4 * p ** 2 * q ** 2)) < 1e-12
    assert abs(tab[1][1] - 2 * p * q) < 1e-12
    assert tab[1][0] == 0.0 and tab[2] == [0.0, 0.0, 1.0]
    for z in range(3):
        assert abs(sum(tab[z]) - 1.0) < 1e-12
    big = max(abs(a - b) for z in range(3)
              for a, b in zip(ibs_given_ibd(2000, 2000, True)[z],
                              ibs_given_ibd(2000, 2000, False)[z]))
    small = max(abs(a - b) for z in range(3)
                for a, b in zip(ibs_given_ibd(20, 20, True)[z],
                                ibs_given_ibd(20, 20, False)[z]))
    assert big < 1e-3 < small


def test_relatedness_on_a_duplicate():
    rnd = _lcg(21)
    dup = [[(0 if rnd() < 0.5 else (1 if rnd() < 0.8 else 2))
            for _ in range(120)] for _ in range(40)]
    dup[9] = list(dup[3])
    K = kinship_matrix(dup)
    assert K[3][9] > 0.7
    assert max(abs(K[3][k]) for k in range(40) if k not in (3, 9)) < 0.4
    for route in ("pihat", "kinship"):
        r = snpqc1(dup, maf_threshold=0.01, het_sd=1e9,
                   relatedness=route)
        assert r["relatedness"] == route
        assert (9 in r["removed"]["relatedness"] or
                3 in r["removed"]["relatedness"])
    try:
        snpqc1(dup, relatedness="grm")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_hwe_filter_drops_only_the_bad_snp():
    hw = _hwe_panel(m=5, seed=41)
    for i in range(len(hw)):
        hw[i][2] = 1
    pheno = [1] * 30 + [0] * 30
    r = snpqc1(hw, pheno, trait="binary", maf_threshold=0.0, het_sd=1e9,
               pihat=1e9)
    assert r["removed"]["hwe"] == [2]


def test_sex_check_and_pruning():
    rnd = _lcg(31)
    male = [[0 if rnd() < 0.7 else 2 for _ in range(80)] for _ in range(10)]
    female = [[0 if rnd() < 0.4 else (1 if rnd() < 0.85 else 2)
               for _ in range(80)] for _ in range(10)]
    sx = sex_check(male + female, [1] * 10 + [2] * 10)
    assert all(v == 1 for v in sx["inferred_sex"][:10])
    assert sx["discrepant"] == []
    swapped = sex_check(male + female, [2] * 10 + [1] * 10)
    assert len(swapped["discrepant"]) >= 18
    corr = [[0, 0, 1], [1, 1, 0], [2, 2, 1], [0, 0, 0], [1, 1, 2],
            [2, 2, 1]]
    kept = ld_prune(corr, window=3, step=3, r2=0.2)
    assert 1 not in kept and 2 in kept


def test_validation():
    base = _hwe_panel()
    for call in (lambda: snpqc1([]),
                 lambda: snpqc1([[0, 3], [1, 1]]),
                 lambda: snpqc1([[0, 1], [1]]),
                 lambda: snpqc1(base, trait="ordinal"),
                 lambda: snpqc1(base, maf_threshold=0.5),
                 lambda: hwe_pvalue(1, 1, 1, "fisher"),
                 lambda: hwe_pvalue(-1, 1, 1)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert snp_quality_control is snpqc1
