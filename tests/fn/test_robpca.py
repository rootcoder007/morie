"""Tests for robpca (Hubert, Rousseeuw & Vanden Branden 2005)."""

import math

from morie.fn import _stats_core as _st
from morie.fn.robpca import (classify_outliers, robust_pca, robpca,
                             univariate_mcd)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


def _norm(v):
    n = math.sqrt(sum(t * t for t in v))
    return [t / n for t in v]


U1 = _norm([1.0, 0.5, -1.0, 0.0, 0.1])
_D = sum(U1[t] * [0.0, 1.0, 0.3, 2.0, -0.4][t] for t in range(5))
U2 = _norm([[0.0, 1.0, 0.3, 2.0, -0.4][t] - _D * U1[t] for t in range(5)])
_E = [0.0, 0.0, 0.0, 0.0, 1.0]
_A = sum(_E[t] * U1[t] for t in range(5))
_B = sum(_E[t] * U2[t] for t in range(5))
W = _norm([_E[t] - _A * U1[t] - _B * U2[t] for t in range(5)])


def _point(a, b, off):
    return [a * U1[t] + b * U2[t] + off * W[t] for t in range(5)]


def _panel(seed=3, kind=None, n_clean=70, n_out=10):
    r = _lcg(seed)
    rows = [_point(4 * _gauss(r), 2 * _gauss(r), 0.05 * _gauss(r))
            for _ in range(n_clean)]
    for _ in range(n_out if kind else 0):
        if kind == "good":
            rows.append(_point(30 + _gauss(r), 2 * _gauss(r),
                               0.05 * _gauss(r)))
        elif kind == "orth":
            rows.append(_point(4 * _gauss(r), 2 * _gauss(r), 6.0))
        elif kind == "bad":
            rows.append(_point(30 + _gauss(r), 2 * _gauss(r), 6.0))
    return rows


def _max_angle(A, B):
    worst = 0.0
    for a in A:
        proj = math.sqrt(sum(sum(a[t] * b[t] for t in range(len(a))) ** 2
                             for b in B))
        worst = max(worst,
                    math.degrees(math.acos(min(1.0, max(0.0, proj)))))
    return worst


def _classical(rows, k=2):
    from morie.fn import _array_core as np
    n, p = len(rows), len(rows[0])
    mu = [sum(r[j] for r in rows) / n for j in range(p)]
    C = [[sum((r[a] - mu[a]) * (r[b] - mu[b]) for r in rows) / (n - 1.0)
          for b in range(p)] for a in range(p)]
    vals, vecs = np.linalg.eigh(np.asarray(C, dtype=float))
    vals = [float(v) for v in vals]
    cols = [[float(vecs[i][j]) for i in range(p)] for j in range(p)]
    order = sorted(range(p), key=lambda j: -vals[j])[:k]
    return [cols[j] for j in order]


def test_univariate_mcd_is_the_tightest_window_and_is_consistent():
    vals = [10.0, 10.1, 10.2, 10.3, 50.0, 51.0]
    loc, _ = univariate_mcd(vals, 4, consistent=False)
    assert abs(loc - 10.15) < 1e-12
    r = _lcg(11)
    z = [_gauss(r) for _ in range(5000)]
    _, sc = univariate_mcd(z, int(0.75 * len(z)))
    assert abs(sc - 1.0) < 0.05
    _, raw = univariate_mcd(z, int(0.75 * len(z)), consistent=False)
    assert raw < 0.7          # the uncorrected shortest-half scale


def test_univariate_mcd_validation():
    for call in (lambda: univariate_mcd([1.0]),
                 lambda: univariate_mcd([1.0, 2.0, 3.0], 1),
                 lambda: univariate_mcd([1.0, 2.0, 3.0], 99)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_h_rule_and_orthonormal_loadings():
    rows = _panel()
    res = robust_pca(rows, k=2)
    assert res["h"] == max(int(0.75 * 70), (70 + 10 + 1) // 2)
    for c in res["loadings"]:
        assert abs(math.sqrt(sum(v * v for v in c)) - 1.0) < 1e-9
    assert abs(sum(res["loadings"][0][t] * res["loadings"][1][t]
                   for t in range(5))) < 1e-9
    assert _max_angle([U1, U2], res["loadings"]) < 1.0


def test_scores_and_distances_match_their_definitions():
    rows = _panel()
    res = robust_pca(rows, k=2)
    for i in (0, 17, 69):
        for j in range(res["k"]):
            want = sum((rows[i][t] - res["center"][t]) *
                       res["loadings"][j][t] for t in range(5))
            assert abs(res["scores"][i][j] - want) < 1e-9      # eq. 1
        sd = math.sqrt(sum(res["scores"][i][j] ** 2 /
                           res["eigenvalues"][j]
                           for j in range(res["k"])))
        assert abs(res["score_distance"][i] - sd) < 1e-12      # eq. 3
        fit = [res["center"][t] + sum(res["scores"][i][j] *
                                      res["loadings"][j][t]
                                      for j in range(res["k"]))
               for t in range(5)]
        od = math.sqrt(sum((rows[i][t] - fit[t]) ** 2 for t in range(5)))
        assert abs(res["orthogonal_distance"][i] - od) < 1e-12  # eq. 4
    assert abs(res["sd_cutoff"] - math.sqrt(_st.chi2.ppf(0.975, 2))) < 1e-12


def test_scores_are_invariant_under_rotation_and_shift():
    rows = _panel()
    res = robust_pca(rows, k=2)
    rot = [[math.cos(0.7), -math.sin(0.7), 0.0, 0.0, 0.0],
           [math.sin(0.7), math.cos(0.7), 0.0, 0.0, 0.0],
           [0.0, 0.0, math.cos(1.3), 0.0, -math.sin(1.3)],
           [0.0, 0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, math.sin(1.3), 0.0, math.cos(1.3)]]
    shift = [3.0, -1.0, 0.5, 2.0, -4.0]
    moved = [[sum(rot[a][t] * row[t] for t in range(5)) + shift[a]
              for a in range(5)] for row in rows]
    res_t = robust_pca(moved, k=2)
    for i in range(0, len(rows), 7):
        for j in range(2):
            assert abs(abs(res_t["scores"][i][j]) -
                       abs(res["scores"][i][j])) < 1e-7
    for a in range(5):
        want = sum(rot[a][t] * res["center"][t] for t in range(5)) + \
            shift[a]
        assert abs(res_t["center"][a] - want) < 1e-7


def test_contamination_turns_classical_pca_but_not_robpca():
    cont = _panel(9, "bad", n_out=12)
    rob = robust_pca(cont, k=2)
    assert _max_angle([U1, U2], rob["loadings"]) < 1.0
    assert _max_angle([U1, U2], _classical(cont)) > 1.0


def test_the_four_types_of_figure_1():
    for kind, want in (("good", "good leverage"),
                       ("orth", "orthogonal outlier"),
                       ("bad", "bad leverage")):
        cls = robust_pca(_panel(9, kind, n_out=10), k=2)["classification"]
        assert sum(1 for c in cls[70:] if c == want) >= 9


def test_classify_outliers_is_the_quadrant_rule():
    cls = classify_outliers([0.5, 9.0, 0.5, 9.0], [0.1, 0.1, 9.0, 9.0],
                            2.0, 1.0)
    assert cls == ["regular", "good leverage", "orthogonal outlier",
                   "bad leverage"]


def test_rank_detection_and_p_greater_than_n():
    rows = _panel()
    res = robust_pca(rows, k=2)
    assert res["rank"] == 3            # two directions plus the noise
    dep = [row + [2.0 * row[0] - 3.0 * row[1]] for row in rows]
    assert robust_pca(dep, k=2)["rank"] == 3
    r = _lcg(41)
    tall = []
    for _ in range(20):
        a, b = 5 * _gauss(r), 2 * _gauss(r)
        tall.append([a * math.sin(j) + b * math.cos(j / 3.0) +
                     0.01 * _gauss(r) for j in range(50)])
    res_tall = robust_pca(tall, k=2)
    assert res_tall["rank"] <= 19 and res_tall["k"] == 2


def test_both_selection_rules():
    r = _lcg(31)
    wide = []
    for _ in range(60):
        a, b, c = 6 * _gauss(r), 3 * _gauss(r), 0.02 * _gauss(r)
        wide.append([a, 0.4 * a + b, -a + c, 2 * b,
                     0.1 * a - 0.4 * b + c, 0.001 * _gauss(r)])
    assert robust_pca(wide, k="cumulative")["k"] <= \
        robust_pca(wide, k="ratio")["k"]


def test_alpha_and_reweighting_are_reachable():
    rows = _panel()
    assert robust_pca(rows, k=2, alpha=0.5)["h"] < \
        robust_pca(rows, k=2, alpha=0.95)["h"]
    assert robust_pca(rows, k=2, reweight=False)["reweighted"] is False
    assert robust_pca(rows, k=2)["consistency_factor"] > 0


def test_validation():
    rows = _panel()
    for call in (lambda: robust_pca(rows, k=2, alpha=0.4),
                 lambda: robust_pca(rows, k=99),
                 lambda: robust_pca(rows, k="scree"),
                 lambda: robust_pca([[1.0, 2.0]]),
                 lambda: robust_pca([[1.0, 2.0], [3.0]]),
                 lambda: robust_pca([[1.0, float("nan")], [1.0, 2.0],
                                     [3.0, 4.0]]),
                 lambda: robust_pca([[1.0, 1.0]] * 5)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert robpca is robust_pca
