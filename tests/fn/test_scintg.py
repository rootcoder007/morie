"""Tests for scintg (Korsunsky et al. 2019, Harmony)."""

import math

from morie.fn.scintg import (cluster_batch_counts, correct_batch,
                             harmony_integrate, l2_normalise,
                             maximum_diversity_clustering, scintg)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


R = [[0.8, 0.6, 0.1, 0.2], [0.2, 0.4, 0.9, 0.8]]
BAT = ["a", "a", "b", "b"]
Z = [[1.0, 0.0], [0.9, 0.4], [0.0, 1.0], [0.3, 0.95]]
Y = l2_normalise([[1.0, 0.1], [0.1, 1.0]])


def test_equations_5_and_6():
    c = cluster_batch_counts(R, BAT)
    assert abs(c["O"][0][0] - 1.4) < 1e-15
    assert abs(c["O"][1][1] - 1.7) < 1e-15
    assert abs(c["E"][0][0] - 0.5 * sum(R[0])) < 1e-15
    assert abs(sum(sum(r) for r in c["O"]) -
               sum(sum(r) for r in c["E"])) < 1e-12
    even = cluster_batch_counts([[0.5] * 4, [0.5] * 4], BAT)
    assert max(abs(even["O"][k][b] - even["E"][k][b])
               for k in range(2) for b in range(2)) < 1e-15


def test_equation_8_by_hand():
    sigma, theta = 0.2, 1.5
    cl = maximum_diversity_clustering(Z, BAT, K=2, sigma=sigma,
                                      theta=theta, max_iter=1, Y=Y)
    Zn = l2_normalise(Z)
    c0 = cluster_batch_counts([[0.5] * 4, [0.5] * 4], BAT)
    i, bi = 1, 0
    vals = []
    for k in range(2):
        dist = 1.0 - sum(Y[k][j] * Zn[i][j] for j in range(2))
        ratio = c0["O"][k][bi] / c0["E"][k][bi]
        vals.append((ratio ** -theta) * math.exp(-2.0 * dist / sigma))
    want = vals[0] / (vals[0] + vals[1])
    assert abs(cl["R"][0][i] - want) < 1e-9
    assert all(abs(sum(cl["R"][k][i] for k in range(2)) - 1.0) < 1e-12
               for i in range(4))


def test_theta_zero_ignores_the_batch_labels():
    a = maximum_diversity_clustering(Z, BAT, K=2, theta=0.0, max_iter=6,
                                     Y=Y)
    b = maximum_diversity_clustering(Z, ["x", "y", "x", "y"], K=2,
                                     theta=0.0, max_iter=6, Y=Y)
    assert max(abs(a["R"][k][i] - b["R"][k][i])
               for k in range(2) for i in range(4)) < 1e-12
    t = maximum_diversity_clustering(Z, BAT, K=2, theta=3.0, max_iter=6,
                                     Y=Y)
    assert max(abs(a["R"][k][i] - t["R"][k][i])
               for k in range(2) for i in range(4)) > 1e-6


def test_centroids_are_unit_length():
    cl = maximum_diversity_clustering(Z, BAT, K=2, max_iter=4, Y=Y)
    assert all(abs(math.sqrt(sum(v * v for v in y)) - 1.0) < 1e-9
               for y in cl["Y"])


def _corr_fixture():
    r = _lcg(19)
    N, d = 60, 3
    Zc = [[_gauss(r) for _ in range(d)] for _ in range(N)]
    bats = ["a" if i % 2 == 0 else "b" for i in range(N)]
    Rr = [[0.7 if i % 3 == 0 else 0.3 for i in range(N)],
          [0.3 if i % 3 == 0 else 0.7 for i in range(N)]]
    return Zc, bats, Rr


def test_the_intercept_row_is_zeroed():
    Zc, bats, Rr = _corr_fixture()
    got = correct_batch(Zc, Rr, bats, lam=1.0)
    assert all(all(abs(v) < 1e-15 for v in W[0]) for W in got["W"])


def test_a_reference_cell_never_moves():
    Zc, bats, Rr = _corr_fixture()
    ref = [i < 10 for i in range(len(Zc))]
    got = correct_batch(Zc, Rr, bats, lam=1.0, reference=ref)
    assert max(abs(got["Z"][i][j] - Zc[i][j])
               for i in range(10) for j in range(3)) < 1e-12
    assert max(abs(got["Z"][i][j] - Zc[i][j])
               for i in range(10, len(Zc)) for j in range(3)) > 1e-6


def test_the_ridge_is_needed_and_shrinks():
    Zc, bats, Rr = _corr_fixture()
    big = correct_batch(Zc, Rr, bats, lam=1e9)
    assert max(abs(big["Z"][i][j] - Zc[i][j])
               for i in range(len(Zc)) for j in range(3)) < 1e-6
    try:
        correct_batch(Zc, Rr, bats, lam=0.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def _integration_fixture():
    r = _lcg(4242)
    types = {"T1": [3.0, 0.0], "T2": [-3.0, 0.0], "T3": [0.0, 3.0]}
    shift = {"a": [0.0, 0.0], "b": [2.5, 2.0]}
    cells, batch, ctype = [], [], []
    for t, mu in types.items():
        for b, sh in shift.items():
            for _ in range(25):
                cells.append([mu[0] + sh[0] + 0.45 * _gauss(r),
                              mu[1] + sh[1] + 0.45 * _gauss(r)])
                batch.append(b)
                ctype.append(t)
    return cells, batch, ctype, types, shift


def _gap(Zx, ctype, types, shift, batch):
    tot = 0.0
    for t in types:
        cs = []
        for b in shift:
            idx = [i for i in range(len(Zx))
                   if ctype[i] == t and batch[i] == b]
            cs.append([sum(Zx[i][j] for i in idx) / len(idx)
                       for j in range(2)])
        tot += math.sqrt(sum((cs[0][j] - cs[1][j]) ** 2 for j in range(2)))
    return tot / len(types)


def test_batch_goes_and_cell_type_stays():
    cells, batch, ctype, types, shift = _integration_fixture()
    before = _gap(cells, ctype, types, shift, batch)
    res = scintg(cells, batch, K=3, sigma=0.1, theta=2.0, lam=1.0,
                 max_iter=10, seed=3)
    after = _gap(res["embedding"], ctype, types, shift, batch)
    assert after < 0.35 * before
    assert len(res["embedding"]) == len(cells)


def test_theta_lowers_dependence_and_the_printed_sign_raises_it():
    cells, batch, _c, _t, _s = _integration_fixture()
    z = maximum_diversity_clustering(cells, batch, K=3, theta=0.0, seed=3)
    p = maximum_diversity_clustering(cells, batch, K=3, theta=2.0, seed=3)
    lit = maximum_diversity_clustering(cells, batch, K=3, theta=2.0,
                                       seed=3, diversity="as_printed")
    assert p["objective"]["kl"] < z["objective"]["kl"]
    assert lit["objective"]["kl"] > z["objective"]["kl"]


def test_validation():
    Zc, bats, Rr = _corr_fixture()
    cells, batch, _c, _t, _s = _integration_fixture()
    for call in (lambda: scintg([], []),
                 lambda: scintg(cells, batch[:-1]),
                 lambda: scintg(cells, ["a"] * len(cells)),
                 lambda: scintg(cells, batch, max_iter=0),
                 lambda: maximum_diversity_clustering(Z, BAT, K=2,
                                                      sigma=0.0),
                 lambda: maximum_diversity_clustering(Z, BAT, K=2,
                                                      theta=-1.0),
                 lambda: maximum_diversity_clustering(Z, BAT, K=2,
                                                      diversity="plus"),
                 lambda: maximum_diversity_clustering(Z, BAT, K=99),
                 lambda: maximum_diversity_clustering(Z, BAT[:-1], K=2),
                 lambda: maximum_diversity_clustering(Z, BAT, K=2,
                                                      max_iter=0),
                 lambda: maximum_diversity_clustering(Z, BAT, K=3, Y=Y),
                 lambda: cluster_batch_counts([[0.5, 0.5]], ["a"]),
                 lambda: correct_batch(Zc, Rr, bats, lam=-1.0),
                 lambda: correct_batch(Zc, Rr, bats[:-1]),
                 lambda: correct_batch(Zc, Rr, bats, reference=[True]),
                 lambda: scintg([[1.0, 0.0], [float("nan"), 1.0]],
                                ["a", "b"])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert harmony_integrate is scintg
