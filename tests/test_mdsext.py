import math

from morie.fn._qpcore import ssum
from morie.fn.mdsext import (
    mds_anisotropy,
    mds_bootstrap,
    mds_flip,
    mds_jackknife,
    mds_polarity,
    mds_reflect,
    procrustes_oblique,
    smacof_indiff,
)
from morie.fn.mdsops import _dist, smacof

D1 = [[0, 3, 4, 6, 5], [3, 0, 5, 4, 2], [4, 5, 0, 3, 6], [6, 4, 3, 0, 4], [5, 2, 6, 4, 0]]
D2 = [[0, 2, 5, 6, 4], [2, 0, 4, 5, 3], [5, 4, 0, 2, 6], [6, 5, 2, 0, 5], [4, 3, 6, 5, 0]]
D3 = [[0, 4, 3, 5, 6], [4, 0, 6, 3, 2], [3, 6, 0, 4, 5], [5, 3, 4, 0, 3], [6, 2, 5, 3, 0]]


def _lower(D):
    n = len(D)
    return [D[i][j] for j in range(n) for i in range(j + 1, n)]


def test_smacof_indiff_stress_recomputed_and_nested_constraints():
    fits = {
        c: smacof_indiff([D1, D2, D3], 2, constraint=c, eps=1e-12, itmax=5000)
        for c in ("identity", "indscal", "idioscal")
    }
    for r in fits.values():
        tot = 0.0
        for D, X in zip([D1, D2, D3], r.conf):
            d = _lower(D)
            dh = [v * math.sqrt(len(d) / ssum(x * x for x in d)) for v in d]
            tot += ssum((a - b) ** 2 for a, b in zip(dh, _dist(X)))
        assert abs(r.stress - math.sqrt(tot / 3 / 10)) < 1e-12
        assert abs(ssum(r.sps) - 100) < 1e-10 and abs(ssum(r.spp) - 100) < 1e-10
    assert fits["indscal"].stress < fits["identity"].stress
    assert fits["idioscal"].stress <= fits["indscal"].stress + 1e-9
    for C in fits["indscal"].cweights:
        assert C[0][1] == 0.0 and C[1][0] == 0.0


def test_smacof_indiff_identity_of_equal_sources_is_smacof():
    r = smacof_indiff([D1, D1, D1], 2, constraint="identity", eps=1e-12, itmax=5000)
    assert abs(r.stress - smacof(D1, 2, eps=1e-12, itmax=5000).stress) < 1e-9
    assert (
        round(
            smacof_indiff(
                [[[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]] * 2, 1, constraint="identity"
            ).stress,
            10,
        )
        == 0.0
    )


def test_jackknife_measures_recomputed():
    for method in ("standard", "smacof"):
        r = mds_jackknife(D1, method=method)
        yy, y0 = r.jackknife_conf, r.comparison_conf
        if method == "standard":
            den = ssum(v * v for Y in yy for row in Y for v in row)
            num = ssum((Y[i][k] - y0[i][k]) ** 2 for Y in yy for i in range(5) for k in range(2))
            assert abs(r.stab - (1 - num / den)) < 1e-12
            cr = 1 - 5 * ssum((r.smacof_conf[i][k] - y0[i][k]) ** 2 for i in range(5) for k in range(2)) / den
            assert abs(r.cross - cr) < 1e-12
        assert abs(r.disp - (2 - r.stab - r.cross)) < 1e-15
        assert abs(r.loss - ssum((y0[i][k] - Y[i][k]) ** 2 for Y in yy for i in range(5) for k in range(2))) < 1e-14
    assert mds_jackknife(D1).loss < mds_jackknife(D1, method="smacof").loss
    D = [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
    r = mds_jackknife(D)
    assert 0 < r.stab <= 1 and abs(r.disp - (2 - r.stab - r.cross)) < 1e-15


X6 = [[1, 2, 3, 1], [2, 1, 4, 0], [3, 5, 2, 2], [4, 3, 6, 1], [5, 6, 5, 3], [6, 4, 8, 2]]


def test_bootstrap_identity_resamples_and_quantiles():
    r = mds_bootstrap(X6, 2, method_dat="euclidean", resamples=[list(range(6))] * 4)
    assert all(abs(s - r.stress) < 1e-15 for s in r.stressvec)
    assert abs(r.stab - 1) < 1e-12
    assert all(abs(v) < 1e-20 for C in r.cov for row in C for v in row)
    r = mds_bootstrap(X6, 2, method_dat="pearson", nrep=7, seed=3)
    s = sorted(r.stressvec)
    h = 6 * 0.025
    assert abs(r.bootci[0] - (s[0] + h * (s[1] - s[0]))) < 1e-15
    assert r.nrep == 7 and len(r.cov) == 4
    r = mds_bootstrap(X6, 2, method_dat="euclidean", nrep=5)
    assert (len(r.cov), r.nrep) == (4, 5)


def test_procrustes_oblique_optimality():
    A = [[0.8, 0.1], [0.7, 0.2], [0.2, 0.9], [0.1, 0.7], [0.5, 0.5]]
    B = [[1, 0], [1, 0], [0, 1], [0, 1], [0.5, 0.5]]
    r = procrustes_oblique(A, B, eps=1e-12)
    T = r.T
    det = T[0][0] * T[1][1] - T[0][1] * T[1][0]
    Ti = [[T[1][1] / det, -T[0][1] / det], [-T[1][0] / det, T[0][0] / det]]
    L = [[ssum(a[t] * Ti[k][t] for t in range(2)) for k in range(2)] for a in A]
    assert max(abs(L[i][k] - r.loadings[i][k]) for i in range(5) for k in range(2)) < 1e-14
    assert abs(r.f - ssum((L[i][k] - B[i][k]) ** 2 for i in range(5) for k in range(2))) < 1e-14
    assert all(abs(ssum(T[a][k] ** 2 for a in range(2)) - 1) < 1e-14 for k in range(2))
    assert r.converged and r.Phi[0][1] > 0.5
    assert round(procrustes_oblique([[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 1], [1, 1]]).f, 12) == 0.0


def test_reflect_flip_polarity_anisotropy():
    assert mds_reflect([[1, -3], [-2, 1]]).signs == [-1, -1]
    assert mds_reflect([[1, 2], [3, 4]], target=[[-1, 1], [-1, 1]]).conf == [[-1, 2], [-3, 4]]
    r = mds_flip([[0, 0], [1, 0], [0, 2]], [[0, 0], [-1, 0], [0, 2]])
    assert (r.reflected, r.flipped_axes) == (True, [0])
    assert r.ss_after < 1e-28 and abs(r.determinant + 1) < 1e-14
    r = mds_flip([[0, 0], [1, 0], [0, 2]], [[0, 0], [0, 1], [-2, 0]])
    assert not r.reflected and r.ss_after < 1e-28
    p = mds_polarity([[1, 2], [-1, -2], [0, 0]], [1, 0])
    assert p.signs == [-1, 1] and p.conf == [[-1, 2], [1, -2], [0, 0]] and p.poles == [[0, 1], [1, 0]]
    a = mds_anisotropy([[-2, 0], [2, 0], [0, -1], [0, 1]])
    assert (round(a.ratio, 12), round(a.angle, 12)) == (4.0, 0.0)
    assert abs(a.eigenvalues[0] - 8 / 3) < 1e-14 and abs(a.eccentricity - math.sqrt(0.75)) < 1e-15
    b = mds_anisotropy([[-1, -1], [1, 1], [-0.1, 0.1], [0.1, -0.1]])
    assert abs(b.angle - 45) < 1e-10
