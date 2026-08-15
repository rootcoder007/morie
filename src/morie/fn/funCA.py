# morie.fn -- function file (rootcoder007/morie)
r"""Functional canonical analysis for two square-integrable processes.

Classical CCA finds the linear combinations of two random vectors with
maximal correlation. Carried over to curves the problem is *ill-posed*:
the canonical correlation between two :math:`L^2` processes is almost
always 1, attained by weight functions concentrating on directions in
which one of the covariance operators has arbitrarily small eigenvalue.
Nothing about that maximum is estimable, and an unregularised sample
version simply reports 1 for any two datasets.

He, Müller and Wang's answer is to restrict the weight functions to the
span of the leading functional principal components of each process, so
the canonical problem is solved in a finite-dimensional subspace where
the covariance operators are invertible. With scores
:math:`\xi \in \mathbb{R}^{p}` and :math:`\eta \in \mathbb{R}^{q}` the
canonical correlations are the singular values of

.. math:: \Sigma_{\xi\xi}^{-1/2}\,\Sigma_{\xi\eta}\,
          \Sigma_{\eta\eta}^{-1/2},

and the canonical weight FUNCTIONS are the corresponding combinations of
eigenfunctions. The truncation is what makes the answer mean anything,
so ``p`` and ``q`` are reported alongside the correlations rather than
being absorbed into a default.

**The first correlation is not a test statistic.** It is bounded above by
1 by construction and rises monotonically with p and q; comparing it
across truncation levels is the only honest way to read it, which is why
``explained_x`` and ``explained_y`` come back with it.

References
----------
He, G., Müller, H.-G. and Wang, J.-L. (2003) "Functional canonical
analysis for square integrable stochastic processes", *Journal of
Multivariate Analysis* **85**(1), 54-77,
doi:10.1016/S0047-259X(02)00056-8. The ill-posedness of unrestricted
functional CCA, the conditions for the canonical correlations to exist,
and the FPC-restricted estimator.

Leurgans, S. E., Moyeed, R. A. and Silverman, B. W. (1993) "Canonical
correlation analysis when the data are curves", *Journal of the Royal
Statistical Society B* **55**(3), 725-740. The earlier smoothing-based
regularisation of the same problem.

Ramsay, J. O. and Silverman, B. W. (2005) *Functional Data Analysis*,
2nd ed., Springer, Ch. 11 (canonical correlation for functional data and
why penalisation is unavoidable).
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["functional_cca"]

_EPS = 1e-12


def _grid_weights(n_t):
    if n_t < 2:
        return [1.0]
    h = 1.0 / (n_t - 1)
    w = [h] * n_t
    w[0] = 0.5 * h
    w[-1] = 0.5 * h
    return w


def _fpca(C, w, n_keep):
    """Eigenfunctions of a covariance operator discretised with weights w."""
    T = len(w)
    rw = [math.sqrt(v) for v in w]
    Cw = [[rw[s] * C[s][t] * rw[t] for t in range(T)] for s in range(T)]
    lam, U = k.jacobi(Cw)
    order = sorted(range(T), key=lambda j: -lam[j])
    lam = [max(lam[j], 0.0) for j in order]
    phi = [[U[s][j] / (rw[s] if rw[s] > _EPS else 1.0) for s in range(T)]
           for j in order]
    # sign is arbitrary; pin it so the reported weights are reproducible
    for j in range(len(phi)):
        top = max(range(T), key=lambda t: abs(phi[j][t]))
        if phi[j][top] < 0.0:
            phi[j] = [-v for v in phi[j]]
    return lam[:n_keep], phi[:n_keep], lam


def _sym_inv_sqrt(M):
    """M^{-1/2} for a symmetric positive-definite M, via its spectrum."""
    d, V = k.jacobi(M)
    p = len(d)
    out = [[0.0] * p for _ in range(p)]
    for a in range(p):
        for b in range(p):
            s = 0.0
            for j in range(p):
                if d[j] > _EPS:
                    s += V[a][j] * V[b][j] / math.sqrt(d[j])
            out[a][b] = s
    return out


def functional_cca(X, Y, p=None, q=None):
    r"""Canonical correlations between two sets of curves.

    Parameters
    ----------
    X, Y : (n, T) and (n, S) array-like
        Paired curves, one per row, each set on its own common grid.
    p, q : int, optional
        Truncation levels for the two functional principal component
        bases. Default: enough components for 95% of each process's
        variance, reported back as ``p`` and ``q``.

    Returns
    -------
    RichResult with ``correlations``, the canonical weight functions
    ``weights_x`` / ``weights_y`` on their grids, the canonical
    ``variates_x`` / ``variates_y``, and the truncation actually used.
    """
    Xm = [[float(v) for v in r] for r in k.mat(X)]
    Ym = [[float(v) for v in r] for r in k.mat(Y)]
    n = len(Xm)
    if n == 0 or len(Ym) != n:
        raise ValueError("funCA: X and Y must hold the same number of curves")
    if n < 3:
        raise ValueError("funCA: canonical analysis needs at least three "
                         "paired curves")
    T, S = len(Xm[0]), len(Ym[0])
    wx, wy = _grid_weights(T), _grid_weights(S)

    xbar = [sum(Xm[i][t] for i in range(n)) / n for t in range(T)]
    ybar = [sum(Ym[i][t] for i in range(n)) / n for t in range(S)]
    Xc = [[Xm[i][t] - xbar[t] for t in range(T)] for i in range(n)]
    Yc = [[Ym[i][t] - ybar[t] for t in range(S)] for i in range(n)]

    Cx = [[sum(Xc[i][a] * Xc[i][b] for i in range(n)) / n
           for b in range(T)] for a in range(T)]
    Cy = [[sum(Yc[i][a] * Yc[i][b] for i in range(n)) / n
           for b in range(S)] for a in range(S)]

    def pick(lam_all, want, cap):
        tot = sum(lam_all)
        if tot <= _EPS:
            raise ValueError("funCA: a process carries no variation")
        # never form a component the data cannot support: a direction with
        # a numerically zero eigenvalue is noise, and the two arms would
        # disagree on that noise at around 1e-9
        rank = sum(1 for v in lam_all if v > _EPS * tot)
        cap = max(1, min(cap, rank))
        if want is not None:
            return max(1, min(int(want), cap))
        run, kk = 0.0, cap
        for j in range(len(lam_all)):
            run += lam_all[j] / tot
            if run >= 0.95:
                kk = j + 1
                break
        return max(1, min(kk, cap))

    _, _, lam_x_all = _fpca(Cx, wx, T)
    _, _, lam_y_all = _fpca(Cy, wy, S)
    pp = pick(lam_x_all, p, min(T, n - 1))
    qq = pick(lam_y_all, q, min(S, n - 1))
    lam_x, phi_x, _ = _fpca(Cx, wx, pp)
    lam_y, phi_y, _ = _fpca(Cy, wy, qq)

    xi = [[sum(Xc[i][t] * phi_x[j][t] * wx[t] for t in range(T))
           for j in range(pp)] for i in range(n)]
    eta = [[sum(Yc[i][t] * phi_y[j][t] * wy[t] for t in range(S))
            for j in range(qq)] for i in range(n)]

    def cross(A, B, na, nb):
        return [[sum(A[i][a] * B[i][b] for i in range(n)) / n
                 for b in range(nb)] for a in range(na)]

    Sxx = cross(xi, xi, pp, pp)
    Syy = cross(eta, eta, qq, qq)
    Sxy = cross(xi, eta, pp, qq)
    for a in range(pp):
        Sxx[a][a] += _EPS
    for a in range(qq):
        Syy[a][a] += _EPS

    Rx = _sym_inv_sqrt(Sxx)
    Ry = _sym_inv_sqrt(Syy)
    M = [[sum(Rx[a][u] * sum(Sxy[u][v] * Ry[v][b] for v in range(qq))
              for u in range(pp)) for b in range(qq)] for a in range(pp)]

    # canonical correlations are the singular values of M; get them from
    # the symmetric eigenproblem of M M' so no SVD routine is needed
    MMt = [[sum(M[a][c] * M[b][c] for c in range(qq)) for b in range(pp)]
           for a in range(pp)]
    d, V = k.jacobi(MMt)
    order = sorted(range(pp), key=lambda j: -d[j])
    r = min(pp, qq)
    corrs = [min(1.0, math.sqrt(max(d[order[j]], 0.0))) for j in range(r)]

    weights_x, weights_y, var_x, var_y = [], [], [], []
    for j in range(r):
        u = [V[a][order[j]] for a in range(pp)]
        a_coef = [sum(Rx[a][b] * u[b] for b in range(pp)) for a in range(pp)]
        wxj = [sum(a_coef[a] * phi_x[a][t] for a in range(pp))
               for t in range(T)]
        Mtu = [sum(M[a][b] * u[a] for a in range(pp)) for b in range(qq)]
        nrm = math.sqrt(sum(v * v for v in Mtu))
        v_coef = [v / nrm if nrm > _EPS else 0.0 for v in Mtu]
        b_coef = [sum(Ry[a][b] * v_coef[b] for b in range(qq))
                  for a in range(qq)]
        wyj = [sum(b_coef[a] * phi_y[a][t] for a in range(qq))
               for t in range(S)]
        top = max(range(T), key=lambda t: abs(wxj[t]))
        if wxj[top] < 0.0:
            wxj = [-v for v in wxj]
            wyj = [-v for v in wyj]
        weights_x.append(wxj)
        weights_y.append(wyj)
        var_x.append([sum(Xc[i][t] * wxj[t] * wx[t] for t in range(T))
                      for i in range(n)])
        var_y.append([sum(Yc[i][t] * wyj[t] * wy[t] for t in range(S))
                      for i in range(n)])

    tx, ty = sum(lam_x_all), sum(lam_y_all)
    return RichResult(payload={
        "estimate": corrs,
        "correlations": corrs,
        "weights_x": weights_x,
        "weights_y": weights_y,
        "variates_x": var_x,
        "variates_y": var_y,
        "p": int(pp),
        "q": int(qq),
        "explained_x": sum(lam_x[:pp]) / tx if tx > _EPS else 0.0,
        "explained_y": sum(lam_y[:qq]) / ty if ty > _EPS else 0.0,
        "eigenvalues_x": lam_x,
        "eigenvalues_y": lam_y,
        "n": n,
        "method": "functional canonical analysis restricted to the leading "
                  "functional principal components (He, Muller & Wang 2003)",
        "note": "unrestricted functional CCA is ill-posed -- the supremum "
                "is 1 for almost any pair of processes -- so the "
                "correlations are only interpretable against the "
                "truncation p, q that produced them",
    })


def cheatsheet():
    return ("funCA: functional_cca(X, Y, p, q) -> canonical correlations "
            "between two sets of curves, restricted to the leading FPCs "
            "(He, Muller & Wang 2003, J. Multivar. Anal. 85(1), 54-77)")
