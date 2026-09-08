# morie.fn -- function file (rootcoder007/morie)
r"""Hybrid prediction: what each parent contributes, and what only the
cross does.

A breeding programme cannot field every cross. With 200 lines there are
nearly 20,000 possible single crosses and a trial holds a few hundred,
so the value of the untested crosses has to be predicted from the tested
ones. Sprague and Tatum's decomposition is what makes that possible: the
performance of a cross splits into what each parent brings to every
cross it enters -- general combining ability -- and what is specific to
this pair,

.. math:: y_{ij} = \mu + g_i + g_j + s_{ij} + e_{ij}.

The genomic version replaces the pedigree with marker kernels. GCA is
additive and shared, so its kernel is the sum of the two parental
relationship matrices; SCA is the interaction, and Technow et al. show
its kernel is their Hadamard product,

.. math:: K_{\mathrm{GCA}} = \tfrac1m(P_1P_1' + P_2P_2'),\qquad
          K_{\mathrm{SCA}} = \tfrac1{m^2}(P_1P_1')\odot(P_2P_2').

The variance components are fitted by profiling the residual variance
out of the restricted likelihood and maximising over the two ratios
:math:`\lambda_a=\sigma_e^2/\sigma_a^2` and
:math:`\lambda_s=\sigma_e^2/\sigma_s^2` with fixed-grid searches
cycled to convergence. Each coordinate maximisation can only raise the
restricted likelihood, so the path is monotone and is returned so that
is checkable rather than asserted. EM-REML gives the same answer and was
tried first; it was still 4e-4 short of the optimum after 300 sweeps at
n = 36, which is a lot of arithmetic to mirror in two languages for a
worse answer. The ratio
:math:`\sigma_s^2/(\sigma_a^2+\sigma_s^2)` is the quantity a breeder
actually asks for: it says whether choosing good parents is enough or
whether the specific combination has to be tested.

Fixing ``sigma2_sca = 0`` reduces the model exactly to additive GBLUP on
the GCA kernel, and that reduction is checked rather than asserted.

References
----------
Sprague, G. F. and Tatum, L. A. (1942) "General versus specific
combining ability in single crosses of corn", *Journal of the American
Society of Agronomy* **34**(10), 923-932,
doi:10.2134/agronj1942.00021962003400100008x.

Technow, F., Riedelsheimer, C., Schrag, T. A. and Melchinger, A. E.
(2012) "Genomic prediction of hybrid performance in maize with models
incorporating dominance and population specific marker effects",
*Theoretical and Applied Genetics* **125**(6), 1181-1194,
doi:10.1007/s00122-012-1905-8.

Technow, F., Schrag, T. A., Schipprack, W., Bauer, E., Simianer, H. and
Melchinger, A. E. (2014) "Genome properties and prospects of genomic
prediction of hybrid performance in a breeding program of maize",
*Genetics* **197**(4), 1343-1355, doi:10.1534/genetics.114.165860.

Bernardo, R. (1994) "Prediction of maize single-cross performance using
RFLPs and information from related hybrids", *Crop Science* **34**(1),
20-25, doi:10.2135/cropsci1994.0011183X003400010003x.

Patterson, H. D. and Thompson, R. (1971) "Recovery of inter-block
information when block sizes are unequal", *Biometrika* **58**(3),
545-554, doi:10.1093/biomet/58.3.545. Restricted maximum likelihood.

Searle, S. R., Casella, G. and McCulloch, C. E. (1992) *Variance
Components*, Wiley, Ch. 6 and 8 (profiling the residual variance out of
the restricted likelihood), doi:10.1002/9780470316856.

VanRaden, P. M. (2008) "Efficient methods to compute genomic
predictions", *Journal of Dairy Science* **91**(11), 4414-4423,
doi:10.3168/jds.2007-0980.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["hibrid_prediction"]

_EPS = 1e-12
_LO = -14.0
_HI = 14.0


def _gridmax(f, lo, hi, points=201, stages=4):
    r"""Maximise ``f`` over ``[lo, hi]`` by a staged fixed-grid argmax.

    A golden-section search is PATH-DEPENDENT. Each arm walks its own
    sequence of brackets, and near a flat maximum the ``fc > fd`` branch is
    decided by the last bits of two nearly equal likelihoods, so the two
    languages take different paths and land on different answers. Quantising
    the result afterwards hides that only when the answer does not fall near
    a cell boundary, which is a coincidence rather than a guarantee: the
    measured failure was two arms landing on ADJACENT points of a 1e-6 grid.

    Here both arms evaluate the SAME list of points -- ``a + i * step`` is
    the same double in both languages -- and take the argmax BY INDEX, ties
    to the lowest index. The winning index is therefore the same by
    construction, and the value returned is an exact grid point rather than a
    bracket midpoint, so the two arms return bit-identical doubles.

    Refinement stops while adjacent grid values still differ by far more than
    floating-point noise. Going finer would push the comparison back below
    the noise floor and reintroduce exactly the disagreement this exists to
    remove. It would also be false precision: a REML optimum this flat is not
    located to better than the square root of machine epsilon by any method,
    and the resolution reached here is already orders of magnitude finer than
    the statistical precision of the estimate.
    """
    a, b = float(lo), float(hi)
    npt = int(points)
    last = int(stages) - 1
    for s in range(int(stages)):
        step = (b - a) / (npt - 1)
        vals = [f(a + i * step) for i in range(npt)]
        best = 0
        for i in range(1, npt):
            if vals[i] > vals[best]:
                best = i
        if s == last:
            return a + best * step
        lo_i = best - 1 if best > 0 else 0
        hi_i = best + 1 if best < npt - 1 else npt - 1
        a, b = a + lo_i * step, a + hi_i * step
        npt = 21
    return a


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    scale = sum(A[i][i] for i in range(n)) / n
    jit = 1e-11 * max(abs(scale), 1.0)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][u] * L[j][u] for u in range(j))
            if i == j:
                s += jit
                if s <= 0.0:
                    raise ValueError("hibrid: the covariance matrix is not "
                                     "positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _solve(L, b):
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        z[i] = (b[i] - sum(L[i][u] * z[u] for u in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(L[u][i] * x[u] for u in range(i + 1, n))) / L[i][i]
    return x


def _inv_from_chol(L):
    n = len(L)
    Vi = [[0.0] * n for _ in range(n)]
    for a in range(n):
        e = [0.0] * n
        e[a] = 1.0
        col = _solve(L, e)
        for b in range(n):
            Vi[b][a] = col[b]
    return Vi


def _logdet(L):
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


def _matmul(A, B):
    n, q, m = len(A), len(B), len(B[0])
    return [[sum(A[i][u] * B[u][j] for u in range(q)) for j in range(m)]
            for i in range(n)]


def _reml_at(la, ls, Kg, Ks, y, X):
    r"""Restricted log likelihood at the two ratios, sigma_e^2 profiled out.

    ``V = Kg/la + Ks/ls + I`` up to the factor sigma_e^2, which has the
    closed-form restricted maximiser ``y'Py / (n - p)`` and so never
    needs searching over.
    """
    n = len(y)
    p = len(X[0])
    V = [[Kg[i][j] / la + Ks[i][j] / ls + (1.0 if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    L = _chol(V)
    Viy = _solve(L, y)
    ViX = [_solve(L, [X[i][a] for i in range(n)]) for a in range(p)]
    XtViX = [[sum(X[i][a] * ViX[b][i] for i in range(n)) for b in range(p)]
             for a in range(p)]
    XtViy = [sum(X[i][a] * Viy[i] for i in range(n)) for a in range(p)]
    Lx = _chol(XtViX)
    beta = _solve(Lx, XtViy)
    r = [y[i] - sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    Vir = _solve(L, r)
    dfr = n - p
    s2e = sum(r[i] * Vir[i] for i in range(n)) / dfr
    ll = -0.5 * (dfr * math.log(max(s2e, 1e-300)) + _logdet(L)
                 + _logdet(Lx) + dfr)
    return ll, beta, s2e, L


def hibrid_prediction(y, p1_geno, p2_geno, sigma2_sca=None, X=None,
                      p1_new=None, p2_new=None, max_iter=300, tol=1e-10):
    r"""Genomic prediction of hybrid performance from GCA and SCA.

    Parameters
    ----------
    y : array-like, length ``n``
        Hybrid performance, one entry per tested cross.
    p1_geno, p2_geno : array-like, shape ``(n, m)``
        Marker genotypes of the two parents of each cross.
    sigma2_sca : float, optional
        Fix the SCA variance instead of estimating it. ``0`` reduces the
        model to additive GBLUP on the GCA kernel exactly.
    p1_new, p2_new : array-like, optional
        Parents of untested crosses to predict.

    Returns
    -------
    RichResult
        ``gca_effect`` and ``sca_effect`` per tested cross, the variance
        components, ``sca_share``, and predictions for the untested
        crosses.
    """
    yv = [float(v) for v in k.vec(y)]
    P1 = [[float(v) for v in row] for row in k.mat(p1_geno)]
    P2 = [[float(v) for v in row] for row in k.mat(p2_geno)]
    n = len(yv)
    if n == 0:
        raise ValueError("hibrid: no crosses")
    if len(P1) != n or len(P2) != n:
        raise ValueError("hibrid: %d phenotypes but %d and %d parental "
                         "genotype rows" % (n, len(P1), len(P2)))
    m = len(P1[0])
    if any(len(r) != m for r in P1) or any(len(r) != m for r in P2):
        raise ValueError("hibrid: both parents must be typed at the same "
                         "%d markers" % m)
    Xm = ([[1.0] for _ in range(n)] if X is None
          else [[float(v) for v in row] for row in k.mat(X)])
    p = len(Xm[0])
    if n - p < 2:
        raise ValueError("hibrid: %d crosses and %d fixed effects leave too "
                         "little information for two variance components"
                         % (n, p))

    G1 = [[sum(P1[i][a] * P1[j][a] for a in range(m)) / m for j in range(n)]
          for i in range(n)]
    G2 = [[sum(P2[i][a] * P2[j][a] for a in range(m)) / m for j in range(n)]
          for i in range(n)]
    Kg = [[G1[i][j] + G2[i][j] for j in range(n)] for i in range(n)]
    Ks = [[G1[i][j] * G2[i][j] for j in range(n)] for i in range(n)]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    fixed_sca = sigma2_sca is not None
    path = []

    if fixed_sca and float(sigma2_sca) <= _EPS:
        # no SCA at all: the model IS additive GBLUP on the GCA kernel, and
        # is fitted as exactly that -- one ratio, nothing else moving
        Kz = [[0.0] * n for _ in range(n)]

        def f1(l):
            return _reml_at(math.exp(l), 1e300, Kg, Kz, yv, Xm)[0]
        la = math.exp(_gridmax(f1, _LO, _HI))
        path.append(f1(math.log(la)))
        ll, beta, s2e, L = _reml_at(la, 1e300, Kg, Kz, yv, Xm)
        s2a, s2s = s2e / la, 0.0
        it, conv = 1, True
        Ks_used = Kz
    else:
        Ks_used = Ks
        la, ls = 1.0, 1.0
        it, conv = 0, False
        path.append(_reml_at(la, ls, Kg, Ks_used, yv, Xm)[0])
        prev_la = prev_ls = None
        for it in range(1, int(max_iter) + 1):
            prev = path[-1]

            def fa(l):
                return _reml_at(math.exp(l), ls, Kg, Ks_used, yv, Xm)[0]
            la = math.exp(_gridmax(fa, _LO, _HI))
            if fixed_sca:
                # the SCA variance is pinned, so its ratio follows the
                # residual variance rather than being searched over
                s2e_now = _reml_at(la, ls, Kg, Ks_used, yv, Xm)[2]
                ls = s2e_now / max(float(sigma2_sca), 1e-300)
            else:
                def fs(l):
                    return _reml_at(la, math.exp(l), Kg, Ks_used, yv, Xm)[0]
                ls = math.exp(_gridmax(fs, _LO, _HI))
            cur = _reml_at(la, ls, Kg, Ks_used, yv, Xm)[0]
            path.append(cur)
            # convergence on the QUANTISED ratios, not on the likelihood:
            # the likelihood test trips one iteration apart in the two
            # languages because cur and prev differ in their last bits,
            # and the two arms then stop at different parameter values.
            if prev_la is not None and la == prev_la and ls == prev_ls:
                conv = True
                break
            prev_la, prev_ls = la, ls
        ll, beta, s2e, L = _reml_at(la, ls, Kg, Ks_used, yv, Xm)
        s2a = s2e / la
        s2s = float(sigma2_sca) if fixed_sca else s2e / ls

    r = [yv[i] - sum(Xm[i][a] * beta[a] for a in range(p)) for i in range(n)]
    w = _solve(L, r)
    gca = [s2a * sum(Kg[i][j] * w[j] for j in range(n)) for i in range(n)]
    sca = [s2s * sum(Ks_used[i][j] * w[j] for j in range(n))
           for i in range(n)]
    fitted = [sum(Xm[i][a] * beta[a] for a in range(p)) + gca[i] + sca[i]
              for i in range(n)]

    tot = s2a + s2s + s2e
    pred_new = None
    if p1_new is not None and p2_new is not None:
        Q1 = [[float(v) for v in row] for row in k.mat(p1_new)]
        Q2 = [[float(v) for v in row] for row in k.mat(p2_new)]
        if len(Q1) != len(Q2):
            raise ValueError("hibrid: p1_new and p2_new must describe the "
                             "same crosses")
        if any(len(rw) != m for rw in Q1) or any(len(rw) != m for rw in Q2):
            raise ValueError("hibrid: new parents must be typed at the same "
                             "%d markers" % m)
        pred_new = []
        for u in range(len(Q1)):
            c1 = [sum(Q1[u][a] * P1[j][a] for a in range(m)) / m
                  for j in range(n)]
            c2 = [sum(Q2[u][a] * P2[j][a] for a in range(m)) / m
                  for j in range(n)]
            cg = [c1[j] + c2[j] for j in range(n)]
            cs = [c1[j] * c2[j] for j in range(n)]
            # an untested cross carries no covariate row, so the fixed part
            # is the intercept alone -- beta[0] by construction of X
            pred_new.append(beta[0]
                            + s2a * sum(cg[j] * w[j] for j in range(n))
                            + s2s * sum(cs[j] * w[j] for j in range(n)))

    return RichResult(payload={
        "estimate": fitted, "fitted": fitted,
        "gca_effect": gca, "sca_effect": sca,
        "coefficients": beta,
        "sigma2_gca": s2a, "sigma2_sca": s2s, "sigma2_e": s2e,
        "sca_share": (s2s / (s2a + s2s)) if s2a + s2s > _EPS else 0.0,
        "h2": (s2a + s2s) / tot if tot > _EPS else float("nan"),
        "gca_kernel": Kg, "sca_kernel": Ks,
        "reml_path": path, "reml_loglik": ll, "iterations": it,
        "converged": conv,
        "sca_fixed": fixed_sca,
        "prediction_new": pred_new,
        "residuals": [yv[i] - fitted[i] for i in range(n)],
        "n": n, "m": m, "p": p,
        "method": "genomic hybrid prediction: additive GCA kernel from the "
                  "sum of the parental relationship matrices, SCA kernel "
                  "from their Hadamard product, variance components by "
                  "profiled REML (Sprague & Tatum 1942; Technow et al. 2012, "
                  "2014)",
        "note": "sca_share is the fraction of genetic variance that only "
                "the specific combination explains -- near zero means "
                "choosing good parents is enough, and large means the "
                "cross itself has to be tested; fixing sigma2_sca at 0 "
                "reduces this exactly to additive GBLUP. Separating the "
                "two needs a factorial design with many parents: with p "
                "lines per pool the GCA kernel already spans about 2p - 1 "
                "dimensions, so at 6 by 6 it takes eleven of the "
                "thirty-six observations and no estimator can tell the "
                "remaining interaction from residual noise. Compare "
                "reml_loglik against the same fit with sigma2_sca = 0 "
                "before reporting an SCA variance.",
    })


def cheatsheet():
    return ("hibrid: hibrid_prediction(y, p1_geno, p2_geno) -> GCA and SCA "
            "variance components and hybrid predictions from marker kernels "
            "(Sprague & Tatum 1942; Technow et al. 2014, Genetics 197:1343)")
