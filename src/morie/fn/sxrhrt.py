# morie.fn -- function file (rootcoder007/morie)
r"""Is the same trait, in men and in women, the same trait?

Heritability is usually reported as one number for a trait, which
presumes that the genetic architecture does not differ by sex. It often
does -- in body composition, in blood lipids, in age at onset -- and
there are two distinct ways it can. The variance explained can differ,
so that genes matter more in one sex than the other; or the *same* genes
can matter differently, so that the genetic correlation across sexes is
below one. A single pooled estimate hides both.

Treating the sexes as two traits measured on disjoint individuals
recovers both. The covariance is block structured,

.. math:: V=\begin{pmatrix}\sigma^2_{g,m}K_{mm} &
          \sigma_{g,mf}K_{mf}\\ \sigma_{g,mf}K_{fm} &
          \sigma^2_{g,f}K_{ff}\end{pmatrix}
          + \begin{pmatrix}\sigma^2_{e,m}I & 0\\ 0 &
          \sigma^2_{e,f}I\end{pmatrix},

with no residual covariance term, because no individual contributes to
both blocks -- that structural zero is what makes the design
identifiable at all, and it is the reason the cross-sex genetic
covariance is estimable only from the *relatedness between* men and
women in the sample.

The parameters are fitted by restricted maximum likelihood over
:math:`(\sigma^2_{g,m},\sigma^2_{g,f},r_g,\sigma^2_{e,m},
\sigma^2_{e,f})`, cycling fixed-grid searches coordinate by
coordinate until the restricted likelihood stops moving. The
correlation is parameterised directly and bounded to
:math:`(-1,1)`, so the fitted covariance is positive semidefinite by
construction rather than by luck; a search over the covariance itself
wanders outside the admissible region and returns a correlation above
one, which is a number no data can support.

References
----------
Yang, J., Lee, S. H., Goddard, M. E. and Visscher, P. M. (2011) "GCTA:
a tool for genome-wide complex trait analysis", *American Journal of
Human Genetics* **88**(1), 76-82, doi:10.1016/j.ajhg.2010.11.011.

Lee, S. H., Yang, J., Goddard, M. E., Visscher, P. M. and Wray, N. R.
(2012) "Estimation of pleiotropy between complex diseases using
single-nucleotide polymorphism-derived genomic relationships and
restricted maximum likelihood", *Bioinformatics* **28**(19), 2540-2542,
doi:10.1093/bioinformatics/bts474. The bivariate REML this uses.

Yang, J., Bakshi, A., Zhu, Z. et al. (2015) "Genetic variance
estimation with imputed variants finds negligible missing heritability
for human height and body mass index", *Nature Genetics* **47**(10),
1114-1120, doi:10.1038/ng.3390.

Rawlik, K., Canela-Xandri, O. and Tenesa, A. (2016) "Evidence for sex-
specific genetic architectures across a spectrum of human complex
traits", *Genome Biology* **17**, 166, doi:10.1186/s13059-016-1025-x.

Patterson, H. D. and Thompson, R. (1971) "Recovery of inter-block
information when block sizes are unequal", *Biometrika* **58**(3),
545-554, doi:10.1093/biomet/58.3.545.

Falconer, D. S. and Mackay, T. F. C. (1996) *Introduction to
Quantitative Genetics*, 4th ed., Longman, Ch. 10 (the genetic
correlation and what a value below one means).
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sex_specific_h2"]

_EPS = 1e-12


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
                    return None
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


def _logdet(L):
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


def _reml(theta, y, X, Km, male):
    """Restricted log likelihood at ``(s2gm, s2gf, rg, s2em, s2ef)``."""
    s2gm, s2gf, rg, s2em, s2ef = theta
    if s2gm <= 0.0 or s2gf <= 0.0 or s2em <= 0.0 or s2ef <= 0.0:
        return None
    if abs(rg) >= 1.0:
        return None
    n = len(y)
    p = len(X[0])
    cov = rg * math.sqrt(s2gm * s2gf)
    V = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if male[i] and male[j]:
                V[i][j] = s2gm * Km[i][j]
            elif (not male[i]) and (not male[j]):
                V[i][j] = s2gf * Km[i][j]
            else:
                V[i][j] = cov * Km[i][j]
        V[i][i] += s2em if male[i] else s2ef
    L = _chol(V)
    if L is None:
        return None
    Viy = _solve(L, y)
    ViX = [_solve(L, [X[i][a] for i in range(n)]) for a in range(p)]
    XtViX = [[sum(X[i][a] * ViX[b][i] for i in range(n)) for b in range(p)]
             for a in range(p)]
    XtViy = [sum(X[i][a] * Viy[i] for i in range(n)) for a in range(p)]
    Lx = _chol(XtViX)
    if Lx is None:
        return None
    beta = _solve(Lx, XtViy)
    r = [y[i] - sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    Vir = _solve(L, r)
    quad = sum(r[i] * Vir[i] for i in range(n))
    ll = -0.5 * (_logdet(L) + _logdet(Lx) + quad)
    return ll, beta, L


def sex_specific_h2(y, sex, K, X=None, max_cycles=60, tol=1e-9,
                    male_label=1):
    r"""Bivariate REML for per-sex heritability and the cross-sex
    genetic correlation.

    Parameters
    ----------
    y : array-like, length ``n``
    sex : array-like, length ``n``
        Sex indicator. Entries equal to ``male_label`` are the first
        group; everything else is the second.
    K : array-like, shape ``(n, n)``
        Genomic relationship matrix over all individuals. The cross-sex
        block is what identifies the genetic correlation, so a matrix
        that is block diagonal by sex carries no information about it
        and the estimate will be flat.

    Returns
    -------
    RichResult
        ``h2_male``, ``h2_female``, ``rg``, the five variance
        parameters, and the restricted likelihood path.
    """
    yv = [float(v) for v in k.vec(y)]
    n = len(yv)
    if n == 0:
        raise ValueError("sxrhrt: no observations")
    sv = list(k.vec(sex))
    if len(sv) != n:
        raise ValueError("sxrhrt: %d phenotypes but %d sex labels"
                         % (n, len(sv)))
    male = [float(v) == float(male_label) for v in sv]
    nm = sum(1 for v in male if v)
    nf = n - nm
    if nm < 2 or nf < 2:
        raise ValueError("sxrhrt: %d in one sex and %d in the other -- a "
                         "variance cannot be estimated from fewer than two"
                         % (nm, nf))
    Km = [[float(v) for v in row] for row in k.mat(K)]
    if len(Km) != n or any(len(r) != n for r in Km):
        raise ValueError("sxrhrt: K must be %d by %d" % (n, n))
    asym = max(abs(Km[i][j] - Km[j][i]) for i in range(n) for j in range(n))
    if asym > 1e-8:
        raise ValueError("sxrhrt: K is not symmetric (largest asymmetry "
                         "%.3g)" % asym)
    # the cross-sex block is the only source of information about rg
    cross = max(abs(Km[i][j]) for i in range(n) for j in range(n)
                if male[i] != male[j])
    Xm = ([[1.0] for _ in range(n)] if X is None
          else [[float(v) for v in row] for row in k.mat(X)])
    p = len(Xm[0])

    mu = sum(yv) / n
    vy = sum((v - mu) ** 2 for v in yv) / max(n - 1, 1)
    theta = [max(vy / 2.0, 1e-6), max(vy / 2.0, 1e-6), 0.0,
             max(vy / 2.0, 1e-6), max(vy / 2.0, 1e-6)]
    lo = math.log(max(vy, 1e-8)) - 8.0
    hi = math.log(max(vy, 1e-8)) + 4.0

    def at(th):
        r = _reml(th, yv, Xm, Km, male)
        return -1e300 if r is None else r[0]

    path = [at(theta)]
    cycles = 0
    converged = False
    prev_theta = None
    for cycles in range(1, int(max_cycles) + 1):
        prev = path[-1]
        for idx in (0, 1, 3, 4):
            def f(logv, idx=idx):
                th = list(theta)
                th[idx] = math.exp(logv)
                return at(th)
            best = _gridmax(f, lo, hi)
            theta[idx] = math.exp(best)
        def fr(r):
            th = list(theta)
            th[2] = r
            return at(th)
        theta[2] = _gridmax(fr, -0.999, 0.999)
        cur = at(theta)
        path.append(cur)
        # convergence on the QUANTISED parameters, not on the likelihood --
        # see hibrid: a last-bit difference in the likelihood makes the two
        # languages stop one cycle apart and land on different estimates
        if prev_theta is not None and theta == prev_theta:
            converged = True
            break
        prev_theta = list(theta)

    res = _reml(theta, yv, Xm, Km, male)
    if res is None:
        raise ValueError("sxrhrt: the fitted covariance is not positive "
                         "definite -- the relationship matrix is probably "
                         "not a valid GRM")
    ll, beta, L = res
    s2gm, s2gf, rg, s2em, s2ef = theta
    h2m = s2gm / (s2gm + s2em)
    h2f = s2gf / (s2gf + s2ef)

    # a likelihood-ratio test against rg = 1: does the architecture differ?
    th1 = list(theta)
    th1[2] = 0.999999
    ll_rg1 = at(th1)
    lrt_rg1 = max(2.0 * (ll - ll_rg1), 0.0)
    # and against equal heritabilities
    def feq(logv):
        th = list(theta)
        th[0] = math.exp(logv)
        th[1] = math.exp(logv)
        return at(th)
    eq = math.exp(_gridmax(feq, lo, hi))
    th2 = [eq, eq, theta[2], theta[3], theta[4]]
    for _ in range(20):
        for idx in (3, 4):
            def f2(logv, idx=idx):
                th = list(th2)
                th[idx] = math.exp(logv)
                return at(th)
            th2[idx] = math.exp(_gridmax(f2, lo, hi))
    ll_eq = at(th2)
    lrt_equal = max(2.0 * (ll - ll_eq), 0.0)

    return RichResult(payload={
        "estimate": [h2m, h2f], "h2_male": h2m, "h2_female": h2f,
        "rg": rg,
        "sigma2_g_male": s2gm, "sigma2_g_female": s2gf,
        "sigma2_g_cross": rg * math.sqrt(s2gm * s2gf),
        "sigma2_e_male": s2em, "sigma2_e_female": s2ef,
        "coefficients": beta,
        "reml_loglik": ll, "reml_path": path,
        "lrt_rg_equals_one": lrt_rg1,
        "p_rg_equals_one": 0.5 * (1.0 - k.pnorm(math.sqrt(lrt_rg1))) * 2.0,
        "lrt_equal_h2": lrt_equal,
        "p_equal_h2": 2.0 * (1.0 - k.pnorm(math.sqrt(lrt_equal))),
        "n": n, "n_male": nm, "n_female": nf, "p": p,
        "max_cross_sex_relatedness": cross,
        "cycles": cycles, "converged": converged,
        "method": "bivariate REML treating the sexes as two traits on "
                  "disjoint individuals, with the genetic correlation "
                  "parameterised directly and bounded to (-1, 1) so the "
                  "fitted covariance is admissible by construction (Yang et "
                  "al. 2011 GCTA; Lee et al. 2012)",
        "note": "max_cross_sex_relatedness is the diagnostic to read "
                "first: the cross-sex block of K is the only thing that "
                "identifies rg, and if it is near zero the correlation is "
                "not estimable however tight the likelihood looks",
    })


def cheatsheet():
    return ("sxrhrt: sex_specific_h2(y, sex, K) -> per-sex heritability and "
            "the cross-sex genetic correlation by bivariate REML (Yang et "
            "al. 2011 GCTA; Lee et al. 2012)")
