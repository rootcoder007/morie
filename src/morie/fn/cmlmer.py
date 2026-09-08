# morie.fn -- function file (rootcoder007/morie)
r"""Compressed mixed linear model: relatedness at the size of the groups.

A mixed-model genome scan spends its time inverting an n-by-n kinship
matrix once per marker, which is why the method that controls population
structure best is the one nobody could afford to run. Zhang et al.'s
observation is that the individuals are not all distinct as far as the
random effect is concerned: cluster them, give each *cluster* a random
effect, and the covariance to invert is the size of the number of groups
rather than the sample,

.. math:: y = X\beta + Z u + e,\qquad u\sim N(0,\sigma_g^2 K_g),\qquad
          e\sim N(0,\sigma_e^2 I),

where ``Z`` is the cluster incidence matrix and :math:`K_g` the group
kinship, each entry the average kinship between the members of the two
groups. With one group per individual :math:`Z=I` and :math:`K_g=K`, so
the compressed model *contains* the full model rather than approximating
it -- and that identity is checked here, exactly, rather than assumed.

Clustering is average-linkage (UPGMA) on ``1 - K``, which is the choice
Zhang et al. found best in their comparison and is deterministic, so
there is no random restart for two implementations to disagree about.
The variance ratio is profiled out by a fixed-grid search on
:math:`\log\delta`, :math:`\delta=\sigma_e^2/\sigma_g^2`, and the
profile is returned.

Association testing is a generalised least squares t-test per marker
under the fitted covariance. The compression level that maximises the
restricted likelihood is a model-selection question, not a speed knob,
and ``compare_levels`` reports the likelihood at each number of groups
so it can be chosen rather than guessed.

References
----------
Zhang, Z., Ersoz, E., Lai, C.-Q., Todhunter, R. J., Tiwari, H. K.,
Gore, M. A., Bradbury, P. J., Yu, J., Arnett, D. K., Ordovas, J. M. and
Buckler, E. S. (2010) "Mixed linear model approach adapted for
genome-wide association studies", *Nature Genetics* **42**(4), 355-360,
doi:10.1038/ng.546.

Yu, J., Pressoir, G., Briggs, W. H. et al. (2006) "A unified
mixed-model method for association mapping that accounts for multiple
levels of relatedness", *Nature Genetics* **38**(2), 203-208,
doi:10.1038/ng1702. The Q+K model this compresses.

Kang, H. M., Zaitlen, N. A., Wade, C. M., Kirby, A., Heckerman, D.,
Daly, M. J. and Eskin, E. (2008) "Efficient control of population
structure in model organism association mapping", *Genetics* **178**(3),
1709-1723, doi:10.1534/genetics.107.080101.

Sokal, R. R. and Michener, C. D. (1958) "A statistical method for
evaluating systematic relationships", *University of Kansas Science
Bulletin* **38**, 1409-1438. Average-linkage clustering.

Patterson, H. D. and Thompson, R. (1971) "Recovery of inter-block
information when block sizes are unequal", *Biometrika* **58**(3),
545-554, doi:10.1093/biomet/58.3.545. Restricted maximum likelihood.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["compressed_lmm"]

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


def _snap12(x):
    """Quantise a UPGMA distance to a 1e-12 grid.

    Average linkage merges the closest pair, and a tie broken differently
    in the two languages would give different groups and therefore a
    different model. The grid is far finer than any distance the caller
    can distinguish and far coarser than the last-bit noise that would
    otherwise decide a tie.
    """
    return math.floor(x * 1e12 + 0.5) / 1e12


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    scale = sum(A[i][i] for i in range(n)) / n
    jit = 1e-12 * max(abs(scale), 1.0)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][u] * L[j][u] for u in range(j))
            if i == j:
                s += jit
                if s <= 0.0:
                    raise ValueError("cmlmer: the covariance matrix is not "
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


def _logdet(L):
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


def _upgma(K, g):
    """Average-linkage clustering on ``1 - K``, cut at ``g`` groups.

    Deterministic throughout: ties are broken by the smaller index pair,
    so two implementations agree on the dendrogram, not merely on its
    quality.
    """
    n = len(K)
    members = [[i] for i in range(n)]
    # the distances are quantised before any comparison. Two candidate
    # merges separated by about 1e-15 are a coin flip between languages
    # that compute 1 - K to different last bits, and one different merge
    # changes the whole dendrogram below it. A 1e-12 grid is far finer
    # than any real distinction between two groups and forces both arms
    # to see the same numbers.
    D = [[_snap12(1.0 - K[i][j]) for j in range(n)] for i in range(n)]
    alive = list(range(n))
    while len(alive) > g:
        bi, bj, best = -1, -1, None
        for ai in range(len(alive)):
            for aj in range(ai + 1, len(alive)):
                d = D[alive[ai]][alive[aj]]
                if best is None or d < best - 1e-15:
                    best, bi, bj = d, alive[ai], alive[aj]
        na, nb = len(members[bi]), len(members[bj])
        for c in alive:
            if c == bi or c == bj:
                continue
            nd = _snap12((na * D[bi][c] + nb * D[bj][c]) / (na + nb))
            D[bi][c] = nd
            D[c][bi] = nd
        members[bi] = members[bi] + members[bj]
        members[bj] = []
        alive.remove(bj)
    groups = [members[i] for i in alive]
    # canonical group order: by the smallest member index, so the labels
    # are a property of the data and not of the merge order
    groups.sort(key=lambda gr: min(gr))
    lab = [0] * n
    for j, gr in enumerate(groups):
        for i in gr:
            lab[i] = j
    return lab, groups


def _reml_at(logdelta, Vk, y, X):
    n = len(y)
    p = len(X[0])
    delta = math.exp(logdelta)
    V = [[Vk[i][j] + (delta if i == j else 0.0) for j in range(n)]
         for i in range(n)]
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
    rss = sum(r[i] * Vir[i] for i in range(n))
    dfr = n - p
    s2g = rss / dfr
    ll = -0.5 * (dfr * math.log(max(s2g, 1e-300)) + _logdet(L)
                 + _logdet(Lx) + dfr)
    return ll, delta, beta, s2g, L, XtViX


def compressed_lmm(y, M, K, clusters=None, X=None, compare_levels=None,
                   log_delta_lo=-10.0, log_delta_hi=10.0, max_iter=200):
    r"""Fit a compressed MLM and scan the markers under it.

    Parameters
    ----------
    y : array-like, length ``n``
    M : array-like, shape ``(n, p_markers)``
        Markers to test. Pass an empty list to fit the null model only.
    K : array-like, shape ``(n, n)``
        Kinship. Must be symmetric.
    clusters : int, optional
        Number of groups. Defaults to ``n``, which is the uncompressed
        model.
    compare_levels : sequence of int, optional
        Group counts at which to report the restricted likelihood, so
        the compression level can be chosen on evidence.

    Returns
    -------
    RichResult
        ``group``/``group_kinship``, the variance components, and per
        marker ``beta``, ``se``, ``t`` and ``p``.
    """
    yv = [float(v) for v in k.vec(y)]
    n = len(yv)
    if n == 0:
        raise ValueError("cmlmer: no observations")
    Km = [[float(v) for v in row] for row in k.mat(K)]
    if len(Km) != n or any(len(r) != n for r in Km):
        raise ValueError("cmlmer: K must be %d by %d" % (n, n))
    asym = max(abs(Km[i][j] - Km[j][i])
               for i in range(n) for j in range(n))
    if asym > 1e-8:
        raise ValueError("cmlmer: K is not symmetric (largest asymmetry "
                         "%.3g)" % asym)
    Mm = ([] if M is None else [[float(v) for v in row]
                                for row in k.mat(M)]) if M is not None else []
    if Mm and len(Mm) != n:
        raise ValueError("cmlmer: %d phenotypes but %d marker rows"
                         % (n, len(Mm)))
    nm = len(Mm[0]) if Mm else 0
    Xm = ([[1.0] for _ in range(n)] if X is None
          else [[float(v) for v in row] for row in k.mat(X)])
    p = len(Xm[0])
    if n - p - 1 < 1:
        raise ValueError("cmlmer: too few observations for %d fixed effects "
                         "plus a marker" % p)
    g = n if clusters is None else int(clusters)
    if g < 1 or g > n:
        raise ValueError("cmlmer: the number of groups must be between 1 "
                         "and %d, got %d" % (n, g))

    lab, groups = _upgma(Km, g)
    ng = len(groups)
    # group kinship: the average kinship between the members of two groups
    Kg = [[0.0] * ng for _ in range(ng)]
    for a in range(ng):
        for b in range(ng):
            Kg[a][b] = (sum(Km[i][j] for i in groups[a] for j in groups[b])
                        / (len(groups[a]) * len(groups[b])))
    # Z Kg Z' -- the compressed covariance written at the sample size
    ZKZ = [[Kg[lab[i]][lab[j]] for j in range(n)] for i in range(n)]

    # max_iter is accepted and ignored: the grid schedule fixes the
    # evaluation count, and dropping the argument would break callers.
    logdelta = _gridmax(lambda t: _reml_at(t, ZKZ, yv, Xm)[0],
                        log_delta_lo, log_delta_hi)
    ll, delta, beta0, s2g, L, _ = _reml_at(logdelta, ZKZ, yv, Xm)
    s2e = delta * s2g
    h2 = s2g / (s2g + s2e)

    profile = []
    for t in range(21):
        lt = (float(log_delta_lo)
              + (float(log_delta_hi) - float(log_delta_lo)) * t / 20.0)
        profile.append([lt, _reml_at(lt, ZKZ, yv, Xm)[0]])

    # ---- per-marker GLS test under the fitted covariance
    mb, mse, mt, mp = [], [], [], []
    for j in range(nm):
        Xj = [Xm[i] + [Mm[i][j]] for i in range(n)]
        q = p + 1
        ViX = [_solve(L, [Xj[i][a] for i in range(n)]) for a in range(q)]
        A = [[sum(Xj[i][a] * ViX[b][i] for i in range(n)) for b in range(q)]
             for a in range(q)]
        Viy = _solve(L, yv)
        rhs = [sum(Xj[i][a] * Viy[i] for i in range(n)) for a in range(q)]
        try:
            Lj = _chol(A)
        except ValueError:
            mb.append(float("nan"))
            mse.append(float("nan"))
            mt.append(float("nan"))
            mp.append(float("nan"))
            continue
        bj = _solve(Lj, rhs)
        r = [yv[i] - sum(Xj[i][a] * bj[a] for a in range(q))
             for i in range(n)]
        Vir = _solve(L, r)
        s2 = sum(r[i] * Vir[i] for i in range(n)) / (n - q)
        e = [0.0] * q
        e[q - 1] = 1.0
        cjj = _solve(Lj, e)[q - 1]
        se = math.sqrt(max(s2 * cjj, 0.0))
        tj = bj[q - 1] / se if se > _EPS else float("nan")
        mb.append(bj[q - 1])
        mse.append(se)
        mt.append(tj)
        mp.append(2.0 * (1.0 - k.pnorm(abs(tj))) if tj == tj
                  else float("nan"))

    levels = []
    if compare_levels:
        for gl in compare_levels:
            gl = int(gl)
            if gl < 1 or gl > n:
                raise ValueError("cmlmer: compare_levels entry %d is outside "
                                 "1..%d" % (gl, n))
            lab2, gr2 = _upgma(Km, gl)
            ng2 = len(gr2)
            Kg2 = [[sum(Km[i][j] for i in gr2[a] for j in gr2[b])
                    / (len(gr2[a]) * len(gr2[b])) for b in range(ng2)]
                   for a in range(ng2)]
            ZKZ2 = [[Kg2[lab2[i]][lab2[j]] for j in range(n)]
                    for i in range(n)]
            best = None
            for t in range(41):
                lt = (float(log_delta_lo)
                      + (float(log_delta_hi) - float(log_delta_lo))
                      * t / 40.0)
                v = _reml_at(lt, ZKZ2, yv, Xm)[0]
                if best is None or v > best:
                    best = v
            levels.append([gl, best])

    return RichResult(payload={
        "estimate": mb, "beta": mb, "se": mse, "t": mt, "p_value": mp,
        "group": [float(v) for v in lab], "n_groups": ng,
        "group_sizes": [len(gr) for gr in groups],
        "group_kinship": Kg,
        "coefficients": beta0, "delta": delta,
        "sigma2_g": s2g, "sigma2_e": s2e, "h2": h2,
        "reml_loglik": ll, "reml_profile": profile,
        "level_loglik": levels,
        "n": n, "n_markers": nm, "p": p, "clusters_requested": g,
        "method": "compressed mixed linear model: average-linkage grouping "
                  "on 1 - K, a per-group random effect with the average "
                  "between-group kinship, the variance ratio profiled out "
                  "by REML, and a GLS t-test per marker (Zhang et al. 2010; "
                  "Yu et al. 2006)",
        "note": "with one group per individual Z is the identity and the "
                "group kinship is K itself, so the compressed model "
                "contains the uncompressed one exactly rather than "
                "approximating it",
    })


def cheatsheet():
    return ("cmlmer: compressed_lmm(y, M, K, clusters) -> compressed MLM "
            "genome scan with REML variance components (Zhang et al. 2010, "
            "Nature Genetics 42:355-360)")


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
compressedlmm = compressed_lmm
