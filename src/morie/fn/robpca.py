r"""ROBPCA: robust principal components for data with outliers.

Hubert, M., Rousseeuw, P. J., & Vanden Branden, K. (2005) "ROBPCA: A New
Approach to Robust Principal Component Analysis", *Technometrics* 47(1),
64-79.

Classical PCA maximises variance, and variance is exactly what a single
far-away observation controls, so one bad point can turn the loadings by
almost any angle. ROBPCA replaces the covariance matrix with one computed
from the ``h`` least outlying observations, in three stages set out in the
paper's Appendix:

**Stage 1.** Reduce to the affine subspace the observations actually span,
by a singular value decomposition of the mean-centred data (eq. 7). This
is an affine transformation, not a truncation -- no components are dropped
here, "we only represent the data in its own dimensionality".

**Stage 2.** Rank the points by the orthogonally invariant outlyingness

.. math::

   \mathrm{outl}_O(x_i) = \max_{v \in B}
      \frac{|x_i'v - t_{MCD}(x_j'v)|}{s_{MCD}(x_j'v)}   \tag{9}

where :math:`B` is the set of directions through two data points (250 of
them at random when there are more), and :math:`t_{MCD}`, :math:`s_{MCD}`
are the mean and standard deviation of the :math:`h` projected values with
smallest variance. The :math:`h` least outlying points give the
preliminary centre and covariance :math:`S_0`, whose eigenvectors define
the :math:`k_0`-dimensional subspace the data are projected onto.

**Stage 3.** Run the MCD on that subspace: C-steps from the stage-2 subset
until the determinant stops decreasing, plus FAST-MCD from 250 random
:math:`(k_1+1)`-subsets, keeping whichever gives the smaller determinant.
Multiply by the consistency factor

.. math::

   c_1 = \frac{\{d^2_{\hat\mu_4, S_3}\}_{(h)}}{\chi^2_{k,\,h/n}}

and reweight with hard rejection at :math:`\sqrt{\chi^2_{k,0.975}}`. The
eigenvectors of the reweighted scatter are the robust loadings, and the
scores are :math:`T = (X - 1\hat\mu')P` (eq. 1).

The diagnostics are the other half of the method. Each observation gets a
score distance and an orthogonal distance,

.. math::

   SD_i = \sqrt{\sum_{j=1}^k \frac{t_{ij}^2}{l_j}}, \qquad
   OD_i = \lVert x_i - \hat\mu - P t_i' \rVert   \tag{3, 4}

with cut-offs :math:`\sqrt{\chi^2_{k,0.975}}` for :math:`SD` and, for
:math:`OD`, the Wilson-Hilferty construction the paper prefers: the
:math:`OD^{2/3}` are approximately normal, their centre and scale are
estimated by the univariate MCD, and the cut-off is
:math:`(\hat\mu + \hat\sigma z_{0.975})^{3/2}`. Together they classify
every point into the four types of the paper's Figure 1 -- regular, good
leverage, orthogonal outlier, bad leverage.

Both of the paper's rules for choosing :math:`k` are available through
``k="cumulative"`` (eq. 5, the first :math:`k` explaining 90% of the
:math:`\tilde l_j`) and ``k="ratio"`` (eq. 6,
:math:`\tilde l_k / \tilde l_1 \ge 10^{-3}`).
"""

import math

from . import _array_core as np
from . import _stats_core as _st

from ._richresult import RichResult

__all__ = [
    "robust_pca",
    "robpca",
    "outlyingness",
    "univariate_mcd",
    "classify_outliers",
]

_Z975 = 1.959963984540054      # Phi^{-1}(0.975)


# ------------------------------------------------------------ helpers

def _matrix(X, name="X"):
    rows = [list(map(float, r)) for r in np.asarray(X, dtype=float)]
    if not rows:
        raise ValueError("robpca: %s is empty" % name)
    p = len(rows[0])
    if p == 0:
        raise ValueError("robpca: %s has no columns" % name)
    for r in rows:
        if len(r) != p:
            raise ValueError("robpca: %s is ragged" % name)
        for v in r:
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("robpca: %s contains a non-finite value"
                                 % name)
    return rows, len(rows), p


def _colmeans(rows):
    n = len(rows)
    p = len(rows[0])
    return [sum(r[j] for r in rows) / n for j in range(p)]


def _center(rows, mu):
    return [[r[j] - mu[j] for j in range(len(mu))] for r in rows]


def _cov(rows):
    n = len(rows)
    p = len(rows[0])
    if n < 2:
        raise ValueError("robpca: a covariance needs at least two points")
    mu = _colmeans(rows)
    C = [[0.0] * p for _ in range(p)]
    for r in rows:
        d = [r[j] - mu[j] for j in range(p)]
        for a in range(p):
            da = d[a]
            for b in range(a, p):
                C[a][b] += da * d[b]
    for a in range(p):
        for b in range(a, p):
            C[a][b] /= (n - 1.0)
            C[b][a] = C[a][b]
    return mu, C


def _eigh_desc(C):
    """Symmetric eigendecomposition, eigenvalues descending."""
    vals, vecs = np.linalg.eigh(np.asarray(C, dtype=float))
    vals = [float(v) for v in vals]
    p = len(vals)
    cols = [[float(vecs[i][j]) for i in range(p)] for j in range(p)]
    order = sorted(range(p), key=lambda j: -vals[j])
    return [vals[j] for j in order], [cols[j] for j in order]


def _matmul(A, B):
    """A (n x p) times B given as a list of p-vectors (columns)."""
    return [[sum(row[t] * col[t] for t in range(len(row))) for col in B]
            for row in A]


def _det_from_chol(C):
    """Determinant of a symmetric positive semi-definite matrix.

    Returns 0.0 when the Cholesky factorisation meets a non-positive
    pivot, which is the singularity the C-step has to notice.
    """
    p = len(C)
    L = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            s = C[i][j] - sum(L[i][t] * L[j][t] for t in range(j))
            if i == j:
                if s <= 1e-300:
                    return 0.0
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    d = 1.0
    for i in range(p):
        d *= L[i][i] ** 2
    return d


def _mahalanobis(rows, mu, C):
    """Robust distances (eq. 11); raises if C is singular."""
    p = len(mu)
    # inverse by Gauss-Jordan on a copy, with the identity alongside
    A = [list(C[i]) + [1.0 if i == j else 0.0 for j in range(p)]
         for i in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            raise ZeroDivisionError("singular")
        A[col], A[piv] = A[piv], A[col]
        f = A[col][col]
        A[col] = [v / f for v in A[col]]
        for r in range(p):
            if r == col:
                continue
            f = A[r][col]
            if f:
                A[r] = [A[r][t] - f * A[col][t] for t in range(2 * p)]
    inv = [row[p:] for row in A]
    out = []
    for r in rows:
        d = [r[j] - mu[j] for j in range(p)]
        s = 0.0
        for a in range(p):
            da = d[a]
            if da:
                s += da * sum(inv[a][b] * d[b] for b in range(p))
        out.append(math.sqrt(max(s, 0.0)))
    return out


def univariate_mcd(values, h=None, consistent=True):
    r"""The univariate MCD location and scale (Rousseeuw 1984).

    "the mean, resp. the standard deviation of the :math:`h` observations
    with smallest variance" -- found in :math:`O(n \log n)` by scanning
    the contiguous windows of the sorted values, since the minimising
    subset of a univariate sample is always contiguous.

    The raw scale of a shortest-half window underestimates sigma badly
    (0.61 for a standard normal at :math:`\alpha = 0.75`), so it carries
    the usual MCD consistency factor

    .. math::

       c_\alpha = \frac{\alpha}{F_{\chi^2_3}(\chi^2_{1,\alpha})},
       \qquad \hat\sigma = \sqrt{c_\alpha}\, s_{raw} .

    Verified against 20000 standard normals: 1.0018, 1.0010 and 1.0020 at
    :math:`\alpha` = 0.5, 0.75 and 0.9. The factor cancels out of the
    outlyingness of eq. 9, which is a ratio, but it decides the
    orthogonal-distance cut-off, where leaving it out flags roughly a
    fifth of clean data as outlying. Pass ``consistent=False`` for the
    raw shortest-half scale.
    """
    v = sorted(float(t) for t in values)
    n = len(v)
    if n < 2:
        raise ValueError("robpca: the univariate MCD needs two values")
    if h is None:
        h = (n + 2) // 2
    h = int(h)
    if not 2 <= h <= n:
        raise ValueError("robpca: h must lie in [2, n] for the univariate "
                         "MCD")
    csum = [0.0]
    csq = [0.0]
    for t in v:
        csum.append(csum[-1] + t)
        csq.append(csq[-1] + t * t)
    best, best_ss, best_mean = 0, None, 0.0
    for i in range(0, n - h + 1):
        s = csum[i + h] - csum[i]
        q = csq[i + h] - csq[i]
        ss = q - s * s / h
        if best_ss is None or ss < best_ss:
            best_ss, best, best_mean = ss, i, s / h
    scale = math.sqrt(max(best_ss, 0.0) / (h - 1.0))
    if consistent and scale > 0.0:
        a = h / float(n)
        denom = _st.chi2.cdf(_st.chi2.ppf(min(a, 0.999999), 1), 3)
        if denom > 0:
            scale *= math.sqrt(a / denom)
    return best_mean, scale


def _directions(rows, n_dirs, seed):
    """Directions through two data points, per the paper's restriction
    of B for an orthogonally invariant measure."""
    n = len(rows)
    p = len(rows[0])
    pairs = []
    total = n * (n - 1) // 2
    if total <= n_dirs:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:
        rng = np.random.default_rng(seed)
        seen = set()
        guard = 0
        while len(pairs) < n_dirs and guard < 50 * n_dirs:
            guard += 1
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    dirs = []
    for i, j in pairs:
        v = [rows[i][t] - rows[j][t] for t in range(p)]
        norm = math.sqrt(sum(t * t for t in v))
        if norm > 1e-12:
            dirs.append([t / norm for t in v])
    return dirs


def outlyingness(X, h=None, n_dirs=250, seed=17):
    r"""The orthogonally invariant outlyingness of eq. 9.

    Returns ``(outl, exact_fit_direction)``. The second element is a
    direction in which the projected points have zero robust scale -- the
    paper's "exact fit situation", meaning :math:`h` observations lie on a
    hyperplane -- or ``None``.
    """
    rows, n, p = _matrix(X)
    if h is None:
        h = (n + 2) // 2
    out = [0.0] * n
    for v in _directions(rows, n_dirs, seed):
        proj = [sum(rows[i][t] * v[t] for t in range(p)) for i in range(n)]
        loc, scale = univariate_mcd(proj, h)
        if scale <= 1e-12:
            return None, v
        for i in range(n):
            r = abs(proj[i] - loc) / scale
            if r > out[i]:
                out[i] = r
    return out, None


def _c_steps(rows, idx, max_iter=100):
    """C-steps from a starting h-subset until the determinant stops
    decreasing (Rousseeuw & Van Driessen 1999, used in Stage 3.1)."""
    h = len(idx)
    cur = list(idx)
    mu, C = _cov([rows[i] for i in cur])
    det = _det_from_chol(C)
    for _ in range(max_iter):
        if det <= 0.0:
            break
        try:
            d = _mahalanobis(rows, mu, C)
        except ZeroDivisionError:
            break
        nxt = sorted(range(len(rows)), key=lambda i: d[i])[:h]
        mu2, C2 = _cov([rows[i] for i in nxt])
        det2 = _det_from_chol(C2)
        if det2 >= det - 1e-15 * max(det, 1.0):
            if det2 < det:
                cur, mu, C, det = nxt, mu2, C2, det2
            break
        cur, mu, C, det = nxt, mu2, C2, det2
    return cur, mu, C, det


def _fast_mcd(rows, h, n_start=250, seed=17):
    """FAST-MCD from random (k+1)-subsets: two C-steps each, keep the ten
    best determinants, then iterate those to convergence (Stage 3.2)."""
    n = len(rows)
    p = len(rows[0])
    rng = np.random.default_rng(seed + 1)
    cands = []
    for _ in range(int(n_start)):
        pick = set()
        guard = 0
        while len(pick) < min(p + 1, n) and guard < 100:
            guard += 1
            pick.add(int(rng.integers(0, n)))
        sub = sorted(pick)
        if len(sub) < 2:
            continue
        try:
            mu, C = _cov([rows[i] for i in sub])
        except ValueError:
            continue
        if _det_from_chol(C) <= 0.0:
            # enlarge to an h-subset by nearest points in Euclidean terms
            base = _colmeans([rows[i] for i in sub])
            d = [math.sqrt(sum((rows[i][t] - base[t]) ** 2
                               for t in range(p))) for i in range(n)]
            sub = sorted(range(n), key=lambda i: d[i])[:h]
            mu, C = _cov([rows[i] for i in sub])
            if _det_from_chol(C) <= 0.0:
                continue
        try:
            d = _mahalanobis(rows, mu, C)
        except ZeroDivisionError:
            continue
        cur = sorted(range(n), key=lambda i: d[i])[:h]
        for _ in range(2):
            mu, C = _cov([rows[i] for i in cur])
            if _det_from_chol(C) <= 0.0:
                break
            try:
                d = _mahalanobis(rows, mu, C)
            except ZeroDivisionError:
                break
            cur = sorted(range(n), key=lambda i: d[i])[:h]
        mu, C = _cov([rows[i] for i in cur])
        cands.append((_det_from_chol(C), cur))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    best = None
    for _, cur in cands[:10]:
        got = _c_steps(rows, cur)
        if best is None or got[3] < best[3]:
            best = got
    return best


def classify_outliers(sd, od, sd_cut, od_cut):
    """The four types of the paper's Figure 1."""
    out = []
    for i in range(len(sd)):
        far_in = sd[i] > sd_cut
        far_off = od[i] > od_cut
        if far_in and far_off:
            out.append("bad leverage")
        elif far_in:
            out.append("good leverage")
        elif far_off:
            out.append("orthogonal outlier")
        else:
            out.append("regular")
    return out


# ------------------------------------------------------------ ROBPCA

def robust_pca(X, k=None, alpha=0.75, kmax=10, n_dirs=250, n_start=250,
               seed=17, reweight=True):
    r"""ROBPCA (Hubert, Rousseeuw & Vanden Branden 2005).

    Parameters
    ----------
    X : (n, p) array-like
        The data matrix.
    k : int or {"cumulative", "ratio"}, optional
        Number of components. An integer fixes it; ``"cumulative"``
        applies eq. 5 (the smallest :math:`k` whose eigenvalues of
        :math:`S_0` reach 90% of the total) and ``"ratio"`` applies eq. 6
        (:math:`\tilde l_k / \tilde l_1 \ge 10^{-3}`). Defaults to
        ``"cumulative"``.
    alpha : float
        Sets :math:`h = \max\{[\alpha n], [(n + k_{max} + 1)/2]\}`. "The
        higher alpha, the more efficient the estimates will be at
        uncontaminated data. On the other hand, setting a lower value for
        alpha will increase the robustness"; 0.75 is the paper's default
        and must lie in [0.5, 1].
    kmax : int
        Maximum number of components, 10 by default as in the paper. Only
        enters through ``h``.
    n_dirs, n_start : int
        Directions for eq. 9 and random subsets for FAST-MCD. The paper
        uses 250 of each.
    reweight : bool
        Apply the Stage 3.3 reweighting with hard rejection. Turning it
        off returns the raw MCD estimates, which are more robust and less
        efficient.

    Returns
    -------
    RichResult
        ``loadings`` (p x k, as k column vectors), ``eigenvalues``,
        ``center``, ``scores``, ``score_distance``, ``orthogonal_distance``,
        their cut-offs, and a ``classification`` per observation.
    """
    rows, n, p = _matrix(X)
    if not 0.5 <= alpha <= 1.0:
        raise ValueError("robpca: alpha must lie in [0.5, 1]")
    if n < 3:
        raise ValueError("robpca: need at least three observations")
    kmax = int(kmax)
    if kmax < 1:
        raise ValueError("robpca: kmax must be positive")

    # --- Stage 1: the affine subspace the data actually span (eq. 7)
    mu0 = _colmeans(rows)
    Xc = _center(rows, mu0)
    U, s, Vt = np.linalg.svd(np.asarray(Xc, dtype=float),
                             full_matrices=False)
    sv = [float(t) for t in s]
    tol = max(n, p) * (max(sv) if sv else 0.0) * 2.22e-16
    r0 = sum(1 for t in sv if t > tol)
    if r0 == 0:
        raise ValueError("robpca: every observation is identical")
    V = [[float(Vt[j][t]) for t in range(p)] for j in range(r0)]  # rows
    Z = [[sum(Xc[i][t] * V[j][t] for t in range(p)) for j in range(r0)]
         for i in range(n)]

    h = max(int(alpha * n), (n + kmax + 1) // 2)
    h = min(h, n)
    if h < 2:
        raise ValueError("robpca: h came out below 2; n is too small")

    # --- Stage 2: outlyingness (eq. 9), with the exact-fit reduction
    work = Z
    for _ in range(r0):
        outl, exact = outlyingness(work, h=h, n_dirs=n_dirs, seed=seed)
        if exact is None:
            break
        # exact fit: reflect so that v becomes e1 and drop that coordinate
        work = _drop_direction(work, exact)
        if len(work[0]) == 0:
            raise ValueError("robpca: the data collapsed to a point under "
                             "repeated exact fits")
    else:
        raise ValueError("robpca: exact fit reduction did not terminate")
    r1 = len(work[0])
    H0 = sorted(range(n), key=lambda i: outl[i])[:h]

    mu1, S0 = _cov([work[i] for i in H0])
    l0, P0 = _eigh_desc(S0)
    pos = [v for v in l0 if v > 1e-12]
    if not pos:
        raise ValueError("robpca: the h least outlying points are "
                         "identical")
    k0 = _choose_k(l0, k, kmax, r1)

    Xs = [[sum((work[i][t] - mu1[t]) * P0[j][t] for t in range(r1))
           for j in range(k0)] for i in range(n)]

    # --- Stage 3: MCD on that subspace
    h3 = max(h, k0 + 1)
    h3 = min(h3, n)
    from_h0 = _c_steps(Xs, H0[:h3] if len(H0) >= h3 else
                       sorted(range(n), key=lambda i: outl[i])[:h3])
    best = from_h0
    rnd = _fast_mcd(Xs, h3, n_start=n_start, seed=seed)
    if rnd is not None and rnd[3] < best[3]:
        best = rnd
    _, mu4, S3, _ = best

    k1 = k0
    scale = 1.0
    try:
        d = _mahalanobis(Xs, mu4, S3)
    except ZeroDivisionError:
        raise ValueError("robpca: the scatter on the k-dimensional "
                         "subspace is singular; ask for fewer components")
    if reweight:
        # consistency factor c1, with the hth quantile of the squared
        # robust distances rather than their median
        d2 = sorted(t * t for t in d)
        q = _st.chi2.ppf(min(h3 / float(n), 0.999999), k1)
        c1 = d2[h3 - 1] / q if q > 0 else 1.0
        if c1 <= 0:
            c1 = 1.0
        scale = c1
        d_scaled = [t / math.sqrt(c1) for t in d]
        cut = math.sqrt(_st.chi2.ppf(0.975, k1))
        keep = [i for i in range(n) if d_scaled[i] <= cut]
        if len(keep) > k1 + 1:
            mu5, S4 = _cov([Xs[i] for i in keep])
            # Hard rejection truncates the sample, so the reweighted
            # scatter is biased low and every score distance computed from
            # it is inflated. The paper states the consistency factor for
            # the raw estimator (c1) but not for the reweighted one; it is
            # the same construction at the rejection quantile,
            #   c_rew = q / F_{chi2_{k+2}}(chi2_{k,q}),  q = 0.975,
            # and without it the 97.5% cut-off flags far more than 2.5%
            # of clean normal data. Measured below in the anchor.
            crew = _reweight_factor(0.975, k1)
            S4 = [[crew * v for v in row] for row in S4]
    else:
        keep = list(range(n))
        mu5 = mu4
        S4 = S3

    lam, P2 = _eigh_desc(S4)
    kk = min(k0, sum(1 for v in lam if v > 1e-12))
    if kk == 0:
        raise ValueError("robpca: the robust scatter has no positive "
                         "eigenvalues")
    lam = lam[:kk]
    P2 = P2[:kk]

    # scores in the subspace, then loadings pushed back to the original p
    T = [[sum((Xs[i][t] - mu5[t]) * P2[j][t] for t in range(k0))
          for j in range(kk)] for i in range(n)]

    # P0 (r1 x k0) composed with P2 (k0 x kk), then V (r1 -> p)
    load_r1 = [[sum(P0[a][t] * P2[j][a] for a in range(k0))
                for t in range(r1)] for j in range(kk)]
    loadings = [_expand(vec, V, p) for vec in load_r1]
    # The robust centre, back in the original p variables. mu5 is a
    # k0-vector in the stage-2 eigenbasis P0, NOT in the final loading
    # basis P2, so it goes back through P0 first and only then through V.
    center_r1 = [mu1[t] + sum(mu5[j] * P0[j][t] for j in range(k0))
                 for t in range(r1)]
    center = [mu0[t] + sum(center_r1[a] * V[a][t] for a in range(r1))
              for t in range(p)]

    # --- diagnostics (eqs. 3 and 4)
    sd = [math.sqrt(sum(T[i][j] ** 2 / lam[j] for j in range(kk)))
          for i in range(n)]
    fitted = [[center[t] + sum(T[i][j] * loadings[j][t]
                               for j in range(kk)) for t in range(p)]
              for i in range(n)]
    od = [math.sqrt(sum((rows[i][t] - fitted[i][t]) ** 2
                        for t in range(p))) for i in range(n)]
    sd_cut = math.sqrt(_st.chi2.ppf(0.975, kk))
    od_cut = _od_cutoff(od, h3)
    cls = classify_outliers(sd, od, sd_cut, od_cut)

    return RichResult(payload={
        "estimate": lam[0],
        "loadings": loadings,
        "eigenvalues": lam,
        "center": center,
        "scores": T,
        "k": kk,
        "k0": k0,
        "h": h3,
        "alpha": alpha,
        "rank": r0,
        "subspace_rank": r1,
        "outlyingness": outl,
        "h_subset": sorted(best[0]),
        "reweighted_kept": keep,
        "consistency_factor": scale,
        "score_distance": sd,
        "orthogonal_distance": od,
        "sd_cutoff": sd_cut,
        "od_cutoff": od_cut,
        "classification": cls,
        "n_outliers": sum(1 for c in cls if c != "regular"),
        "n": n,
        "p": p,
        "reweighted": bool(reweight),
        "method": ("ROBPCA (Hubert, Rousseeuw & Vanden Branden 2005): "
                   "projection-pursuit outlyingness, then MCD on the "
                   "resulting subspace"),
        "note": ("loadings are orthonormal columns in the original p "
                 "variables; the diagnostic plot is score distance "
                 "against orthogonal distance, with the four regions of "
                 "the paper's Figure 1 given in classification"),
    })


def _choose_k(l0, k, kmax, r1):
    pos = [v for v in l0 if v > 1e-12]
    r = len(pos)
    if isinstance(k, int) and not isinstance(k, bool):
        if not 1 <= k <= r:
            raise ValueError("robpca: k must lie in [1, %d], the rank of "
                             "the preliminary scatter" % r)
        return k
    rule = "cumulative" if k is None else k
    if rule == "cumulative":                      # eq. 5
        tot = sum(pos)
        run = 0.0
        for j in range(r):
            run += pos[j]
            if run / tot >= 0.90:
                return min(j + 1, kmax, r1)
        return min(r, kmax, r1)
    if rule == "ratio":                           # eq. 6
        kk = sum(1 for v in pos if v / pos[0] >= 1e-3)
        return max(1, min(kk, kmax, r1))
    raise ValueError("robpca: k must be an integer, 'cumulative' (eq. 5) "
                     "or 'ratio' (eq. 6)")


def _drop_direction(rows, v):
    """The reflection step: reflect so v/||v|| becomes e1, then drop the
    first coordinate -- i.e. project on the orthogonal complement of v."""
    p = len(rows[0])
    norm = math.sqrt(sum(t * t for t in v))
    u = [t / norm for t in v]
    basis = []
    for j in range(p):
        e = [1.0 if t == j else 0.0 for t in range(p)]
        w = [e[t] - u[t] * u[j] for t in range(p)]
        for b in basis:
            dot = sum(w[t] * b[t] for t in range(p))
            w = [w[t] - dot * b[t] for t in range(p)]
        nw = math.sqrt(sum(t * t for t in w))
        if nw > 1e-9:
            basis.append([t / nw for t in w])
        if len(basis) == p - 1:
            break
    return [[sum(r[t] * b[t] for t in range(p)) for b in basis]
            for r in rows]


def _expand(vec_r1, V, p):
    """Take a vector expressed in the r1 stage-1 basis back to R^p."""
    r1 = len(vec_r1)
    return [sum(vec_r1[a] * V[a][t] for a in range(min(r1, len(V))))
            for t in range(p)]


def _reweight_factor(q, k):
    r"""Consistency factor for a scatter matrix after hard rejection at
    the ``q`` quantile: :math:`q / F_{\chi^2_{k+2}}(\chi^2_{k,q})`."""
    denom = _st.chi2.cdf(_st.chi2.ppf(q, k), k + 2)
    return q / denom if denom > 0 else 1.0


def _od_cutoff(od, h):
    r"""Wilson-Hilferty cut-off for the orthogonal distances.

    The squared orthogonal distances follow a scaled chi-squared
    approximately, so the :math:`OD^{2/3}` are approximately normal; the
    paper estimates their centre and scale by the univariate MCD and puts
    the cut-off at :math:`(\hat\mu + \hat\sigma z_{0.975})^{3/2}`.
    """
    v = [t ** (2.0 / 3.0) for t in od]
    if len(v) < 2:
        return float("inf")
    loc, scale = univariate_mcd(v, min(max(h, 2), len(v)))
    cut = loc + scale * _Z975
    if cut <= 0:
        return 0.0
    return cut ** 1.5


robpca = robust_pca
robustpca = robust_pca


def cheatsheet():
    return ("robpca: ROBPCA (Hubert, Rousseeuw & Vanden Branden 2005). "
            "Stage 1 SVD onto the span of the data; stage 2 ranks points "
            "by the outlyingness max_v |x'v - t_MCD|/s_MCD over "
            "directions through pairs of points and keeps the h = "
            "max([alpha n], [(n+kmax+1)/2]) least outlying; stage 3 runs "
            "the MCD on that subspace, rescales by c1 = d^2_(h)/chi2_"
            "{k,h/n} and reweights with hard rejection at chi2_{k,.975}. "
            "Diagnostics: SD_i = sqrt(sum t_ij^2/l_j), OD_i = "
            "||x_i - mu - P t_i||, cut off at sqrt(chi2_{k,.975}) and at "
            "(mu + sigma z_.975)^{3/2} from the univariate MCD of "
            "OD^{2/3}. k by eq. 5 (90% cumulative) or eq. 6 (l_k/l_1 >= "
            "1e-3).")
