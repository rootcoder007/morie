# morie.fn -- function file (rootcoder007/morie)
r"""Stahel-Donoho outlyingness and the estimator built on it.

**The idea.** A multivariate outlier need not be extreme in any
coordinate -- it can sit inside every marginal range and still be far
from the data cloud, in a direction nobody thought to look. So look
in *every* direction:

.. math:: r_i = \sup_{\|a\| = 1}
          \frac{|a'x_i - \mathrm{med}_j(a'x_j)|}
               {\mathrm{MAD}_j(a'x_j)}.

Each direction reduces the problem to one dimension, where the median
and the median absolute deviation give a robust standardisation. The
worst such standardised distance is the outlyingness. Downweighting
by :math:`w(r_i)` and taking the weighted mean and covariance gives
an affine-equivariant estimator with the highest possible breakdown
point.

**The supremum is not computable, and that matters.** It is a
maximisation over the sphere with a non-smooth objective. Every
implementation searches a finite set of directions, so every reported
outlyingness is a *lower bound* -- and adding directions can only
raise it. Two ways of choosing them are offered because they behave
differently:

``subsample`` takes the normal to the hyperplane through :math:`p`
data points, as Maronna and Yohai analyse. These directions are
generated *by the data*, so they transform with it: the estimator
built on them is exactly affine equivariant, and the anchor checks
that equality rather than a tolerance.

``random`` draws directions uniformly on the sphere. Simpler, and not
equivariant -- rotating the data changes the answer, because the
direction set does not rotate with it. It is provided because it is
what a fixed grid amounts to, and because seeing the equivariance
fail is the clearest statement of why the subsample route exists.

**The number of directions is the honest tuning knob.** With
:math:`\binom{n}{p}` small enough, every subsample is used and the
supremum over that family is exact; otherwise the count is a budget
and is reported alongside the estimate.

**A direction with zero MAD is skipped, and that is visible.** If
more than half the projections onto some direction coincide -- exactly
collinear data, or a discrete coordinate -- the MAD is zero and the
ratio is undefined. Such directions are dropped and ``n_used`` falls
below ``n_directions``, which is worth reading: on perfectly
collinear data the *most* informative direction is precisely the one
that gets dropped.

**The one-dimensional case pins the definition.** For :math:`p = 1`
there is only one direction, and :math:`r_i` must reduce exactly to
:math:`|x_i - \mathrm{med}| / \mathrm{MAD}`. That is a closed form,
and it is anchored.

References
----------
Maronna, R. A. & Yohai, V. J. (1995) "The behavior of the
Stahel-Donoho robust multivariate estimator", *Journal of the
American Statistical Association* 90(429), 330-341,
doi:10.1080/01621459.1995.10476517. The outlyingness above, its
origin in Stahel's (1981) ETH dissertation and Donoho's (1982)
Harvard qualifying paper, the subsampling of directions normal to
hyperplanes through p observations, the weighted mean and covariance,
and the breakdown and equivariance properties reproduced here.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["median", "mad", "outlyingness", "stahel_donoho",
           "DIRECTIONS", "stahel_donoho_outlyingness"]

DIRECTIONS = ("subsample", "random")
_CONSISTENCY = 1.4826  # MAD -> sigma at the normal


def median(v):
    r"""The sample median."""
    s = sorted(float(x) for x in v)
    n = len(s)
    if n == 0:
        raise ValueError("stahdo: the median of nothing is undefined")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def mad(v, consistent=True):
    r"""Median absolute deviation, scaled to estimate sigma."""
    m = median(v)
    d = median([abs(float(x) - m) for x in v])
    return d * (_CONSISTENCY if consistent else 1.0)


def _prep(X):
    M = [[float(v) for v in row] for row in X]
    n = len(M)
    if n < 3:
        raise ValueError("stahdo: need at least three observations")
    p = len(M[0]) if M else 0
    if p == 0 or any(len(r) != p for r in M):
        raise ValueError("stahdo: the data matrix is ragged or empty")
    return M, n, p


def _subsample_dirs(M, n, p, n_dirs, seed):
    """Normals to hyperplanes through p observations."""
    import itertools
    combos = None
    total = 1
    for i in range(p):
        total = total * (n - i) // (i + 1)
    if total <= max(int(n_dirs), 1):
        combos = list(itertools.combinations(range(n), p))
    else:
        rng = np.random.default_rng(int(seed))
        seen, combos = set(), []
        while len(combos) < int(n_dirs):
            idx = tuple(sorted({int(rng.random() * n) % n
                                for _ in range(p * 3)}))
            if len(idx) < p:
                continue
            idx = idx[:p]
            if idx in seen:
                continue
            seen.add(idx)
            combos.append(idx)
    dirs = []
    for idx in combos:
        base = M[idx[0]]
        rows = [[M[i][k] - base[k] for k in range(p)]
                for i in idx[1:]]
        a = _null_vector(rows, p)
        if a is not None:
            dirs.append(a)
    if not dirs:
        raise ValueError("stahdo: every sampled subset was degenerate "
                         "-- the data may lie in a lower-dimensional "
                         "subspace")
    return dirs, (combos is not None and total <= max(int(n_dirs), 1))


def _null_vector(rows, p):
    """A unit vector orthogonal to every row, or None."""
    A = [list(r) for r in rows]
    piv_cols, r = [], 0
    for c in range(p):
        piv = None
        for i in range(r, len(A)):
            if abs(A[i][c]) > 1e-10:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        f = A[r][c]
        A[r] = [v / f for v in A[r]]
        for i in range(len(A)):
            if i != r and abs(A[i][c]) > 0:
                g = A[i][c]
                A[i] = [A[i][k] - g * A[r][k] for k in range(p)]
        piv_cols.append(c)
        r += 1
    free = [c for c in range(p) if c not in piv_cols]
    if not free:
        return None
    fc = free[0]
    v = [0.0] * p
    v[fc] = 1.0
    for i, c in enumerate(piv_cols):
        v[c] = -A[i][fc]
    nrm = math.sqrt(sum(x * x for x in v))
    if nrm < 1e-12:
        return None
    return [x / nrm for x in v]


def _random_dirs(p, n_dirs, seed):
    rng = np.random.default_rng(int(seed))
    out = []
    while len(out) < int(n_dirs):
        v = [rng.random() * 2.0 - 1.0 for _ in range(p)]
        nrm = math.sqrt(sum(x * x for x in v))
        if nrm > 1e-9:
            out.append([x / nrm for x in v])
    return out


def outlyingness(X, directions="subsample", n_directions=500,
                 seed=1):
    r"""The Stahel-Donoho outlyingness of every observation.

    A lower bound on the supremum, over whichever directions were
    searched; the count and whether the family was exhausted are
    reported.
    """
    if directions not in DIRECTIONS:
        raise ValueError("stahdo: directions must be one of %s, got "
                         "%r" % (", ".join(DIRECTIONS), directions))
    M, n, p = _prep(X)
    if p == 1:
        dirs, exhaustive = [[1.0]], True
    elif directions == "subsample":
        dirs, exhaustive = _subsample_dirs(M, n, p, n_directions, seed)
    else:
        dirs, exhaustive = _random_dirs(p, n_directions, seed), False
    r = [0.0] * n
    used = 0
    for a in dirs:
        proj = [sum(M[i][k] * a[k] for k in range(p))
                for i in range(n)]
        s = mad(proj)
        if s <= 1e-12:
            continue      # a direction in which the data is constant
        used += 1
        m = median(proj)
        for i in range(n):
            d = abs(proj[i] - m) / s
            if d > r[i]:
                r[i] = d
    if used == 0:
        raise ValueError("stahdo: every searched direction has zero "
                         "MAD, so no outlyingness is defined")
    return {"outlyingness": r, "n_directions": len(dirs),
            "n_used": used, "exhaustive": exhaustive,
            "directions": directions}


def _weight(r, cutoff):
    c = float(cutoff)
    return 1.0 if r <= c else (c / r) ** 2


def stahel_donoho(X, directions="subsample", n_directions=500,
                  seed=1, cutoff=None):
    r"""The weighted location and scatter."""
    M, n, p = _prep(X)
    o = outlyingness(X, directions, n_directions, seed)
    r = o["outlyingness"]
    c = (math.sqrt(_chi2_median(p)) if cutoff is None
         else float(cutoff))
    if c <= 0:
        raise ValueError("stahdo: the cutoff must be positive")
    w = [_weight(x, c) for x in r]
    sw = sum(w)
    if sw <= 0:
        raise ValueError("stahdo: every observation was downweighted "
                         "to zero")
    loc = [sum(w[i] * M[i][k] for i in range(n)) / sw
           for k in range(p)]
    cov = [[sum(w[i] * (M[i][a] - loc[a]) * (M[i][b] - loc[b])
                for i in range(n)) / sw
            for b in range(p)] for a in range(p)]
    return RichResult(payload={
        "estimate": loc, "location": loc, "scatter": cov,
        "outlyingness": r, "weights": w, "cutoff": c,
        "n_directions": o["n_directions"], "n_used": o["n_used"],
        "exhaustive": o["exhaustive"], "directions": directions,
        "n_downweighted": sum(1 for x in w if x < 1.0),
        "n": n, "p": p,
        "method": "Stahel-Donoho estimator (Maronna & Yohai 1995) "
                  "with %s directions" % directions,
    })


def _chi2_median(p):
    """Median of chi-square with p degrees of freedom, by bisection."""
    lo, hi = 0.0, 100.0 + 10.0 * p
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(mid, p) < 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _chi2_cdf(x, k):
    if x <= 0:
        return 0.0
    # regularised lower incomplete gamma P(k/2, x/2) by series
    a, z = k / 2.0, x / 2.0
    if z < a + 1.0:
        term = 1.0 / a
        s = term
        for i in range(1, 500):
            term *= z / (a + i)
            s += term
            if abs(term) < 1e-15 * abs(s):
                break
        return s * math.exp(-z + a * math.log(z) - math.lgamma(a))
    # continued fraction for the upper tail
    b, cc, d, h = z + 1.0 - a, 1e300, 1.0 / (z + 1.0 - a), 0.0
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        cc = b + an / cc
        if abs(cc) < 1e-300:
            cc = 1e-300
        d = 1.0 / d
        delta = d * cc
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-z + a * math.log(z) - math.lgamma(a)) * h
    return 1.0 - q


def stahel_donoho_outlyingness(X, directions="subsample",
                               n_directions=500, seed=1,
                               cutoff=None):
    r"""Entry point: see :func:`stahel_donoho`."""
    return stahel_donoho(X, directions, n_directions, seed, cutoff)


# Catalogue aliases (src/morie/fn/_lazy_map.json resolves these by name).
staheldonoho = stahel_donoho
