r"""Trimmed k-means: clustering that decides for itself what to discard.

Cuesta-Albertos, J. A., Gordaliza, A., & Matrán, C. (1997) "Trimmed
k-Means: An Attempt to Robustify Quantizers", *The Annals of
Statistics* 25(2), 553-576.

Ordinary k-means minimises :math:`V_\Phi(M) = \int \Phi(\min_j \|X -
m_j\|)\,dP` over :math:`k`-sets :math:`M = \{m_1,\dots,m_k\}`. It has a
breakdown point of zero: one sufficiently remote observation captures a
centre, and the paper opens by pointing out that swapping in a robust
location estimator does not fix it -- "the selection of two joint
medians through that formulation is very unstable: the introduction of
one, even very improbable, sufficiently remote value implies the
selection of such a value as one of the medians!"

The fix is **impartial trimming** (Gordaliza 1991a): discard a
proportion :math:`\alpha` of the mass, but let the *data* choose which
part, rather than nominating directions or zones in advance. For a
trimming set :math:`A` with :math:`P(A) \ge 1 - \alpha`,

.. math:: V_\Phi^A(M) = \frac{1}{P(A)} \int_A
          \Phi\big(d(X, M)\big)\,dP,

and the procedure minimises over both arguments in turn:

.. math:: V^A_{k,\Phi} = \inf_{\#M = k} V^A_\Phi(M),
          \qquad
          V_{k,\Phi,\alpha} = \inf_{P(A) \ge 1-\alpha} V^A_{k,\Phi}.

The paper calls the minimiser the "impartially :math:`\alpha`-trimmed
:math:`k`-:math:`\Phi`-mean", then mercifully shortens it to trimmed
:math:`k`-mean. Section 2 relaxes trimming *sets* to trimming
*functions* :math:`\tau : \mathbb{R}^p \to [0,1]` with
:math:`\int \tau\,dP \ge 1-\alpha`, which are more tractable; Corollary
3.2 then shows the best trimming function is essentially an indicator,
so the relaxation is tight and a hard 0/1 trim of the sample loses
nothing. That is what is implemented.

The penalty :math:`\Phi` is assumed continuous, nondecreasing, with
:math:`\Phi(0) = 0` and :math:`\Phi(x) < \Phi(\infty)`. Three are
offered, all satisfying that:

``"square"``
    :math:`\Phi(t) = t^2`. Trimmed k-means proper; the centre of a
    cluster is its mean.
``"absolute"``
    :math:`\Phi(t) = t`. Trimmed k-medians; the centre is the spatial
    median, computed by Weiszfeld iteration.
``"huber"``
    :math:`\Phi(t) = t^2` for :math:`t \le c` and
    :math:`c(2t - c)` beyond, i.e. Huber's function. Downweights
    rather than discards the moderately distant points, on top of the
    trimming of the rest.

Fitting is the concentration algorithm the trimming formulation
implies, and is structurally the same as Rousseeuw's LTS -- which the
paper names as its precedent. From a starting :math:`M`:

1. score every point by :math:`\Phi(d(x_i, M))`;
2. keep the :math:`\lceil n(1-\alpha) \rceil` smallest scores -- the
   trimming set, determined entirely by the data;
3. re-estimate each centre from the kept points assigned to it;
4. repeat until the assignment and the trimmed set stop changing.

Each iteration cannot increase :math:`V`, so it converges, but to a
local minimum -- hence ``n_start`` random restarts, keeping the best.

A trimmed k-means fit is also an outlier detector: the points left out
of the final trimming set are exactly the ones the criterion could not
accommodate, and section 6 of the paper is devoted to that use. They
are returned as ``outliers``.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rkmeans", "trimmed_kmeans"]

_PENALTIES = ("square", "absolute", "huber")


def _phi(t, penalty, c):
    if penalty == "square":
        return t * t
    if penalty == "absolute":
        return t
    return t * t if t <= c else c * (2.0 * t - c)


def _dist(x, m):
    s = 0.0
    for i in range(len(x)):
        d = x[i] - m[i]
        s += d * d
    return math.sqrt(s)


def _mean(pts):
    p = len(pts[0])
    out = [0.0] * p
    for x in pts:
        for i in range(p):
            out[i] += x[i]
    n = float(len(pts))
    return [v / n for v in out]


def _spatial_median(pts, tol=1e-10, max_iter=200):
    """Weiszfeld iteration: the minimiser of sum ||x - m||, i.e. the
    centre implied by Phi(t) = t."""
    m = _mean(pts)
    for _ in range(max_iter):
        num = [0.0] * len(m)
        den = 0.0
        coincident = False
        for x in pts:
            d = _dist(x, m)
            if d < 1e-12:
                coincident = True
                continue
            w = 1.0 / d
            den += w
            for i in range(len(m)):
                num[i] += w * x[i]
        if den <= 0.0:
            return m
        new = [v / den for v in num]
        shift = _dist(new, m)
        m = new
        if shift < tol and not coincident:
            break
    return m


def _huber_centre(pts, c, tol=1e-10, max_iter=200):
    """IRLS for Huber's Phi: weight 1 inside c, c/d outside."""
    m = _mean(pts)
    for _ in range(max_iter):
        num = [0.0] * len(m)
        den = 0.0
        for x in pts:
            d = _dist(x, m)
            w = 1.0 if d <= c else c / d
            den += w
            for i in range(len(m)):
                num[i] += w * x[i]
        if den <= 0.0:
            return m
        new = [v / den for v in num]
        shift = _dist(new, m)
        m = new
        if shift < tol:
            break
    return m


def rkmeans(X, k=2, alpha=0.1, penalty="square", n_start=20, max_iter=100,
            huber_c=1.345, seed=0, centers=None):
    r"""Fit an impartially :math:`\alpha`-trimmed :math:`k`-mean.

    Parameters
    ----------
    X : array-like
        ``(n, p)`` data matrix.
    k : int
        Number of clusters, the :math:`k` of the :math:`k`-set.
    alpha : float
        Trimming level :math:`\alpha \in [0, 1)`. The fit discards
        :math:`n - \lceil n(1-\alpha) \rceil` observations, chosen by
        the criterion rather than nominated. ``alpha=0`` reduces to
        ordinary k-means (Lloyd), which the anchors check.
    penalty : {"square", "absolute", "huber"}
        :math:`\Phi`; see the module docstring.
    n_start : int
        Random restarts. The criterion has local minima, so one start
        is not enough; the best fit by :math:`V` is returned.
    max_iter : int
        Concentration steps per start.
    huber_c : float
        The :math:`c` of Huber's :math:`\Phi`. Ignored otherwise.
    seed : int
        Seed for the restarts.
    centers : array-like, optional
        Explicit starting centres, ``(k, p)``. Used as an extra start
        alongside the random ones, so a good guess can only help.

    Returns
    -------
    RichResult
        ``estimate`` / ``centers`` are the fitted :math:`k` centres.
        ``labels`` gives each observation's cluster, with ``-1`` for
        the trimmed ones; ``kept`` and ``outliers`` are the index sets;
        ``criterion`` is :math:`V_{k,\Phi,\alpha}`, the mean penalty
        over the kept points, which is the quantity being minimised;
        ``sizes`` counts the clusters; ``n_trimmed`` and
        ``distances`` complete the picture.

    Examples
    --------
    Two tight clusters plus a wild point; the trimming finds it without
    being told where to look::

        X = [[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5], [5, 5.1],
             [100, 100]]
        fit = rkmeans(X, k=2, alpha=1.0 / 7)
        fit["outliers"]        # [6]

    References
    ----------
    Cuesta-Albertos, Gordaliza & Matrán (1997), *Ann. Statist.* 25(2),
    553-576: section 1 for :math:`V_\Phi`, section 2 for the trimming
    functions, Corollary 3.2 for the sufficiency of hard trimming.
    """
    rows = [[float(v) for v in r]
            for r in np.atleast_2d(np.asarray(X, dtype=float))]
    if not rows or not rows[0]:
        raise ValueError("rkmeans: X must be a non-empty (n, p) matrix")
    p = len(rows[0])
    for r in rows:
        if len(r) != p:
            raise ValueError("rkmeans: X must be rectangular")
    n = len(rows)
    k = int(k)
    if k < 1:
        raise ValueError("rkmeans: k must be >= 1")
    if k > n:
        raise ValueError("rkmeans: k = %d exceeds n = %d" % (k, n))
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0:
        raise ValueError("rkmeans: alpha must lie in [0, 1), got %r"
                         % (alpha,))
    if penalty not in _PENALTIES:
        raise ValueError("rkmeans: penalty must be one of %r, got %r"
                         % (_PENALTIES, penalty))
    huber_c = float(huber_c)
    if penalty == "huber" and not huber_c > 0.0:
        raise ValueError("rkmeans: huber_c must be > 0")

    n_keep = int(math.ceil(n * (1.0 - alpha)))
    if n_keep < k:
        raise ValueError("rkmeans: alpha = %g keeps only %d points, fewer "
                         "than k = %d" % (alpha, n_keep, k))

    rng = np.random.default_rng(seed)
    starts = []
    if centers is not None:
        c0 = [[float(v) for v in r]
              for r in np.atleast_2d(np.asarray(centers, dtype=float))]
        if len(c0) != k or len(c0[0]) != p:
            raise ValueError("rkmeans: centers must be (k, p)")
        starts.append(c0)
    for _ in range(max(1, int(n_start))):
        idx = set()
        while len(idx) < k:
            idx.add(int(rng.random() * n))
        starts.append([list(rows[i]) for i in sorted(idx)])

    best = None
    for init in starts:
        got = _concentrate(rows, init, k, n_keep, penalty, huber_c,
                           int(max_iter))
        if best is None or got[0] < best[0]:
            best = got
    crit, cen, labels, kept, dists = best

    sizes = [0] * k
    for i in kept:
        sizes[labels[i]] += 1
    outliers = [i for i in range(n) if labels[i] < 0]

    return RichResult(payload={
        "estimate": cen,
        "centers": cen,
        "labels": labels,
        "kept": kept,
        "outliers": outliers,
        "criterion": float(crit),
        "distances": dists,
        "sizes": sizes,
        "n_trimmed": len(outliers),
        "n_kept": len(kept),
        "alpha": alpha,
        "k": k,
        "penalty": penalty,
        "method": "trimmed k-means (Cuesta-Albertos et al. 1997)",
    })


def _concentrate(rows, cen, k, n_keep, penalty, huber_c, max_iter):
    """Steps 1-4 of the concentration algorithm; V never increases."""
    n = len(rows)
    cen = [list(c) for c in cen]
    prev = None
    labels = [-1] * n
    kept = []
    dists = [0.0] * n
    crit = float("inf")
    for _ in range(max_iter):
        # 1. score every point by Phi(d(x, M)), remembering the nearest
        #    centre. Ties go to the lower index, which keeps the fit
        #    reproducible.
        best_j = [0] * n
        for i in range(n):
            bd = None
            bj = 0
            for j in range(k):
                d = _dist(rows[i], cen[j])
                if bd is None or d < bd:
                    bd, bj = d, j
            dists[i] = bd
            best_j[i] = bj
        scores = sorted(range(n), key=lambda i: (_phi(dists[i], penalty,
                                                      huber_c), i))
        # 2. keep the n_keep smallest -- the trimming set.
        kept = sorted(scores[:n_keep])
        keptset = set(kept)
        labels = [best_j[i] if i in keptset else -1 for i in range(n)]
        crit = sum(_phi(dists[i], penalty, huber_c) for i in kept) / n_keep
        state = (tuple(labels), tuple(kept))
        if state == prev:
            break
        prev = state
        # 3. re-estimate each centre from its kept members.
        for j in range(k):
            pts = [rows[i] for i in kept if labels[i] == j]
            if not pts:
                # An emptied cluster is re-seeded on the worst-fitting
                # kept point, which is the standard repair and keeps the
                # k-set at size k rather than silently collapsing it.
                worst = max(kept, key=lambda i: dists[i])
                cen[j] = list(rows[worst])
                continue
            if penalty == "square":
                cen[j] = _mean(pts)
            elif penalty == "absolute":
                cen[j] = _spatial_median(pts)
            else:
                cen[j] = _huber_centre(pts, huber_c)
    return crit, cen, labels, kept, dists


def cheatsheet():
    return ("rkmeans: impartially alpha-trimmed k-Phi-means "
            "(Cuesta-Albertos, Gordaliza & Matran 1997). Minimises "
            "V = (1/P(A)) int_A Phi(d(x, M)) dP over k-sets M AND "
            "trimming sets A with P(A) >= 1-alpha -- the data choose "
            "what to discard, not the analyst. Phi in {square (k-means), "
            "absolute (k-medians), huber}. Corollary 3.2: hard trimming "
            "is optimal. Trimmed points are returned as outliers.")


# compact alias per ledger/NAMING.md
trimmed_kmeans = rkmeans

# public names resolved by fn/_lazy_map.json
trimmedkmeans = rkmeans
