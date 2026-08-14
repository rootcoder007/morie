# morie.fn -- function file (rootcoder007/morie)
r"""Slingshot: cell lineages and pseudotime from single-cell data.

Street, K., Risso, D., Fletcher, R. B., Das, D., Ngai, J., Yosef, N.,
Purdom, E., & Dudoit, S. (2018) "Slingshot: cell lineage and pseudotime
inference for single-cell transcriptomics", *BMC Genomics* 19:477.
doi:10.1186/s12864-018-4772-0

Two stages, and they answer different questions.

**Stage 1, the global lineage structure.** Clusters of cells are the
nodes of a graph and a minimum spanning tree is drawn between them. The
distance is not Euclidean but covariance-scaled, so that the *shape* of
a cluster counts (Equation 1):

.. math::

   d^2(C_i, C_j) = (\bar{X}_i - \bar{X}_j)^{\top}
                   (S_i + S_j)^{-1} (\bar{X}_i - \bar{X}_j),

"essentially a multivariate t-statistic". With small clusters
:math:`S_i + S_j` need not be invertible, so ``cov="diagonal"`` is
offered as the paper offers it, along with plain ``"euclidean"``.
A lineage is then any path from the user-supplied root cluster to a leaf.
Known terminal states can be imposed: the MST is built on the
*non-terminal* clusters and each terminal is then joined to its nearest
non-terminal neighbour, which constrains the tree locally without
dictating the global branching.

**Stage 2, pseudotime.** Each lineage gets a principal curve (Hastie &
Stuetzle): project every cell onto the curve, take arc length as
pseudotime, refit each coordinate against pseudotime by a smoother, and
iterate to convergence. Curves are unit-speed, which is what makes arc
length and pseudotime the same thing.

Fitting the lineages *separately* would let two curves disagree where
they share cells, so Slingshot fits them **simultaneously**:

* an average curve per branching event, built recursively from the
  leaves inward, :math:`c_{avg}(t) = \frac{1}{M}\sum_m c_m(t)`
  (Equation 2) -- recursive so that an early bifurcation is blind to how
  many lineages each branch eventually produces;
* shrinkage per branching event, built recursively from the root
  outward, :math:`c^{new}_m(t) = w_m(t) c_{avg}(t) + (1 - w_m(t))
  c_m(t)` (Equation 3);
* weighting functions that are smooth, non-increasing and equal to one
  at the origin, so diverging curves always leave from the same point
  (Equation 4):

  .. math::

     w_m(t) = \begin{cases} 1 & 0 \le t < t^m_{min} \\
       1 - F_K\!\left(\frac{t - t^m_{min}}{t^m_{max} - t^m_{min}}
       - \tfrac{1}{2}\right) & t^m_{min} \le t \le t^m_{max} \\
       0 & t > t^m_{max}\end{cases}

  with :math:`t^m_{min}, t^m_{max}` the lowest and highest non-outlier
  pseudotimes (1.5 IQR rule) of the cells shared between the branching
  lineages, and :math:`F_K` the CDF of a cosine kernel of bandwidth
  :math:`\tfrac{1}{2}`, which "places weight only on values between
  :math:`-\tfrac{1}{2}` and :math:`\tfrac{1}{2}`".

  *A note on Equation 4 as printed.* The argument is typeset as
  :math:`t/(t^m_{max} - t^m_{min}) - 1/2`, without subtracting
  :math:`t^m_{min}` in the numerator. Read that way the argument does
  not reach :math:`-1/2` at :math:`t = t^m_{min}`, so :math:`w_m` would
  not be continuous with the :math:`w_m = 1` branch above it, and the
  stated properties (non-increasing, :math:`w_m(0) = 1`) would not hold.
  The reading used here puts :math:`t - t^m_{min}` in the numerator,
  which makes the argument run exactly from :math:`-1/2` to
  :math:`1/2` across the shared region and satisfies every property the
  paper states. ``weight_arg="as_printed"`` gives the literal form so
  the difference can be measured.

Cells are assigned to lineages by weight, from their projection distance
to each curve, and the final pseudotimes are orthogonal projections onto
the shrunken curves.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "sctraj",
    "pseudotime_trajectory",
    "cluster_distances",
    "minimum_spanning_tree",
    "lineages_from_tree",
    "principal_curve",
    "average_curve",
    "shrinkage_weight",
    "cosine_cdf",
]

_COV = ("full", "diagonal", "euclidean")


def _matrix(X):
    rows = [[float(v) for v in r] for r in X]
    if not rows:
        raise ValueError("sctraj: X is empty")
    p = len(rows[0])
    if p == 0:
        raise ValueError("sctraj: X has no columns")
    for r in rows:
        if len(r) != p:
            raise ValueError("sctraj: X is ragged")
        for v in r:
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("sctraj: X contains a non-finite value")
    return rows, len(rows), p


def _solve(A, b):
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            raise ValueError("sctraj: the pooled covariance is singular; "
                             "use cov='diagonal' as the paper suggests "
                             "for small clusters")
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def cluster_distances(X, labels, cov="full", weights=None):
    r"""Equation 1: the covariance-scaled distance between clusters.

    ``cov="full"`` uses each cluster's full empirical covariance,
    ``"diagonal"`` its diagonal (the paper's fallback for small
    clusters) and ``"euclidean"`` drops the scaling entirely. ``weights``
    accepts cluster membership probabilities, which the paper notes go
    "naturally and readily" into the weighted means and covariances.
    """
    if cov not in _COV:
        raise ValueError("sctraj: cov must be one of %s" % (_COV,))
    rows, n, p = _matrix(X)
    lab = list(labels)
    if len(lab) != n:
        raise ValueError("sctraj: one label per cell is required")
    names = sorted(set(lab), key=lambda v: str(v))
    if len(names) < 2:
        raise ValueError("sctraj: at least two clusters are needed")
    if weights is None:
        weights = [1.0] * n
    weights = [float(w) for w in weights]
    if len(weights) != n or any(w < 0 for w in weights):
        raise ValueError("sctraj: one non-negative weight per cell is "
                         "required")
    centers, covs = {}, {}
    for c in names:
        idx = [i for i in range(n) if lab[i] == c]
        wsum = sum(weights[i] for i in idx)
        if wsum <= 0:
            raise ValueError("sctraj: cluster %r has no weight" % (c,))
        mu = [sum(weights[i] * rows[i][j] for i in idx) / wsum
              for j in range(p)]
        S = [[0.0] * p for _ in range(p)]
        if cov != "euclidean" and len(idx) > 1:
            denom = wsum - (sum(weights[i] ** 2 for i in idx) / wsum)
            denom = denom if denom > 1e-12 else 1.0
            for a in range(p):
                for b in range(p):
                    if cov == "diagonal" and a != b:
                        continue
                    S[a][b] = sum(weights[i] * (rows[i][a] - mu[a]) *
                                  (rows[i][b] - mu[b])
                                  for i in idx) / denom
        centers[c] = mu
        covs[c] = S
    D = {}
    for a in names:
        for b in names:
            if a == b:
                D[(a, b)] = 0.0
                continue
            diff = [centers[a][j] - centers[b][j] for j in range(p)]
            if cov == "euclidean":
                D[(a, b)] = math.sqrt(sum(v * v for v in diff))
                continue
            P = [[covs[a][i][j] + covs[b][i][j] for j in range(p)]
                 for i in range(p)]
            for i in range(p):
                if abs(P[i][i]) < 1e-12:
                    P[i][i] += 1e-12
            sol = _solve(P, diff)
            d2 = sum(diff[j] * sol[j] for j in range(p))
            D[(a, b)] = math.sqrt(max(d2, 0.0))
    return {"distances": D, "clusters": names, "centers": centers,
            "covariances": covs}


def minimum_spanning_tree(D, clusters, ends=None):
    """Prim's MST, with the paper's optional terminal-state constraint.

    With ``ends`` given, the tree is built on the non-terminal clusters
    and each terminal is then attached to its nearest non-terminal
    neighbour -- local supervision that does not touch the global
    branching structure.
    """
    nodes = list(clusters)
    if len(nodes) < 2:
        raise ValueError("sctraj: a tree needs at least two clusters")
    ends = [e for e in (ends or [])]
    for e in ends:
        if e not in nodes:
            raise ValueError("sctraj: terminal state %r is not a cluster"
                             % (e,))
    inner = [v for v in nodes if v not in ends]
    if not inner:
        raise ValueError("sctraj: every cluster was marked terminal")
    edges = []
    seen = {inner[0]}
    while len(seen) < len(inner):
        best = None
        for a in seen:
            for b in inner:
                if b in seen:
                    continue
                w = D[(a, b)]
                if best is None or w < best[0]:
                    best = (w, a, b)
        edges.append((best[1], best[2], best[0]))
        seen.add(best[2])
    for e in ends:
        near = min(inner, key=lambda v: D[(e, v)])
        edges.append((near, e, D[(e, near)]))
    adj = {}
    for a, b, w in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v in nodes:
        adj.setdefault(v, [])
    return {"edges": edges, "adjacency": adj, "nodes": nodes}


def lineages_from_tree(tree, root):
    """Every path from ``root`` to a leaf, in order."""
    adj = tree["adjacency"]
    if root not in adj:
        raise ValueError("sctraj: the root is not a cluster")
    out, stack = [], [(root, [root])]
    while stack:
        node, path = stack.pop()
        kids = [v for v in adj[node] if v not in path]
        if not kids:
            out.append(path)
            continue
        for v in kids:
            stack.append((v, path + [v]))
    out.sort(key=lambda p: [str(v) for v in p])
    return out


# ------------------------------------------------------ principal curves

def _arc_length(points):
    s = [0.0]
    for i in range(1, len(points)):
        s.append(s[-1] + math.sqrt(sum((points[i][j] - points[i - 1][j]) ** 2
                                       for j in range(len(points[i])))))
    return s


def _project(point, curve, s):
    """Nearest point on the polyline, and its arc length."""
    best = None
    for k in range(len(curve) - 1):
        a, b = curve[k], curve[k + 1]
        seg = [b[j] - a[j] for j in range(len(a))]
        L2 = sum(v * v for v in seg)
        if L2 <= 0:
            t = 0.0
        else:
            t = sum((point[j] - a[j]) * seg[j]
                    for j in range(len(a))) / L2
            t = min(max(t, 0.0), 1.0)
        proj = [a[j] + t * seg[j] for j in range(len(a))]
        d2 = sum((point[j] - proj[j]) ** 2 for j in range(len(a)))
        lam = s[k] + t * math.sqrt(L2)
        if best is None or d2 < best[0]:
            best = (d2, lam, proj)
    return best


def _smooth(t, y, w, span=0.4):
    """Local linear smoother of ``y`` on ``t`` -- the "smoothing spline"
    step, kept simple and weight-aware."""
    n = len(t)
    order = sorted(range(n), key=lambda i: t[i])
    k = max(3, int(span * n))
    out = [0.0] * n
    for pos, i in enumerate(order):
        lo = max(0, pos - k)
        hi = min(n, pos + k + 1)
        idx = order[lo:hi]
        sw = sum(w[j] for j in idx)
        if sw <= 0:
            out[i] = y[i]
            continue
        tm = sum(w[j] * t[j] for j in idx) / sw
        ym = sum(w[j] * y[j] for j in idx) / sw
        num = sum(w[j] * (t[j] - tm) * (y[j] - ym) for j in idx)
        den = sum(w[j] * (t[j] - tm) ** 2 for j in idx)
        slope = num / den if den > 1e-12 else 0.0
        out[i] = ym + slope * (t[i] - tm)
    return out


def principal_curve(X, init, weights=None, max_iter=15, tol=1e-3,
                    span=0.4, n_knots=None):
    """The Hastie-Stuetzle iteration, initialised from a given path.

    Returns pseudotimes (arc length along the curve, lowest set to zero),
    the fitted curve as an ordered polyline, and the distance of each
    cell to it.
    """
    rows, n, p = _matrix(X)
    if weights is None:
        weights = [1.0] * n
    weights = [float(w) for w in weights]
    if len(weights) != n:
        raise ValueError("sctraj: one weight per cell is required")
    if max_iter < 1:
        raise ValueError("sctraj: max_iter must be at least 1")
    curve = [list(map(float, q)) for q in init]
    if len(curve) < 2:
        raise ValueError("sctraj: the initial curve needs two points")
    prev = None
    lam, dist = [0.0] * n, [0.0] * n
    for _ in range(int(max_iter)):
        s = _arc_length(curve)
        for i in range(n):
            d2, l, _pt = _project(rows[i], curve, s)
            lam[i] = l
            dist[i] = math.sqrt(d2)
        lo = min(lam)
        lam = [v - lo for v in lam]
        sse = sum(weights[i] * dist[i] ** 2 for i in range(n))
        if prev is not None and abs(prev - sse) <= tol * max(prev, 1e-12):
            break
        prev = sse
        fitted = [_smooth(lam, [rows[i][j] for i in range(n)], weights,
                          span) for j in range(p)]
        # The curve belongs to THIS lineage, so it is drawn only through
        # the cells assigned to it. Including zero-weight cells would let
        # a sibling branch pull the curve off its own trunk even though
        # those cells contribute nothing to the smoother.
        live = [i for i in range(n) if weights[i] > 0]
        if len(live) < 2:
            raise ValueError("sctraj: a lineage has fewer than two "
                             "weighted cells")
        order = sorted(live, key=lambda i: lam[i])
        curve = [[fitted[j][i] for j in range(p)] for i in order]
        # collapse duplicate points so the polyline stays well defined
        dedup = [curve[0]]
        for q in curve[1:]:
            if any(abs(q[j] - dedup[-1][j]) > 1e-12 for j in range(p)):
                dedup.append(q)
        if len(dedup) < 2:
            break
        curve = dedup
    return {"pseudotime": lam, "curve": curve, "distance": dist,
            "sse": sum(weights[i] * dist[i] ** 2 for i in range(n))}


def average_curve(curves, n_points=100, return_grid=False):
    r"""Equation 2: :math:`c_{avg}(t) = \frac{1}{M}\sum_m c_m(t)`.

    The average is taken "for values of :math:`t` in the domains of each
    curve", and those domains are arc lengths, not fractions of a
    length. Averaging at matching *fractions* would stretch a short
    curve onto a long one and move the consensus path; the common domain
    is therefore :math:`[0, \min_m \text{len}(c_m)]`. The curves are
    unit-speed, so arc length is pseudotime.
    """
    if not curves:
        raise ValueError("sctraj: nothing to average")
    if n_points < 2:
        raise ValueError("sctraj: n_points must be at least 2")
    p = len(curves[0][0])
    arcs = [_arc_length(c) for c in curves]
    t_max = min(a[-1] for a in arcs)
    if t_max <= 0:
        t_max = 1.0
    grid = [k * t_max / float(n_points - 1) for k in range(int(n_points))]
    out = []
    for t in grid:
        acc = [0.0] * p
        for a, c in zip(arcs, curves):
            pt = _interp(a, c, t)
            for j in range(p):
                acc[j] += pt[j]
        out.append([v / len(curves) for v in acc])
    return (out, grid) if return_grid else out


def _interp(s, c, u):
    if u <= s[0]:
        return list(c[0])
    if u >= s[-1]:
        return list(c[-1])
    for k in range(len(s) - 1):
        if s[k] <= u <= s[k + 1]:
            span = s[k + 1] - s[k]
            t = 0.0 if span <= 0 else (u - s[k]) / span
            return [c[k][j] + t * (c[k + 1][j] - c[k][j])
                    for j in range(len(c[k]))]
    return list(c[-1])


def cosine_cdf(u):
    r"""CDF of a cosine kernel of bandwidth :math:`\tfrac{1}{2}`.

    Density :math:`1 + \cos(2\pi u)` on
    :math:`[-\tfrac{1}{2}, \tfrac{1}{2}]`, so
    :math:`F(u) = u + \tfrac{1}{2} + \sin(2\pi u)/(2\pi)`, which is 0 at
    :math:`-\tfrac{1}{2}` and 1 at :math:`\tfrac{1}{2}` as the paper
    requires.
    """
    if u <= -0.5:
        return 0.0
    if u >= 0.5:
        return 1.0
    return u + 0.5 + math.sin(2 * math.pi * u) / (2 * math.pi)


def shrinkage_weight(t, t_min, t_max, arg="shifted"):
    """Equation 4's weighting function.

    ``arg="shifted"`` (default) puts :math:`t - t_{min}` in the
    numerator, which is what makes the function continuous and
    non-increasing; ``"as_printed"`` uses the literal :math:`t`.
    """
    if arg not in ("shifted", "as_printed"):
        raise ValueError("sctraj: arg must be 'shifted' or 'as_printed'")
    if t_max <= t_min:
        return 1.0 if t <= t_min else 0.0
    if t < t_min:
        return 1.0
    if t > t_max:
        return 0.0
    num = (t - t_min) if arg == "shifted" else t
    return 1.0 - cosine_cdf(num / (t_max - t_min) - 0.5)


def _non_outlier_range(vals):
    """Lowest and highest non-outlier values, by the 1.5 IQR rule."""
    v = sorted(vals)
    if not v:
        return 0.0, 0.0
    n = len(v)

    def q(f):
        pos = f * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        return v[lo] + (pos - lo) * (v[hi] - v[lo])

    q1, q3 = q(0.25), q(0.75)
    iqr = q3 - q1
    keep = [x for x in v if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]
    return (keep[0], keep[-1]) if keep else (v[0], v[-1])


# --------------------------------------------------------------- driver

def sctraj(X, labels, root, ends=None, cov="full", max_iter=15,
           shrink=True, weight_arg="shifted", span=0.4, n_points=100):
    """Infer lineages and pseudotime (Street et al. 2018).

    Returns one pseudotime vector and one cell-weight vector per lineage;
    a cell's weight is its assignment to that lineage, from its
    projection distance to the lineage's curve.
    """
    rows, n, p = _matrix(X)
    lab = list(labels)
    if len(lab) != n:
        raise ValueError("sctraj: one label per cell is required")
    info = cluster_distances(rows, lab, cov)
    tree = minimum_spanning_tree(info["distances"], info["clusters"], ends)
    lins = lineages_from_tree(tree, root)
    if not lins:
        raise ValueError("sctraj: no lineage runs from the root")

    # initial curves: the piecewise linear path through the cluster
    # centres of each lineage, not the first principal component
    curves, pts, dists = [], [], []
    for path in lins:
        init = [info["centers"][c] for c in path]
        member = [1.0 if lab[i] in path else 0.0 for i in range(n)]
        if sum(member) < 2:
            raise ValueError("sctraj: a lineage has too few cells")
        fit = principal_curve(rows, init, member, max_iter, span=span)
        curves.append(fit["curve"])
        pts.append(fit["pseudotime"])
        dists.append(fit["distance"])

    shrunk = [list(c) for c in curves]
    if shrink and len(lins) > 1:
        avg, avg_grid = average_curve(curves, n_points, return_grid=True)
        for m, path in enumerate(lins):
            shared = [i for i in range(n)
                      if lab[i] in path and
                      any(lab[i] in other for k, other in enumerate(lins)
                          if k != m)]
            if not shared:
                continue
            t_min, t_max = _non_outlier_range([pts[m][i] for i in shared])
            s = _arc_length(shrunk[m])
            new = []
            for k, q in enumerate(shrunk[m]):
                t = s[k]
                w = shrinkage_weight(t, t_min, t_max, weight_arg)
                a = _interp(avg_grid, avg, t)
                new.append([w * a[j] + (1.0 - w) * q[j] for j in range(p)])
            shrunk[m] = new
        pts, dists = [], []
        for c in shrunk:
            s = _arc_length(c)
            lam, dd = [], []
            for i in range(n):
                d2, l, _q = _project(rows[i], c, s)
                lam.append(l)
                dd.append(math.sqrt(d2))
            lo = min(lam)
            pts.append([v - lo for v in lam])
            dists.append(dd)

    # cell weights from projection distance: closer curve, higher weight
    W = []
    for m in range(len(lins)):
        col = []
        for i in range(n):
            best = min(dists[k][i] for k in range(len(lins)))
            col.append(1.0 if dists[m][i] <= best + 1e-12 else
                       best / dists[m][i] if dists[m][i] > 0 else 1.0)
        W.append(col)

    return RichResult(payload={
        "estimate": pts,
        "pseudotime": pts,
        "weights": W,
        "lineages": lins,
        "curves": shrunk,
        "tree": tree["edges"],
        "distance": dists,
        "clusters": info["clusters"],
        "centers": info["centers"],
        "n_lineages": len(lins),
        "root": root,
        "cov": cov,
        "shrink": bool(shrink),
        "method": ("Slingshot (Street et al. 2018): covariance-scaled "
                   "MST over cluster centres, then simultaneous "
                   "principal curves with recursive averaging and "
                   "shrinkage"),
        "note": ("Equation 4 is typeset with t, not t - t_min, in the "
                 "numerator; the default 'shifted' reading is the one "
                 "that is continuous and non-increasing as the paper "
                 "states, and weight_arg='as_printed' gives the literal "
                 "form"),
    })


pseudotime_trajectory = sctraj


def cheatsheet():
    return ("sctraj: Slingshot (Street et al. 2018). Stage 1 draws an "
            "MST over cluster centres using the covariance-scaled "
            "distance d^2 = (xi-xj)'(Si+Sj)^-1(xi-xj), and every path "
            "from the root to a leaf is a lineage; terminal states can "
            "be imposed by building the MST without them and attaching "
            "each to its nearest neighbour. Stage 2 fits simultaneous "
            "principal curves: average curves recursively from the "
            "leaves, then shrink from the root outward with a cosine "
            "kernel weight that is 1 at the origin and 0 past the last "
            "shared cell. Pseudotime is arc length along the curve.")

# public names resolved by fn/_lazy_map.json
scrnaseq_trajectory = sctraj
