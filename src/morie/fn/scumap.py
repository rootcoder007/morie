r"""UMAP: uniform manifold approximation and projection.

McInnes, L., Healy, J., & Melville, J. (2018) "UMAP: Uniform Manifold
Approximation and Projection for Dimension Reduction", arXiv:1802.03426.

The paper's Algorithm 1 has four parts, and all four are here.

**Local fuzzy simplicial sets** (Algorithm 2). For each point take its
``n_neighbors`` nearest neighbours, let :math:`\rho_i` be the distance to
the *nearest* one, and give the edge to neighbour :math:`j` the membership

.. math::

   \mu_{i \to j} = \exp\!\bigl(-\max\{0,\ d(x_i, x_j) - \rho_i\}
                               / \sigma_i \bigr).

Subtracting :math:`\rho_i` is what makes the construction local: every
point is connected to its nearest neighbour with membership exactly 1, no
matter how sparse its neighbourhood is.

**The smooth k-NN distance** (Algorithm 3). :math:`\sigma_i` is found by
binary search so that

.. math::

   \sum_{j=1}^{n} \exp\bigl(-(d_{ij} - \rho_i) / \sigma_i\bigr)
   = \log_2 n ,

which fixes the *fuzzy cardinality* of each point's neighbourhood rather
than its radius. The paper chose :math:`\log_2 n` "based on empirical
experiments".

There is one case where that equation has no solution: if every one of
the :math:`n` neighbours is the same distance away, then
:math:`d_j - \rho = 0` for all of them, every term is
:math:`\exp(0) = 1` whatever :math:`\sigma` is, and the sum is pinned at
:math:`n > \log_2 n`. The search then returns the scale floor rather
than a solution, which is the honest outcome -- the neighbourhood
carries no distance information to normalise.

**Symmetrisation.** The directed memberships are combined by the
probabilistic t-conorm,

.. math::

   B = A + A^{\top} - A \circ A^{\top},

read as: the probability that at least one of the two directed edges
exists.

**Layout.** Initialise with a spectral embedding of the graph
(Algorithm 4), then minimise the fuzzy cross entropy by stochastic
gradient descent with the paper's attractive and repulsive forces,

.. math::

   \frac{-2ab\,\lVert y_i - y_j \rVert_2^{2(b-1)}}
        {1 + \lVert y_i - y_j \rVert_2^{2}}\, w\,(y_i - y_j),
   \qquad
   \frac{2b}{(\epsilon + \lVert y_i - y_j \rVert_2^{2})
              (1 + a\lVert y_i - y_j \rVert_2^{2b})}\,(1 - w)\,(y_i - y_j)

with :math:`\epsilon = 0.001` as in the reference implementation, and the
learning rate annealed to zero over the epochs. :math:`a` and :math:`b`
are fitted so that the smooth curve
:math:`(1 + a d^{2b})^{-1}` matches the piecewise target
:math:`1` for :math:`d < \text{min\_dist}` and
:math:`\exp(-(d - \text{min\_dist}))` beyond it.

**A misprint in the paper.** Algorithm 4 writes the symmetric normalised
Laplacian as :math:`L = D^{1/2}(D - A)D^{1/2}`. That is not normalised --
it scales the Laplacian *up* by the degrees instead of down, and on a
graph with uneven degrees it gives a different eigenbasis. The intended
operator, and the one used here, is
:math:`L = D^{-1/2}(D - A)D^{-1/2}`; the reference implementation uses
that too. Set ``laplacian="as_printed"`` to get the paper's literal
formula and see the difference.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "umap_singlecell",
    "scumap",
    "smooth_knn_dist",
    "fuzzy_simplicial_set",
    "spectral_layout",
    "fit_ab",
]

_EPS = 0.001


def _matrix(X):
    rows = [list(map(float, r)) for r in np.asarray(X, dtype=float)]
    if not rows:
        raise ValueError("scumap: X is empty")
    p = len(rows[0])
    if p == 0:
        raise ValueError("scumap: X has no columns")
    for r in rows:
        if len(r) != p:
            raise ValueError("scumap: X is ragged")
        for v in r:
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("scumap: X contains a non-finite value")
    return rows, len(rows), p


def _dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(len(a))))


def smooth_knn_dist(distances, n_neighbors, rho=None, tol=1e-5,
                    max_iter=64, min_scale=1e-3):
    r"""Algorithm 3: the :math:`\sigma` that fixes the fuzzy cardinality.

    Binary search for :math:`\sigma` with
    :math:`\sum_j \exp(-(d_j - \rho)/\sigma) = \log_2 n`. Returns
    ``(sigma, rho)``.
    """
    d = sorted(float(v) for v in distances)
    if not d:
        raise ValueError("scumap: no distances to smooth over")
    n = int(n_neighbors)
    if n < 2:
        raise ValueError("scumap: n_neighbors must be at least 2")
    target = math.log(n, 2)
    if rho is None:
        nz = [v for v in d if v > 0.0]
        rho = nz[0] if nz else 0.0
    lo, hi, mid = 0.0, float("inf"), 1.0
    for _ in range(int(max_iter)):
        total = sum(math.exp(-max(0.0, v - rho) / mid) for v in d)
        if abs(total - target) < tol:
            break
        if total > target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            mid = mid * 2.0 if hi == float("inf") else (lo + hi) / 2.0
    # the reference implementation floors sigma relative to the local
    # scale so that a duplicated point cannot drive it to zero
    mean_d = sum(d) / len(d)
    if rho > 0.0:
        mid = max(mid, min_scale * mean_d)
    elif mean_d > 0.0:
        mid = max(mid, min_scale * mean_d)
    return mid, rho


def fuzzy_simplicial_set(X, n_neighbors=15, symmetrize=True):
    r"""Algorithms 2 and 3 plus the t-conorm: the weighted UMAP graph.

    Returns a dict with the directed memberships ``A``, the symmetrised
    ``B`` (:math:`A + A^{\top} - A \circ A^{\top}`), and the per-point
    :math:`\rho` and :math:`\sigma`.
    """
    rows, n, _ = _matrix(X)
    k = int(n_neighbors)
    if k < 2:
        raise ValueError("scumap: n_neighbors must be at least 2")
    if k >= n:
        raise ValueError("scumap: n_neighbors (%d) must be smaller than "
                         "the number of points (%d)" % (k, n))
    A = [[0.0] * n for _ in range(n)]
    rhos, sigmas, neighbours = [], [], []
    for i in range(n):
        order = sorted((j for j in range(n) if j != i),
                       key=lambda j: _dist(rows[i], rows[j]))[:k]
        dists = [_dist(rows[i], rows[j]) for j in order]
        sigma, rho = smooth_knn_dist(dists, k)
        rhos.append(rho)
        sigmas.append(sigma)
        neighbours.append(order)
        for j, dij in zip(order, dists):
            A[i][j] = math.exp(-max(0.0, dij - rho) / sigma)
    if not symmetrize:
        return {"A": A, "B": A, "rho": rhos, "sigma": sigmas,
                "neighbours": neighbours, "n": n}
    B = [[A[i][j] + A[j][i] - A[i][j] * A[j][i] for j in range(n)]
         for i in range(n)]
    return {"A": A, "B": B, "rho": rhos, "sigma": sigmas,
            "neighbours": neighbours, "n": n}


def _eigh_small(M):
    vals, vecs = np.linalg.eigh(np.asarray(M, dtype=float))
    vals = [float(v) for v in vals]
    p = len(vals)
    cols = [[float(vecs[i][j]) for i in range(p)] for j in range(p)]
    order = sorted(range(p), key=lambda j: vals[j])
    return [vals[j] for j in order], [cols[j] for j in order]


def spectral_layout(B, n_components=2, laplacian="normalised"):
    r"""Algorithm 4: initialise from the graph Laplacian's eigenvectors.

    ``laplacian="normalised"`` uses :math:`D^{-1/2}(D-A)D^{-1/2}`;
    ``"as_printed"`` uses the paper's literal :math:`D^{1/2}(D-A)D^{1/2}`,
    which is a misprint (see the module docstring).
    """
    if laplacian not in ("normalised", "as_printed"):
        raise ValueError("scumap: laplacian must be 'normalised' or "
                         "'as_printed'")
    n = len(B)
    d = [sum(B[i]) for i in range(n)]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            lij = (d[i] if i == j else 0.0) - B[i][j]
            if laplacian == "normalised":
                s = math.sqrt(d[i]) * math.sqrt(d[j])
                L[i][j] = lij / s if s > 0 else 0.0
            else:
                L[i][j] = math.sqrt(d[i]) * lij * math.sqrt(d[j])
    _, vecs = _eigh_small(L)
    # skip the trivial first eigenvector
    picked = vecs[1:1 + int(n_components)]
    while len(picked) < int(n_components):
        picked.append([0.0] * n)
    Y = [[picked[c][i] for c in range(int(n_components))]
         for i in range(n)]
    # scale to a sensible starting spread, as the reference does
    span = 0.0
    for c in range(int(n_components)):
        col = [Y[i][c] for i in range(n)]
        span = max(span, max(col) - min(col))
    if span > 0:
        Y = [[10.0 * v / span for v in row] for row in Y]
    return Y


def fit_ab(min_dist=0.1, spread=1.0, n_grid=300, iters=200):
    r"""Fit :math:`a, b` so that :math:`(1 + a d^{2b})^{-1}` matches the
    target curve (1 inside ``min_dist``, exponential decay outside)."""
    if min_dist < 0:
        raise ValueError("scumap: min_dist must be non-negative")
    if spread <= 0:
        raise ValueError("scumap: spread must be positive")
    xs = [3.0 * spread * t / float(n_grid - 1) for t in range(n_grid)]
    ys = [1.0 if x < min_dist else math.exp(-(x - min_dist) / spread)
          for x in xs]

    def loss(a, b):
        tot = 0.0
        for x, y in zip(xs, ys):
            if x <= 0:
                continue
            tot += (1.0 / (1.0 + a * x ** (2 * b)) - y) ** 2
        return tot

    a, b = 1.0, 1.0
    step = 0.5
    for _ in range(int(iters)):
        best = (loss(a, b), a, b)
        for da, db in ((step, 0.0), (-step, 0.0), (0.0, step),
                       (0.0, -step), (step, step), (-step, -step)):
            na, nb = a + da, b + db
            if na <= 0 or nb <= 0:
                continue
            v = loss(na, nb)
            if v < best[0]:
                best = (v, na, nb)
        if best[1] == a and best[2] == b:
            step /= 2.0
            if step < 1e-6:
                break
        else:
            a, b = best[1], best[2]
    return a, b


def _rng(seed):
    st = [int(seed) & 0x7FFFFFFF or 1]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def umap_singlecell(X, n_neighbors=15, min_dist=0.1, n_components=2,
                    n_epochs=200, learning_rate=1.0, spread=1.0,
                    negative_sample_rate=5, init="spectral", seed=0,
                    laplacian="normalised", a=None, b=None):
    """UMAP embedding of ``X`` (McInnes, Healy & Melville 2018).

    ``init`` is ``"spectral"`` (Algorithm 4, the paper's recommendation)
    or ``"random"``. ``a`` and ``b`` default to the fit against
    ``min_dist`` and ``spread``.
    """
    rows, n, _ = _matrix(X)
    if init not in ("spectral", "random"):
        raise ValueError("scumap: init must be 'spectral' or 'random'")
    if n_components < 1:
        raise ValueError("scumap: n_components must be at least 1")
    if learning_rate <= 0:
        raise ValueError("scumap: learning_rate must be positive")
    if n_epochs < 1:
        raise ValueError("scumap: n_epochs must be at least 1")
    graph = fuzzy_simplicial_set(X, n_neighbors)
    B = graph["B"]
    if a is None or b is None:
        a, b = fit_ab(min_dist, spread)
    d = int(n_components)

    rnd = _rng(seed + 1)
    if init == "spectral":
        Y = spectral_layout(B, d, laplacian)
    else:
        Y = [[20.0 * (rnd() - 0.5) for _ in range(d)] for _ in range(n)]

    edges = [(i, j, B[i][j]) for i in range(n) for j in range(i + 1, n)
             if B[i][j] > 0.0]
    if not edges:
        raise ValueError("scumap: the fuzzy graph has no edges")
    w_max = max(w for _, _, w in edges)

    for epoch in range(int(n_epochs)):
        alpha = learning_rate * (1.0 - epoch / float(n_epochs))
        for i, j, w in edges:
            if rnd() > w / w_max:      # sample edges by membership
                continue
            diff = [Y[i][c] - Y[j][c] for c in range(d)]
            dist2 = sum(v * v for v in diff)
            if dist2 > 0.0:
                coeff = (-2.0 * a * b * dist2 ** (b - 1.0)) / \
                    (1.0 + a * dist2 ** b)
            else:
                coeff = 0.0
            for c in range(d):
                g = _clip(coeff * diff[c])
                Y[i][c] += alpha * g
                Y[j][c] -= alpha * g
            for _ in range(int(negative_sample_rate)):
                k = int(rnd() * n)
                if k == i or k >= n:
                    continue
                diff = [Y[i][c] - Y[k][c] for c in range(d)]
                dist2 = sum(v * v for v in diff)
                if dist2 > 0.0:
                    coeff = (2.0 * b) / \
                        ((_EPS + dist2) * (1.0 + a * dist2 ** b))
                elif i != k:
                    coeff = 0.0
                else:
                    continue
                for c in range(d):
                    g = _clip(coeff * diff[c]) if dist2 > 0.0 else 4.0
                    Y[i][c] += alpha * g

    return RichResult(payload={
        "estimate": Y,
        "embedding": Y,
        "graph": B,
        "directed_graph": graph["A"],
        "rho": graph["rho"],
        "sigma": graph["sigma"],
        "neighbours": graph["neighbours"],
        "a": a,
        "b": b,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "n_components": d,
        "n_epochs": int(n_epochs),
        "init": init,
        "laplacian": laplacian,
        "n": n,
        "method": ("UMAP (McInnes, Healy & Melville 2018): fuzzy "
                   "simplicial sets, t-conorm symmetrisation, %s "
                   "initialisation, cross-entropy SGD" % init),
        "note": ("distances are Euclidean and neighbours are found "
                 "exactly, not approximately, so this is O(n^2) and "
                 "meant for the sample sizes an anchor can check; the "
                 "paper's Algorithm 4 misprints the normalised "
                 "Laplacian, see laplacian="),
    })


def _clip(v, lim=4.0):
    """The reference implementation clips gradients to +/- 4."""
    if v > lim:
        return lim
    if v < -lim:
        return -lim
    return v


scumap = umap_singlecell


def cheatsheet():
    return ("scumap: UMAP (McInnes, Healy & Melville 2018). Membership "
            "exp(-max(0, d - rho)/sigma) to each of the n_neighbors "
            "nearest points, rho the nearest-neighbour distance and "
            "sigma solved so the memberships sum to log2(n_neighbors); "
            "symmetrise by the t-conorm B = A + A' - A.A'; initialise "
            "from the normalised-Laplacian eigenvectors; then SGD on the "
            "fuzzy cross entropy with the paper's attractive and "
            "repulsive forces, a and b fitted to min_dist.")
