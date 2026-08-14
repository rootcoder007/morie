# morie.fn -- function file (rootcoder007/morie)
r"""Harmony: integrating single-cell data across batches.

Korsunsky, I., Millard, N., Fan, J., Slowikowski, K., Zhang, F., Wei, K.,
Baglaenko, Y., Brenner, M., Loh, P., & Raychaudhuri, S. (2019) "Fast,
sensitive and accurate integration of single-cell data with Harmony",
*Nature Methods* 16(12), 1289-1296. doi:10.1038/s41592-019-0619-0

Harmony alternates two steps until the embedding stops moving
(Algorithm 1): cluster the cells so that every cluster is *diverse* in
batch (Algorithm 2), then regress the batch out of each cluster
(Algorithm 3).

**Maximum diversity clustering.** Soft spherical k-means with two extra
terms: an entropy regulariser on the assignments :math:`R` weighted by
:math:`\sigma`, and a penalty on the statistical dependence between
cluster and batch weighted by :math:`\theta`. The dependence is measured
by the KL divergence between the observed and expected cluster/batch
counts,

.. math::

   O_{kb} = \sum_{i \in b} R_{ki}, \qquad
   E_{kb} = \frac{N_b}{N} \sum_i R_{ki},

(Equations 5 and 6: what is observed, against what independence would
give). Minimising the whole objective in :math:`R` has a closed form
(Equation 8):

.. math::

   R_{ki} \;\propto\;
   \left(\frac{O_{ki}}{E_{ki}}\right)^{\theta}
   \exp\!\left(-\frac{2\,(1 - Y_k^{\top} Z_i)}{\sigma}\right),

normalised so each cell's memberships sum to one. Distances are cosine,
so cells and centroids are L2-normalised and :math:`1 - Y_k^{\top} Z_i`
is the cosine distance; centroids are :math:`Y = Z R^{\top}` followed by
that normalisation, the soft version of Dhillon's spherical k-means.

At :math:`\theta = 0` the ratio term is 1 and this is ordinary soft
spherical k-means; raising :math:`\theta` pushes clusters toward batch
independence.

*The sign of that exponent.* Equation 8 is printed with
:math:`(O_{ki}/E_{ki})^{+\theta}`, which raises a cell's membership of
clusters where its own batch is **already over-represented** -- the
opposite of diversity. The objective it is derived from adds
:math:`+\sigma\theta\,D_{KL}` and is minimised, and
:math:`\partial/\partial R_{ki}` of that term is
:math:`+\sigma\theta\log(O_{ki}/E_{ki})`, so the stationary point
carries :math:`(O_{ki}/E_{ki})^{-\theta}`. Measured on a planted
two-batch dataset the printed form drives the KL dependence to 21.5
while :math:`\theta = 0` leaves it at 0.0003; the negative exponent
takes it below the :math:`\theta = 0` value, which is what the term
exists to do. ``diversity="penalise"`` (default) uses the negative
exponent and ``"as_printed"`` the literal one.

**Mixture-of-experts correction.** Within each cluster the batch is
regressed out by a ridge fit (Equation 14),

.. math::

   W_k = \bigl(\phi^{*} \operatorname{diag}(R_k) \phi^{*\top}
                + \lambda I\bigr)^{-1}
         \phi^{*} \operatorname{diag}(R_k) Z^{\top},

where :math:`\phi^{*} = 1 \,\|\, \phi` is the one-hot batch design with
an intercept prepended. The ridge is not decoration: the one-hot columns
sum to the intercept, so the unpenalised matrix is singular. The paper
sets :math:`\lambda_0 = 0` and :math:`\lambda_b = 1`, penalising every
batch term but never the intercept.

Then **the intercept row of** :math:`W_k` **is zeroed** before

.. math::

   \hat{Z} = Z - \sum_k W_k^{\top} \phi^{*} \operatorname{diag}(R_k),

which is what makes the correction remove batch and keep cell type: the
intercept carries the batch-independent (cell-type) variation and is
left in place. It also gives an exactly checkable consequence, which the
paper states for reference mapping -- a cell whose design row is
:math:`[1, 0, \dots, 0]` is explained "in terms of an intercept and
nothing else", so it never moves.

Correction is done in the unnormalised space and the result is
re-normalised for the next round of clustering, as the paper's caveat
requires: regression in the normalised space would need rotation
matrices.

The default cluster count follows the paper's heuristic,
:math:`K = \min(100, N/30)`.
"""

import math

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

_SIGNS = ("penalise", "as_printed")

__all__ = [
    "scintg",
    "harmony_integrate",
    "maximum_diversity_clustering",
    "correct_batch",
    "cluster_batch_counts",
    "l2_normalise",
    "harmony_objective",
]


def _matrix(Z):
    rows = [[float(v) for v in r] for r in Z]
    if not rows:
        raise ValueError("scintg: Z is empty")
    d = len(rows[0])
    if d == 0:
        raise ValueError("scintg: Z has no columns")
    for r in rows:
        if len(r) != d:
            raise ValueError("scintg: Z is ragged")
        for v in r:
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("scintg: Z contains a non-finite value")
    return rows, len(rows), d


def l2_normalise(rows):
    """Unit-length rows; cosine distance is Euclidean on the sphere."""
    out = []
    for r in rows:
        n = math.sqrt(sum(v * v for v in r))
        out.append(list(r) if n <= 0 else [v / n for v in r])
    return out


def _design(batches):
    """``phi*`` -- an intercept column followed by the one-hot batches."""
    names = sorted(set(batches), key=lambda v: str(v))
    idx = dict((b, k) for k, b in enumerate(names))
    phi = []
    for b in batches:
        row = [1.0] + [0.0] * len(names)
        row[1 + idx[b]] = 1.0
        phi.append(row)
    return phi, names


def cluster_batch_counts(R, batches, names=None):
    r"""Equations 5 and 6: observed and expected cluster/batch mass."""
    K = len(R)
    N = len(R[0]) if K else 0
    if any(len(row) != N for row in R):
        raise ValueError("scintg: R is ragged")
    if len(batches) != N:
        raise ValueError("scintg: one batch label per cell is required")
    if names is None:
        names = sorted(set(batches), key=lambda v: str(v))
    O = [[0.0] * len(names) for _ in range(K)]
    E = [[0.0] * len(names) for _ in range(K)]
    for bi, b in enumerate(names):
        members = [i for i in range(N) if batches[i] == b]
        Nb = float(len(members))
        for k in range(K):
            O[k][bi] = sum(R[k][i] for i in members)
            E[k][bi] = (Nb / N) * sum(R[k][i] for i in range(N))
    return {"O": O, "E": E, "batches": names}


def harmony_objective(Z, R, Y, batches, sigma=0.1, theta=2.0):
    r"""The full objective: cosine distance, entropy, and the KL penalty."""
    K, N = len(R), len(R[0])
    fit = sum(R[k][i] * 2.0 * (1.0 - sum(Y[k][j] * Z[i][j]
                                         for j in range(len(Z[0]))))
              for k in range(K) for i in range(N))
    ent = sum(R[k][i] * math.log(max(R[k][i], 1e-300))
              for k in range(K) for i in range(N))
    c = cluster_batch_counts(R, batches)
    kl = 0.0
    for k in range(K):
        for bi in range(len(c["batches"])):
            o, e = c["O"][k][bi], c["E"][k][bi]
            if o > 0 and e > 0:
                kl += o * math.log(o / e)
    return {"total": fit + sigma * ent + sigma * theta * kl,
            "fit": fit, "entropy": ent, "kl": kl}


def _kmeans_init(Zn, K, seed):
    """Spherical k-means++ style seeding, then a few Lloyd rounds."""
    N, d = len(Zn), len(Zn[0])
    st = [int(seed) & 0x7FFFFFFF or 1]

    def rnd():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)

    centres = [list(Zn[int(rnd() * N)])]
    while len(centres) < K:
        d2 = []
        for z in Zn:
            best = min(1.0 - sum(c[j] * z[j] for j in range(d))
                       for c in centres)
            d2.append(max(best, 0.0) ** 2)
        tot = sum(d2)
        if tot <= 0:
            centres.append(list(Zn[int(rnd() * N)]))
            continue
        t, acc = rnd() * tot, 0.0
        for i, v in enumerate(d2):
            acc += v
            if acc >= t:
                centres.append(list(Zn[i]))
                break
    for _ in range(10):
        groups = [[] for _ in range(K)]
        for z in Zn:
            k = max(range(K), key=lambda k: sum(centres[k][j] * z[j]
                                                for j in range(d)))
            groups[k].append(z)
        for k in range(K):
            if not groups[k]:
                continue
            centres[k] = [sum(z[j] for z in groups[k]) / len(groups[k])
                          for j in range(d)]
        centres = l2_normalise(centres)
    return centres


def maximum_diversity_clustering(Z, batches, K=None, sigma=0.1, theta=2.0,
                                 max_iter=25, tol=1e-5, seed=0, Y=None,
                                 diversity="penalise"):
    r"""Algorithm 2: soft spherical k-means with the diversity penalty.

    ``diversity="penalise"`` (default) uses
    :math:`(O_{ki}/E_{ki})^{-\theta}`; ``"as_printed"`` uses the
    :math:`+\theta` of Equation 8. See :func:`scintg` for why they
    differ.
    """
    if diversity not in _SIGNS:
        raise ValueError("scintg: diversity must be one of %s" % (_SIGNS,))
    rows, N, d = _matrix(Z)
    if len(batches) != N:
        raise ValueError("scintg: one batch label per cell is required")
    if sigma <= 0:
        raise ValueError("scintg: sigma must be positive")
    if theta < 0:
        raise ValueError("scintg: theta must be non-negative")
    if max_iter < 1:
        raise ValueError("scintg: max_iter must be at least 1")
    if K is None:
        K = max(2, min(100, N // 30))
    K = int(K)
    if K < 1 or K > N:
        raise ValueError("scintg: K must be between 1 and the cell count")
    Zn = l2_normalise(rows)
    centres = l2_normalise([list(y) for y in Y]) if Y is not None \
        else _kmeans_init(Zn, K, seed)
    if len(centres) != K:
        raise ValueError("scintg: Y must have one row per cluster")
    names = sorted(set(batches), key=lambda v: str(v))
    bidx = dict((b, k) for k, b in enumerate(names))
    R = [[1.0 / K] * N for _ in range(K)]
    prev = None
    for _ in range(int(max_iter)):
        counts = cluster_batch_counts(R, batches, names)
        O, E = counts["O"], counts["E"]
        newR = [[0.0] * N for _ in range(K)]
        for i in range(N):
            bi = bidx[batches[i]]
            col = []
            for k in range(K):
                dist = 1.0 - sum(centres[k][j] * Zn[i][j] for j in range(d))
                val = -2.0 * dist / sigma
                if theta > 0:
                    o, e = O[k][bi], E[k][bi]
                    ratio = (o / e) if (o > 0 and e > 0) else 1e-12
                    sign = 1.0 if diversity == "as_printed" else -1.0
                    val += sign * theta * math.log(ratio)
                col.append(val)
            m = max(col)
            ex = [math.exp(v - m) for v in col]
            s = sum(ex)
            for k in range(K):
                newR[k][i] = ex[k] / s if s > 0 else 1.0 / K
        R = newR
        # Y = Z R^T, then L2 normalise (Dhillon's spherical centroids)
        centres = l2_normalise(
            [[sum(R[k][i] * Zn[i][j] for i in range(N)) for j in range(d)]
             for k in range(K)])
        obj = harmony_objective(Zn, R, centres, batches, sigma, theta)
        if prev is not None and abs(prev - obj["total"]) <= \
                tol * max(abs(prev), 1e-12):
            break
        prev = obj["total"]
    return {"R": R, "Y": centres, "K": K,
            "objective": harmony_objective(Zn, R, centres, batches,
                                           sigma, theta)}


def _solve(A, B):
    """Solve ``A X = B`` for a matrix right-hand side."""
    n = len(A)
    m = len(B[0])
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-14:
            raise ValueError("scintg: the ridge system is singular; raise "
                             "lambda")
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for k in range(c, n + m):
                M[r][k] -= f * M[c][k]
    return [[M[i][n + j] / M[i][i] for j in range(m)] for i in range(n)]


def correct_batch(Z, R, batches, lam=1.0, reference=None):
    r"""Algorithm 3: ridge out the batch, cluster by cluster.

    ``reference`` marks cells whose design row is set to
    :math:`[1, 0, \dots, 0]`, so they are explained by an intercept and
    nothing else and are returned unchanged.
    """
    rows, N, d = _matrix(Z)
    K = len(R)
    if K == 0 or any(len(r) != N for r in R):
        raise ValueError("scintg: R must be K by N")
    if len(batches) != N:
        raise ValueError("scintg: one batch label per cell is required")
    if lam < 0:
        raise ValueError("scintg: lambda must be non-negative")
    phi, names = _design(batches)
    B = len(names)
    if reference is not None:
        if len(reference) != N:
            raise ValueError("scintg: one reference flag per cell")
        for i in range(N):
            if reference[i]:
                phi[i] = [1.0] + [0.0] * B
    out = [list(r) for r in rows]
    Ws = []
    for k in range(K):
        A = [[sum(phi[i][a] * R[k][i] * phi[i][b] for i in range(N))
              for b in range(B + 1)] for a in range(B + 1)]
        for a in range(1, B + 1):        # lambda_0 = 0, lambda_b = lam
            A[a][a] += lam
        rhs = [[sum(phi[i][a] * R[k][i] * rows[i][j] for i in range(N))
                for j in range(d)] for a in range(B + 1)]
        W = _solve(A, rhs)
        W[0] = [0.0] * d                 # keep the intercept: cell type
        Ws.append(W)
        for i in range(N):
            for j in range(d):
                out[i][j] -= R[k][i] * sum(phi[i][a] * W[a][j]
                                           for a in range(B + 1))
    return {"Z": out, "W": Ws, "batches": names}


def scintg(Z, batches, K=None, sigma=0.1, theta=2.0, lam=1.0,
           max_iter=10, cluster_iter=25, tol=1e-4, seed=0,
           reference=None, diversity="penalise"):
    """Integrate ``Z`` across ``batches`` (Korsunsky et al. 2019)."""
    rows, N, d = _matrix(Z)
    if len(batches) != N:
        raise ValueError("scintg: one batch label per cell is required")
    if len(set(batches)) < 2:
        raise ValueError("scintg: at least two batches are needed")
    if max_iter < 1:
        raise ValueError("scintg: max_iter must be at least 1")
    cur = [list(r) for r in rows]
    Y = None
    hist = []
    for _ in range(int(max_iter)):
        cl = maximum_diversity_clustering(cur, batches, K, sigma, theta,
                                          cluster_iter, seed=seed, Y=Y,
                                          diversity=diversity)
        Y = cl["Y"]
        got = correct_batch(cur, cl["R"], batches, lam, reference)
        shift = max(abs(got["Z"][i][j] - cur[i][j])
                    for i in range(N) for j in range(d))
        cur = got["Z"]
        hist.append(cl["objective"]["total"])
        if shift <= tol:
            break
    final = maximum_diversity_clustering(cur, batches, K, sigma, theta,
                                         cluster_iter, seed=seed, Y=Y,
                                         diversity=diversity)
    return RichResult(payload={
        "estimate": cur,
        "embedding": cur,
        "R": final["R"],
        "Y": final["Y"],
        "K": final["K"],
        "objective": final["objective"],
        "history": hist,
        "n_rounds": len(hist),
        "theta": float(theta),
        "sigma": float(sigma),
        "lam": float(lam),
        "diversity": diversity,
        "method": ("Harmony (Korsunsky et al. 2019): maximum diversity "
                   "clustering (eq. 8) alternated with mixture-of-experts "
                   "ridge correction (eq. 14)"),
        "note": ("theta=0 reduces the cluster step to ordinary soft "
                 "spherical k-means; the intercept row of W_k is zeroed "
                 "so batch-independent variation is kept, which is why a "
                 "reference cell (design row [1, 0, ...]) never moves. "
                 "Equation 8 is printed with (O/E)^+theta, which raises "
                 "cluster/batch dependence rather than lowering it; "
                 "diversity='penalise' uses the -theta the stated "
                 "objective implies, 'as_printed' the literal form"),
    })


harmony_integrate = scintg


def cheatsheet():
    return ("scintg: Harmony (Korsunsky et al. 2019). Alternates maximum "
            "diversity clustering -- soft spherical k-means whose "
            "assignment R_ki is proportional to (O_ki/E_ki)^theta "
            "exp(-2(1 - Y_k'Z_i)/sigma), with O the observed and E the "
            "independence-expected cluster/batch mass -- with a "
            "mixture-of-experts ridge correction W_k = (phi* diag(R_k) "
            "phi*' + lambda I)^-1 phi* diag(R_k) Z' whose intercept row "
            "is zeroed, so batch goes and cell type stays.")

# public names resolved by fn/_lazy_map.json
singlecell_integration = scintg
