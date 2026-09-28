# morie.fn -- function file (rootcoder007/morie)
"""LU/Cholesky simulation toolkit: banded and incomplete Cholesky factors, Wendland covariance
tapering, linear-model-of-coregionalisation co-simulation, sensitivity of realisations to range and
sill under common random numbers, nested-component decomposition, iterative refinement of
covariance solves, and block-sequential LU simulation of large grids."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_normal
from .krgsys import kriging_covariance

__all__ = [
    "banded_cholesky",
    "incomplete_cholesky",
    "wendland_taper",
    "tapered_simulate",
    "lmc_cosimulate",
    "simulation_sensitivity",
    "nested_decomposition",
    "refined_solve",
    "block_lu_simulate",
]


def _mat(A):
    return [[float(v) for v in r] for r in (A.tolist() if hasattr(A, "tolist") else A)]


def _pts(P):
    return [tuple(float(v) for v in r) for r in (P.tolist() if hasattr(P, "tolist") else P)]


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("matrix is not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _cov(P, Q, model):
    return [[kriging_covariance(math.dist(p, q), model) for q in Q] for p in P]


def banded_cholesky(A, bandwidth: int) -> RichResult:
    r"""Cholesky factor of a symmetric positive-definite band matrix (``A_ij = 0`` for ``|i - j| > bandwidth``) in ``O(n b^2)``.

    The factor keeps the band (Golub and Van Loan 2013, algorithm 4.3.5);
    for simulation along a line with a compactly supported covariance
    (range below ``bandwidth`` spacings) it equals the full Cholesky factor.
    Returns the lower factor ``L`` and the entries outside the band that
    were ignored (their maximum magnitude).

    References
    ----------
    Golub, G. H. and Van Loan, C. F. (2013). *Matrix Computations*, 4th edn.
    Johns Hopkins University Press.

    Examples
    --------
    >>> banded_cholesky([[4, 2, 0], [2, 5, 2], [0, 2, 5]], 1).L
    [[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 2.0]]
    """
    M = _mat(A)
    n = len(M)
    b = int(bandwidth)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(max(0, i - b), i + 1):
            s = M[i][j] - ssum(L[i][k] * L[j][k] for k in range(max(0, i - b, j - b), j))
            if i == j:
                if s <= 0:
                    raise ValueError("matrix is not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    off = max((abs(M[i][j]) for i in range(n) for j in range(n) if abs(i - j) > b), default=0.0)
    return RichResult(payload={"L": L, "bandwidth": b, "ignored_max": off})


def incomplete_cholesky(A, *, drop_tol: float = 0.0) -> RichResult:
    r"""Incomplete Cholesky factor IC(0): Cholesky restricted to the sparsity pattern of ``A`` (entries with ``|A_ij| > drop_tol``).

    Fill-in outside the pattern is discarded (Meijerink and Van der Vorst
    1977), giving a cheap approximate factor for preconditioning or for
    approximate simulation of tapered covariances. Returns ``L``, the
    Frobenius error ``||A - L L'||`` relative to ``||A||`` and the number of
    stored entries. On a pattern without fill-in (e.g. a band) it equals the
    exact factor.

    References
    ----------
    Meijerink, J. A. and Van der Vorst, H. A. (1977). An iterative solution
    method for linear systems of which the coefficient matrix is a
    symmetric M-matrix. *Mathematics of Computation*, 31(137), 148-162.

    Examples
    --------
    >>> round(incomplete_cholesky([[4, 2, 0], [2, 5, 2], [0, 2, 5]]).relative_error, 15)
    0.0
    """
    M = _mat(A)
    n = len(M)
    keep = [[abs(M[i][j]) > drop_tol for j in range(n)] for i in range(n)]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            if not keep[i][j] and i != j:
                continue
            s = M[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("incomplete factorisation broke down (non-positive pivot)")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    R = [[M[i][j] - ssum(L[i][k] * L[j][k] for k in range(min(i, j) + 1)) for j in range(n)] for i in range(n)]
    nrm = math.sqrt(ssum(v * v for r in M for v in r))
    return RichResult(
        payload={
            "L": L,
            "relative_error": math.sqrt(ssum(v * v for r in R for v in r)) / nrm,
            "nnz": sum(1 for i in range(n) for j in range(i + 1) if L[i][j] != 0.0),
        }
    )


def wendland_taper(d, theta: float, *, dimension: int = 2, k: int = 1) -> list:
    r"""Wendland's compactly supported correlation ``psi_{l,k}(d / theta)``, ``l = floor(dimension / 2) + k + 1``, as ``fields::Wendland``.

    ``k = 0``: ``(1 - r)^l``; ``k = 1``: ``(1 - r)^{l+1} ((l + 1) r + 1)``;
    ``k = 2``: ``(1 - r)^{l+2} ((l^2 + 4l + 3) r^2 + (3l + 6) r + 3) / 3``;
    ``k = 3``: ``(1 - r)^{l+3} ((l^3 + 9l^2 + 23l + 15) r^3 + (6l^2 + 36l + 45)
    r^2 + (15l + 45) r + 15) / 15``; zero for ``r >= 1``. Positive definite in
    ``R^dimension`` and ``2k`` times differentiable (Wendland 1995), the
    usual taper for sparse covariance matrices (Furrer, Genton and Nychka
    2006).

    References
    ----------
    Wendland, H. (1995). Piecewise polynomial, positive definite and
    compactly supported radial functions of minimal degree. *Advances in
    Computational Mathematics*, 4(1), 389-396.
    Furrer, R., Genton, M. G. and Nychka, D. (2006). Covariance tapering for
    interpolation of large spatial datasets. *Journal of Computational and
    Graphical Statistics*, 15(3), 502-523.

    Examples
    --------
    >>> wendland_taper([0.0, 0.5, 1.0], 1.0, dimension=2, k=1)
    [1.0, 0.1875, 0.0]
    """
    l_ = dimension // 2 + k + 1
    out = []
    for v in d:
        r = abs(float(v)) / theta
        if r >= 1:
            out.append(0.0)
            continue
        if k == 0:
            p = 1.0
        elif k == 1:
            p = (l_ + 1) * r + 1
        elif k == 2:
            p = ((l_ * l_ + 4 * l_ + 3) * r * r + (3 * l_ + 6) * r + 3) / 3
        elif k == 3:
            p = (
                (l_**3 + 9 * l_ * l_ + 23 * l_ + 15) * r**3
                + (6 * l_ * l_ + 36 * l_ + 45) * r * r
                + (15 * l_ + 45) * r
                + 15
            ) / 15
        else:
            raise ValueError("k must be 0, 1, 2 or 3")
        out.append((1 - r) ** (l_ + k) * p)
    return out


def tapered_simulate(
    coords, model, theta: float, *, k: int = 1, nsim: int = 1, seed: int = 1, mean: float = 0.0
) -> RichResult:
    r"""Simulation from the tapered covariance ``C(h) psi(h / theta)`` (a valid, sparse covariance), by Cholesky.

    Returns the realisations (Philox stream ``s`` for realisation ``s``,
    as :func:`~morie.fn.zschl.chol_sim`), the tapered covariance matrix and
    its share of zero entries.

    References
    ----------
    Furrer, R., Genton, M. G. and Nychka, D. (2006). Covariance tapering for
    interpolation of large spatial datasets. *Journal of Computational and
    Graphical Statistics*, 15(3), 502-523.

    Examples
    --------
    >>> r = tapered_simulate([(0, 0), (5, 0)], {"model": "Exp", "psill": 1.0, "range": 1.0}, 2.0)
    >>> r.cov[0][1], r.sparsity
    (0.0, 0.5)
    """
    P = _pts(coords)
    n = len(P)
    d = len(P[0])
    C = [
        [
            kriging_covariance(math.dist(P[i], P[j]), model)
            * wendland_taper([math.dist(P[i], P[j])], theta, dimension=d, k=k)[0]
            for j in range(n)
        ]
        for i in range(n)
    ]
    L = _chol(C)
    sims = []
    for s in range(nsim):
        e = [float(v) for v in random_normal(n, seed=seed, stream=s)]
        sims.append([mean + ssum(L[i][t] * e[t] for t in range(i + 1)) for i in range(n)])
    return RichResult(
        payload={"simulations": sims, "cov": C, "sparsity": sum(1 for r in C for v in r if v == 0.0) / (n * n)}
    )


def lmc_cosimulate(coords, components, *, nsim: int = 1, seed: int = 1, means=None) -> RichResult:
    r"""Joint Gaussian simulation of ``p`` coregionalised fields under a linear model of coregionalisation.

    ``components`` is a list of ``(B_k, model_k)``: ``B_k`` a ``p x p``
    positive semi-definite coregionalisation matrix and ``model_k`` a
    unit-sill structure, so ``Cov(Z_a(s), Z_b(s')) = sum_k B_k[a][b]
    rho_k(|s - s'|)`` (Goovaerts 1997). The joint covariance (fields
    stacked, field-major) is factorised by Cholesky; returns realisations as
    ``[field][location]`` per simulation and the joint covariance.

    References
    ----------
    Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*.
    Oxford University Press.

    Examples
    --------
    >>> comps = [([[1.0, 0.8], [0.8, 1.0]], {"model": "Exp", "psill": 1.0, "range": 1.0})]
    >>> r = lmc_cosimulate([(0, 0), (1, 0)], comps)
    >>> round(r.cov[0][2], 12)
    0.8
    """
    P = _pts(coords)
    n = len(P)
    p = len(components[0][0])
    N = n * p
    C = [[0.0] * N for _ in range(N)]
    for B, m in components:
        rho = _cov(P, P, m)
        for a in range(p):
            for b in range(p):
                for i in range(n):
                    for j in range(n):
                        C[a * n + i][b * n + j] += float(B[a][b]) * rho[i][j]
    L = _chol(C)
    mu = [0.0] * p if means is None else [float(v) for v in means]
    sims = []
    for s in range(nsim):
        e = [float(v) for v in random_normal(N, seed=seed, stream=s)]
        x = [ssum(L[i][t] * e[t] for t in range(i + 1)) for i in range(N)]
        sims.append([[mu[a] + x[a * n + i] for i in range(n)] for a in range(p)])
    return RichResult(payload={"simulations": sims, "cov": C})


def simulation_sensitivity(coords, model, *, ranges=None, sills=None, seed: int = 1) -> RichResult:
    r"""Realisations of one Gaussian field under several ranges or sills with common random numbers.

    The same standard-normal vector ``e`` (Philox stream 0) drives ``L_theta
    e`` for every parameter value, so differences between realisations are
    due to the parameter only. For a sill change the realisation scales by
    ``sqrt(sill / sill_0)`` exactly (variance scaling). Returns the
    realisations, their sample variances and the correlation of each
    realisation with the first.

    References
    ----------
    Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
    Springer, section 4.1 (common random numbers).

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    >>> r = simulation_sensitivity([(0, 0), (1, 0), (2, 0)], m, sills=[1.0, 4.0])
    >>> [round(b / a, 12) for a, b in zip(*r.simulations)]
    [2.0, 2.0, 2.0]
    """
    P = _pts(coords)
    n = len(P)
    e = [float(v) for v in random_normal(n, seed=seed, stream=0)]
    params = []
    if ranges is not None:
        params = [{**model, "range": float(r)} for r in ranges]
    elif sills is not None:
        params = [{**model, "psill": float(s)} for s in sills]
    else:
        raise ValueError("give ranges or sills")
    sims = []
    for m in params:
        L = _chol(_cov(P, P, m))
        sims.append([ssum(L[i][t] * e[t] for t in range(i + 1)) for i in range(n)])

    def cor(a, b):
        ma, mb = ssum(a) / n, ssum(b) / n
        return ssum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(
            ssum((x - ma) ** 2 for x in a) * ssum((y - mb) ** 2 for y in b)
        )

    var = [ssum((v - ssum(s) / n) ** 2 for v in s) / (n - 1) for s in sims]
    return RichResult(payload={"simulations": sims, "variance": var, "correlation": [cor(sims[0], s) for s in sims]})


def nested_decomposition(coords, components, *, seed: int = 1) -> RichResult:
    r"""Simulate a nested covariance ``sum_k C_k`` as the sum of independent component fields, returning each component.

    Component ``k`` uses its own Cholesky factor and Philox stream ``k``;
    the total field has covariance ``sum_k C_k``. Returns the component
    fields, the total and each component's share of the total sample
    variance (a variance decomposition of the realisation; the nominal
    shares are the sills over the total sill).

    References
    ----------
    Chiles, J.-P. and Delfiner, P. (2012). *Geostatistics: Modeling Spatial
    Uncertainty*, 2nd edn. Wiley, section 7.3.

    Examples
    --------
    >>> comps = [{"model": "Nug", "psill": 0.5}, {"model": "Exp", "psill": 1.5, "range": 2.0}]
    >>> r = nested_decomposition([(0, 0), (1, 0), (3, 0)], comps)
    >>> r.nominal_share
    [0.25, 0.75]
    """
    P = _pts(coords)
    n = len(P)
    fields = []
    for k, c in enumerate(components):
        L = _chol(_cov(P, P, c))
        e = [float(v) for v in random_normal(n, seed=seed, stream=k)]
        fields.append([ssum(L[i][t] * e[t] for t in range(i + 1)) for i in range(n)])
    total = [ssum(f[i] for f in fields) for i in range(n)]

    def var(v):
        m = ssum(v) / n
        return ssum((x - m) ** 2 for x in v) / (n - 1)

    vt = var(total)
    sills = [kriging_covariance(0.0, c) for c in components]
    return RichResult(
        payload={
            "components": fields,
            "total": total,
            "share": [var(f) / vt for f in fields],
            "nominal_share": [s / ssum(sills) for s in sills],
        }
    )


def refined_solve(A, b, *, iterations: int = 3) -> RichResult:
    r"""Solve ``A x = b`` (symmetric positive definite) by Cholesky with iterative refinement.

    After ``x_0 = L^-T L^-1 b`` each step computes the residual ``r = b - A
    x`` (sums compensated) and corrects ``x <- x + L^-T L^-1 r`` (Wilkinson
    1963; Higham 2002, chapter 12). In working precision this drives the
    solution to a small componentwise backward error, the level at which the
    residual stalls; it cannot beat the conditioning of ``A`` (e.g.
    Gaussian covariance kriging systems). Returns the solution and the
    residual norms per step.

    References
    ----------
    Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*,
    2nd edn. SIAM.

    Examples
    --------
    >>> [round(v, 12) for v in refined_solve([[4.0, 2.0], [2.0, 3.0]], [2.0, 1.0]).x]
    [0.5, 0.0]
    """
    M = _mat(A)
    bv = [float(v) for v in b]
    n = len(M)
    L = _chol(M)

    def solve(r):
        y = [0.0] * n
        for i in range(n):
            y[i] = (r[i] - ssum(L[i][k] * y[k] for k in range(i))) / L[i][i]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - ssum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
        return x

    x = solve(bv)
    hist = []
    for _ in range(iterations + 1):
        r = [bv[i] - ssum(M[i][j] * x[j] for j in range(n)) for i in range(n)]
        hist.append(math.sqrt(ssum(v * v for v in r)))
        if len(hist) > iterations:
            break
        d = solve(r)
        x = [a + c for a, c in zip(x, d)]
    return RichResult(payload={"x": x, "residual_norms": hist})


def block_lu_simulate(
    xs, ys, model, *, block: int = 4, halo: float = 1.0, seed: int = 1, mean: float = 0.0
) -> RichResult:
    r"""Approximate LU simulation of a large regular grid by blocks, each conditioned on the already simulated nodes within ``halo``.

    The grid ``xs x ys`` (row-major, ``y`` outer) is cut into ``block x
    block`` tiles simulated in raster order; each tile is drawn from its
    Gaussian distribution conditional on previously simulated nodes closer
    than ``halo`` to the tile (simple-kriging conditional mean and
    covariance, Philox stream = tile index). Exact within a tile and for
    covariances whose range is below ``halo``; otherwise an approximation
    trading accuracy for ``O(block^6)`` per tile instead of ``O(N^3)``
    (a moving-neighbourhood LU; Davis 1987; Dimitrakopoulos and Luo 2004).

    References
    ----------
    Davis, M. W. (1987). Production of conditional simulations via the LU
    triangular decomposition of the covariance matrix. *Mathematical
    Geology*, 19(2), 91-98.
    Dimitrakopoulos, R. and Luo, X. (2004). Generalized sequential Gaussian
    simulation on group size nu and screen-effect approximations for large
    field simulations. *Mathematical Geology*, 36(5), 567-591.

    Examples
    --------
    >>> r = block_lu_simulate([0, 1, 2, 3], [0, 1], {"model": "Exp", "psill": 1.0, "range": 1.0}, block=2)
    >>> len(r.field), len(r.field[0]), r.n_blocks
    (2, 4, 2)
    """
    X = [float(v) for v in xs]
    Y = [float(v) for v in ys]
    nx, ny = len(X), len(Y)
    val = {}
    tiles = [(bj, bi) for bj in range(0, ny, block) for bi in range(0, nx, block)]
    for t, (bj, bi) in enumerate(tiles):
        nodes = [(j, i) for j in range(bj, min(bj + block, ny)) for i in range(bi, min(bi + block, nx))]
        pts = [(X[i], Y[j]) for j, i in nodes]
        cond = [
            (key, (X[key[1]], Y[key[0]]))
            for key in val
            if min(math.dist((X[key[1]], Y[key[0]]), p) for p in pts) < halo
        ]
        cond.sort()
        e = [float(v) for v in random_normal(len(pts), seed=seed, stream=t)]
        Cxx = _cov(pts, pts, model)
        if cond:
            D = [c[1] for c in cond]
            zc = [val[c[0]] - mean for c in cond]
            Ci = [[float(v) for v in r] for r in inverse(_cov(D, D, model))]
            Cxd = _cov(pts, D, model)
            W = [[ssum(Cxd[i][a] * Ci[a][b] for a in range(len(D))) for b in range(len(D))] for i in range(len(pts))]
            mu = [mean + ssum(W[i][b] * zc[b] for b in range(len(D))) for i in range(len(pts))]
            S = [
                [Cxx[i][j] - ssum(W[i][b] * Cxd[j][b] for b in range(len(D))) for j in range(len(pts))]
                for i in range(len(pts))
            ]
            L = _chol_psd(S)
        else:
            mu = [mean] * len(pts)
            L = _chol(Cxx)
        draw = [mu[i] + ssum(L[i][k] * e[k] for k in range(i + 1)) for i in range(len(pts))]
        for key, v in zip(nodes, draw):
            val[key] = v
    return RichResult(payload={"field": [[val[(j, i)] for i in range(nx)] for j in range(ny)], "n_blocks": len(tiles)})


def _chol_psd(A):
    """Cholesky of a positive semi-definite matrix (zero columns where the pivot vanishes)."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    scale = max(abs(A[i][i]) for i in range(n)) if n else 0.0
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][i] = math.sqrt(s) if s > 1e-12 * scale else 0.0
            else:
                L[i][j] = s / L[j][j] if L[j][j] > 0 else 0.0
    return L


def cheatsheet() -> str:
    return (
        "banded_cholesky / incomplete_cholesky / wendland_taper / tapered_simulate / lmc_cosimulate / "
        "simulation_sensitivity / nested_decomposition / refined_solve / block_lu_simulate -> LU/Cholesky toolkit."
    )
