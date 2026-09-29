# morie.fn -- function file (rootcoder007/morie)
"""Geostatistical simulation on the Philox stream: sequential Gaussian simulation with point and
block data, conditional ensembles, p-field simulation, conditional LMC co-simulation of p > 2
variables, collocated co-simulation, sequential indicator simulation with Markov-Bayes soft
data, SNESIM multiple-point simulation and simulated annealing to a target variogram."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform

__all__ = [
    "sgs_block_simulate",
    "conditional_ensemble",
    "pfield_simulate",
    "lmc_conditional_simulate",
    "collocated_cosimulate",
    "sis_markov_bayes",
    "snesim_simulate",
    "annealing_simulate",
]


def _rho(h, model):
    a = model["range"]
    kind = model.get("model", "Exp")
    if kind == "Sph":
        return 1.0 - 1.5 * h / a + 0.5 * (h / a) ** 3 if h < a else 0.0
    if kind == "Gau":
        return math.exp(-((h / a) ** 2))
    return math.exp(-h / a)


def _cov(p, q, model):
    h = math.sqrt(ssum((a - b) ** 2 for a, b in zip(p, q)))
    if h == 0.0:
        return model["sill"] + model.get("nugget", 0.0)
    return model["sill"] * _rho(h, model)


def _perm(n, seed, stream):
    u = random_uniform(n, seed=seed, stream=stream) if n else []
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(math.floor(float(u[i]) * (i + 1)))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _chol(C):
    n = len(C)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = C[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(max(s, 0.0)) if i == j else (s / L[j][j] if L[j][j] > 0 else 0.0)
    return L


def sgs_block_simulate(
    data_coords, data_values, targets, model, *, mean: float = 0.0, k: int = 12, blocks=None, seed: int = 0
) -> RichResult:
    r"""Sequential Gaussian simulation with point data and (optionally) block-average data.

    Along a random path (Philox stream 0) each target is drawn from
    ``N(m_SK, s2_SK)`` by simple kriging with known ``mean`` from its ``k``
    nearest informed points (data and previously simulated nodes; ties by
    order of entry) plus every block datum. ``blocks`` lists
    ``(discretisation_points, value)``; point-to-block and block-to-block
    covariances are averages over the discretisation (Journel and
    Huijbregts 1978), as in block sequential simulation (Liu and Journel
    2009). Normal deviates come from stream 1 in path order. Data are
    assumed Gaussian (transform beforehand).

    References
    ----------
    Deutsch, C. V. and Journel, A. G. (1998). GSLIB, 2nd ed., section V.2.3.
    Liu, Y. and Journel, A. G. (2009). A package for geostatistical
    integration of coarse and fine scale data. Computers & Geosciences 35, 527-547.

    Examples
    --------
    >>> m = {"model": "Exp", "sill": 1.0, "range": 2.0}
    >>> r = sgs_block_simulate([(0.0, 0.0)], [1.5], [(0.0, 0.0), (5.0, 0.0)], m, seed=1)
    >>> r.simulated[0]
    1.5
    """
    blocks = blocks or []
    known = [(tuple(float(v) for v in c), float(z)) for c, z in zip(data_coords, data_values)]
    T = [tuple(float(v) for v in c) for c in targets]
    path = _perm(len(T), seed, 0)
    zn = [float(v) for v in random_normal(len(T), seed=seed, stream=1)] if T else []
    bpts = [[tuple(float(v) for v in p) for p in b[0]] for b in blocks]
    bval = [float(b[1]) for b in blocks]
    BB = [
        [
            ssum(_cov(p, q, model) for p in bpts[a] for q in bpts[b]) / (len(bpts[a]) * len(bpts[b]))
            for b in range(len(blocks))
        ]
        for a in range(len(blocks))
    ]
    out = [0.0] * len(T)
    for step, t in enumerate(path):
        x = T[t]
        exact = [z for c, z in known if c == x]
        if exact:
            out[t] = exact[0]
            continue
        order = sorted(range(len(known)), key=lambda q: (math.dist(known[q][0], x), q))[:k]
        pts = [known[q][0] for q in order]
        vals = [known[q][1] for q in order]
        n1 = len(pts)
        m = n1 + len(blocks)
        C = [[0.0] * m for _ in range(m)]
        c0 = [0.0] * m
        for a in range(n1):
            for b in range(n1):
                C[a][b] = _cov(pts[a], pts[b], model)
            for b in range(len(blocks)):
                v = ssum(_cov(pts[a], q, model) for q in bpts[b]) / len(bpts[b])
                C[a][n1 + b] = C[n1 + b][a] = v
            c0[a] = _cov(pts[a], x, model)
        for a in range(len(blocks)):
            for b in range(len(blocks)):
                C[n1 + a][n1 + b] = BB[a][b]
            c0[n1 + a] = ssum(_cov(q, x, model) for q in bpts[a]) / len(bpts[a])
        resid = [v - mean for v in vals] + [v - mean for v in bval]
        if m:
            lam = solve(C, c0)
            mu = mean + ssum(a * b for a, b in zip(lam, resid))
            var = max(_cov(x, x, model) - ssum(a * b for a, b in zip(lam, c0)), 0.0)
        else:
            mu, var = mean, _cov(x, x, model)
        out[t] = mu + math.sqrt(var) * zn[step]
        known.append((x, out[t]))
    blk = []
    for b in bpts:
        vals = [out[T.index(p)] for p in b if p in T]
        blk.append(ssum(vals) / len(vals) if vals else float("nan"))
    return RichResult(payload={"simulated": out, "path": path, "block_means": blk})


def _q7(s, prob):
    h = (len(s) - 1) * prob
    lo = int(math.floor(h))
    hi = min(lo + 1, len(s) - 1)
    w = h - lo
    return (1.0 - w) * s[lo] + w * s[hi] if w > 0 else s[lo]


def conditional_ensemble(
    data_coords,
    data_values,
    targets,
    model,
    *,
    mean: float = 0.0,
    k: int = 12,
    n_real: int = 20,
    probs=(0.1, 0.5, 0.9),
    seed: int = 0,
) -> RichResult:
    r"""Ensemble of conditional SGS realisations and its node-wise summaries.

    Realisation ``r`` uses seed ``seed + r``; returns the E-type mean, the
    conditional variance (divisor ``R``), the type 7 quantiles ``probs`` and
    all realisations (Journel and Deutsch 1993; Goovaerts 1997).

    References
    ----------
    Goovaerts, P. (1997). Geostatistics for Natural Resources Evaluation,
    ch. 8. Journel, A. G. and Deutsch, C. V. (1993). Entropy and spatial
    disorder. Math. Geology 25, 329-355.

    Examples
    --------
    >>> m = {"model": "Exp", "sill": 1.0, "range": 2.0}
    >>> r = conditional_ensemble([(0.0, 0.0)], [1.0], [(0.0, 0.0), (1.0, 0.0)], m, n_real=5)
    >>> r.etype[0], r.variance[0]
    (1.0, 0.0)
    """
    reals = [
        sgs_block_simulate(data_coords, data_values, targets, model, mean=mean, k=k, seed=seed + r).simulated
        for r in range(n_real)
    ]
    n = len(targets)
    et = [ssum(reals[r][i] for r in range(n_real)) / n_real for i in range(n)]
    var = [ssum((reals[r][i] - et[i]) ** 2 for r in range(n_real)) / n_real for i in range(n)]
    qs = [[_q7(sorted(reals[r][i] for r in range(n_real)), p) for i in range(n)] for p in probs]
    return RichResult(payload={"etype": et, "variance": var, "quantiles": qs, "realisations": reals})


def pfield_simulate(krige_mean, krige_sd, coords, model, *, seed: int = 0) -> list:
    r"""P-field simulation (Froidevaux 1993; Srivastava 1992).

    An unconditional unit-variance Gaussian field ``Y`` with the correlogram
    of ``model`` (Cholesky, Philox stream 0) supplies correlated
    probabilities ``p = Phi(Y)`` that are read from Gaussian local ccdfs:
    ``z(u) = m(u) + s(u) Y(u)`` with kriging mean ``m`` and standard deviation ``s``.

    References
    ----------
    Froidevaux, R. (1993). Probability field simulation. In Geostatistics
    Troia '92, 73-84. Srivastava, R. M. (1992). Reservoir characterization
    with probability field simulation. SPE 24753.

    Examples
    --------
    >>> m = {"model": "Exp", "sill": 1.0, "range": 1.0}
    >>> z = pfield_simulate([1.0, 2.0], [0.0, 0.0], [(0.0, 0.0), (1.0, 0.0)], m)
    >>> z
    [1.0, 2.0]
    """
    pts = [tuple(float(v) for v in c) for c in coords]
    n = len(pts)
    unit = dict(model, sill=1.0, nugget=0.0)
    L = _chol([[_cov(pts[a], pts[b], unit) for b in range(n)] for a in range(n)])
    z = [float(v) for v in random_normal(n, seed=seed, stream=0)]
    Y = [ssum(L[i][j] * z[j] for j in range(i + 1)) for i in range(n)]
    return [float(m) + float(s) * y for m, s, y in zip(krige_mean, krige_sd, Y)]


def lmc_conditional_simulate(data_coords, data_values, targets, components, *, means=None, seed: int = 0) -> RichResult:
    r"""Conditional joint simulation of ``p`` coregionalised Gaussian fields (any ``p``, including ``p > 2``).

    ``components`` is a list of ``(B_k, model_k)`` (linear model of
    coregionalisation, unit-sill structures). The observed entries of
    ``data_values[location][variable]`` (``None``/NaN missing) and the
    target values are jointly Gaussian; the targets are drawn from
    ``N(mu_T + S_TD S_DD^(-1) (z_D - mu_D), S_TT - S_TD S_DD^(-1) S_DT)`` by
    Cholesky (Philox stream 0). Returns ``simulated[location][variable]``.

    References
    ----------
    Goovaerts, P. (1997). Geostatistics for Natural Resources Evaluation,
    ch. 8. Chiles, J.-P. and Delfiner, P. (2012). Geostatistics, 2nd ed., ch. 7.

    Examples
    --------
    >>> comps = [([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]], {"model": "Exp", "sill": 1.0, "range": 2.0})]
    >>> r = lmc_conditional_simulate([(0.0, 0.0)], [[1.0, 0.5, None]], [(0.0, 0.0)], comps)
    >>> r.simulated[0][:2]
    [1.0, 0.5]
    """
    p = len(components[0][0])
    mu = [0.0] * p if means is None else [float(v) for v in means]
    D = [
        (tuple(float(v) for v in c), v, float(z))
        for c, row in zip(data_coords, data_values)
        for v, z in enumerate(row)
        if z is not None and z == z
    ]
    Tn = [(tuple(float(x) for x in c), v) for c in targets for v in range(p)]

    def cc(a, b):
        return ssum(B[a[1]][b[1]] * (_rho(math.dist(a[0], b[0]), m) if a[0] != b[0] else 1.0) for B, m in components)

    known = {(c, v): z for c, v, z in D}
    free = [t for t in Tn if t not in known]
    Dl = [(c, v) for c, v, _ in D]
    nd = len(Dl)
    Sdd = [[cc(a, b) for b in Dl] for a in Dl]
    Std = [[cc(a, b) for b in Dl] for a in free]
    resid = [z - mu[v] for _, v, z in D]
    if nd:
        alpha = solve(Sdd, resid)
        cond_mu = [mu[t[1]] + ssum(Std[i][j] * alpha[j] for j in range(nd)) for i, t in enumerate(free)]
        W = [solve(Sdd, [Std[i][j] for j in range(nd)]) for i in range(len(free))]
        S = [
            [cc(a, b) - ssum(Std[i][q] * W[j][q] for q in range(nd)) for j, b in enumerate(free)]
            for i, a in enumerate(free)
        ]
    else:
        cond_mu = [mu[t[1]] for t in free]
        S = [[cc(a, b) for b in free] for a in free]
    L = _chol(S)
    z = [float(v) for v in random_normal(len(free), seed=seed, stream=0)] if free else []
    sim = {t: cond_mu[i] + ssum(L[i][j] * z[j] for j in range(i + 1)) for i, t in enumerate(free)}
    vals = [
        [known.get((tuple(float(x) for x in c), v), sim.get((tuple(float(x) for x in c), v))) for v in range(p)]
        for c in targets
    ]
    return RichResult(payload={"simulated": vals})


def collocated_cosimulate(
    data_coords, data_values, targets, secondary, model, rho: float, *, k: int = 12, seed: int = 0
) -> RichResult:
    r"""Sequential Gaussian co-simulation by collocated cokriging under Markov model 1 (Almeida and Journel 1994).

    Standardised primary ``Z`` (mean 0, unit variance) with correlogram
    ``model`` and secondary ``Y`` known at every target (``secondary``);
    ``C_ZY(h) = rho C_Z(h)``. Each target along the Philox path uses simple
    cokriging with its ``k`` nearest informed primary values plus the
    collocated secondary datum.

    References
    ----------
    Almeida, A. S. and Journel, A. G. (1994). Joint simulation of multiple
    variables with a Markov-type coregionalization model. Math. Geology 26,
    565-588. Xu, W. et al. (1992). Integrating seismic data in reservoir
    modeling: the collocated cokriging alternative. SPE 24742.

    Examples
    --------
    >>> m = {"model": "Exp", "sill": 1.0, "range": 2.0}
    >>> r = collocated_cosimulate([(0.0, 0.0)], [0.5], [(0.0, 0.0), (3.0, 0.0)], [0.4, -0.2], m, 0.7)
    >>> r.simulated[0]
    0.5
    """
    unit = dict(model, sill=1.0, nugget=0.0)
    known = [(tuple(float(v) for v in c), float(z)) for c, z in zip(data_coords, data_values)]
    T = [tuple(float(v) for v in c) for c in targets]
    path = _perm(len(T), seed, 0)
    zn = [float(v) for v in random_normal(len(T), seed=seed, stream=1)] if T else []
    out = [0.0] * len(T)
    for step, t in enumerate(path):
        x = T[t]
        exact = [z for c, z in known if c == x]
        if exact:
            out[t] = exact[0]
            continue
        order = sorted(range(len(known)), key=lambda q: (math.dist(known[q][0], x), q))[:k]
        pts = [known[q][0] for q in order]
        n1 = len(pts)
        C = [[_cov(pts[a], pts[b], unit) for b in range(n1)] + [rho * _cov(pts[a], x, unit)] for a in range(n1)]
        C.append([rho * _cov(x, pts[b], unit) for b in range(n1)] + [1.0])
        c0 = [_cov(pts[a], x, unit) for a in range(n1)] + [rho]
        lam = solve(C, c0)
        mu = ssum(lam[a] * known[order[a]][1] for a in range(n1)) + lam[n1] * float(secondary[t])
        var = max(1.0 - ssum(a * b for a, b in zip(lam, c0)), 0.0)
        out[t] = mu + math.sqrt(var) * zn[step]
        known.append((x, out[t]))
    return RichResult(payload={"simulated": out, "path": path})


def sis_markov_bayes(
    data_coords, data_categories, targets, proportions, models, *, k: int = 12, soft=None, B=None, seed: int = 0
) -> RichResult:
    r"""Sequential indicator simulation of a categorical variable, with optional Markov-Bayes soft data.

    Category ``c`` has global proportion ``p_c`` and indicator covariance
    ``models[c]``. Along a Philox path each target gets simple indicator
    kriging estimates ``p_c + sum lambda (i_c - p_c)`` from its ``k`` nearest
    informed nodes; with ``soft`` probabilities at the targets (``soft[t][c]``)
    the collocated soft datum enters by the Markov-Bayes model
    ``C_IY(h) = B_c C_I(h)``, ``C_Y(h) = B_c^2 C_I(h)`` for ``h > 0`` and
    ``C_Y(0) = B_c C_I(0)`` (Zhu and Journel 1993). Probabilities are
    clipped to [0, 1] and renormalised, and a uniform (stream 1) draws the category.

    References
    ----------
    Deutsch, C. V. and Journel, A. G. (1998). GSLIB, section V.3. Zhu, H.
    and Journel, A. G. (1993). Formatting and integrating soft data:
    stochastic imaging via the Markov-Bayes algorithm. In Geostatistics
    Troia '92, 1-12.

    Examples
    --------
    >>> m = {"model": "Sph", "sill": 0.25, "range": 3.0}
    >>> r = sis_markov_bayes([(0.0, 0.0)], [1], [(0.0, 0.0), (1.0, 0.0)], [0.5, 0.5], [m, m], seed=2)
    >>> r.simulated[0]
    1
    """
    K = len(proportions)
    known = [(tuple(float(v) for v in c), int(z)) for c, z in zip(data_coords, data_categories)]
    T = [tuple(float(v) for v in c) for c in targets]
    path = _perm(len(T), seed, 0)
    u = [float(v) for v in random_uniform(len(T), seed=seed, stream=1)] if T else []
    out = [0] * len(T)
    probs = [None] * len(T)
    for step, t in enumerate(path):
        x = T[t]
        exact = [z for c, z in known if c == x]
        if exact:
            out[t] = exact[0]
            probs[t] = [1.0 if c == exact[0] else 0.0 for c in range(K)]
            continue
        order = sorted(range(len(known)), key=lambda q: (math.dist(known[q][0], x), q))[:k]
        pts = [known[q][0] for q in order]
        cats = [known[q][1] for q in order]
        n1 = len(pts)
        pr = []
        for c in range(K):
            m = models[c]
            C = [[_cov(pts[a], pts[b], m) for b in range(n1)] for a in range(n1)]
            c0 = [_cov(pts[a], x, m) for a in range(n1)]
            res = [(1.0 if cats[a] == c else 0.0) - proportions[c] for a in range(n1)]
            if soft is not None:
                bc = B[c]
                for a in range(n1):
                    C[a].append(bc * _cov(pts[a], x, m))
                C.append([bc * _cov(x, pts[b], m) for b in range(n1)] + [bc * _cov(x, x, m)])
                c0.append(bc * _cov(x, x, m))
                res.append(float(soft[t][c]) - proportions[c])
            lam = solve(C, c0) if C else []
            pr.append(min(max(proportions[c] + ssum(a * b for a, b in zip(lam, res)), 0.0), 1.0))
        tot = ssum(pr)
        pr = [v / tot for v in pr] if tot > 0 else [float(v) for v in proportions]
        acc, cat = 0.0, K - 1
        for c in range(K):
            acc += pr[c]
            if u[step] <= acc:
                cat = c
                break
        out[t] = cat
        probs[t] = pr
        known.append((x, cat))
    return RichResult(payload={"simulated": out, "probabilities": probs, "path": path})


def snesim_simulate(training, nx: int, ny: int, template, *, conditioning=None, seed: int = 0) -> RichResult:
    r"""Single normal equation multiple-point simulation (SNESIM, Strebelle 2002) on a grid.

    ``training[j][i]`` is the categorical training image; ``template`` a list
    of ``(di, dj)`` offsets. Along a Philox random path each node's data
    event (informed template nodes) is matched against every training
    location; the conditional probability of each category is its share
    among the matching replicates (if none match, the farthest informed
    template node is dropped and the scan repeated; with no informed nodes
    the training proportions are used). A uniform (stream 1) draws the
    category. ``conditioning`` maps ``(i, j)`` to hard data.

    References
    ----------
    Strebelle, S. (2002). Conditional simulation of complex geological
    structures using multiple-point statistics. Math. Geology 34, 1-21.
    Guardiano, F. B. and Srivastava, R. M. (1993). Multivariate
    geostatistics: beyond bivariate moments. In Geostatistics Troia '92, 133-144.

    Examples
    --------
    >>> ti = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
    >>> r = snesim_simulate(ti, 3, 3, [(1, 0), (0, 1), (-1, 0), (0, -1)], conditioning={(0, 0): 1}, seed=1)
    >>> r.grid[0][0]
    1
    """
    TJ, TI = len(training), len(training[0])
    cats = sorted(set(v for row in training for v in row))
    grid = [[None] * nx for _ in range(ny)]
    for (i, j), v in (conditioning or {}).items():
        grid[j][i] = v
    nodes = [(i, j) for j in range(ny) for i in range(nx) if grid[j][i] is None]
    path = _perm(len(nodes), seed, 0)
    u = [float(v) for v in random_uniform(len(nodes), seed=seed, stream=1)] if nodes else []
    tot = TI * TJ
    marg = [ssum(1.0 for row in training for v in row if v == c) / tot for c in cats]
    for step, q in enumerate(path):
        i, j = nodes[q]
        ev = [
            (di, dj, grid[j + dj][i + di])
            for di, dj in template
            if 0 <= i + di < nx and 0 <= j + dj < ny and grid[j + dj][i + di] is not None
        ]
        ev.sort(key=lambda e: (e[0] ** 2 + e[1] ** 2, e[0], e[1]))
        pr = marg
        while ev:
            counts = [0.0] * len(cats)
            for tj in range(TJ):
                for ti in range(TI):
                    ok = True
                    for di, dj, v in ev:
                        a, b = ti + di, tj + dj
                        if not (0 <= a < TI and 0 <= b < TJ) or training[b][a] != v:
                            ok = False
                            break
                    if ok:
                        counts[cats.index(training[tj][ti])] += 1.0
            s = ssum(counts)
            if s > 0:
                pr = [c / s for c in counts]
                break
            ev = ev[:-1]
        acc, pick = 0.0, cats[-1]
        for c, pv in zip(cats, pr):
            acc += pv
            if u[step] <= acc:
                pick = c
                break
        grid[j][i] = pick
    return RichResult(payload={"grid": grid, "categories": cats})


def annealing_simulate(
    values,
    nx: int,
    ny: int,
    lags,
    target,
    *,
    n_iter: int = 5000,
    t0: float = 1.0,
    cooling: float = 0.9,
    every: int = 100,
    seed: int = 0,
) -> RichResult:
    r"""Simulated annealing of a grid to a target variogram by value swaps (Deutsch and Cowan 1996).

    The initial image is ``values`` placed by a Philox permutation (stream
    0), preserving the histogram. The objective
    ``O = sum_h (g(h) - g*(h))^2 / g*(h)^2`` uses the experimental
    semivariogram ``g(h)`` averaged over the x and y directions at the
    integer ``lags``. Iteration ``s`` swaps two random nodes (uniforms on
    stream ``s + 1``), accepted when ``O`` decreases or with probability
    ``exp(-dO / T)``; ``T`` starts at ``t0`` and is multiplied by
    ``cooling`` every ``every`` iterations. Swap effects are updated
    incrementally.

    References
    ----------
    Deutsch, C. V. and Cowan, P. W. (1996). Simulated annealing applied to
    geostatistics. In GSLIB, 2nd ed., section V.6. Kirkpatrick, S., Gelatt,
    C. D. and Vecchi, M. P. (1983). Science 220, 671-680.

    Examples
    --------
    >>> r = annealing_simulate([float(v) for v in range(16)], 4, 4, [1], [8.0], n_iter=50)
    >>> sorted(v for row in r.grid for v in row) == [float(v) for v in range(16)]
    True
    """
    n = nx * ny
    perm = _perm(n, seed, 0)
    g = [[0.0] * nx for _ in range(ny)]
    for idx in range(n):
        g[idx // nx][idx % nx] = float(values[perm[idx]])
    L = len(lags)
    npair = [ny * (nx - h) + nx * (ny - h) for h in lags]

    def sums():
        s = [0.0] * L
        for a, h in enumerate(lags):
            for j in range(ny):
                for i in range(nx):
                    if i + h < nx:
                        s[a] += (g[j][i] - g[j][i + h]) ** 2
                    if j + h < ny:
                        s[a] += (g[j][i] - g[j + h][i]) ** 2
        return s

    def objective(s):
        return ssum(((s[a] / (2 * npair[a])) - target[a]) ** 2 / target[a] ** 2 for a in range(L))

    def contrib(i, j):
        c = [0.0] * L
        for a, h in enumerate(lags):
            for di, dj in ((h, 0), (-h, 0), (0, h), (0, -h)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    c[a] += (g[j][i] - g[jj][ii]) ** 2
        return c

    S = sums()
    obj = objective(S)
    T = t0
    acc = 0
    for s in range(n_iter):
        u = random_uniform(3, seed=seed, stream=s + 1)
        p1 = min(int(float(u[0]) * n), n - 1)
        p2 = min(int(float(u[1]) * n), n - 1)
        if p1 != p2:
            i1, j1, i2, j2 = p1 % nx, p1 // nx, p2 % nx, p2 // nx
            before = [a + b for a, b in zip(contrib(i1, j1), contrib(i2, j2))]
            g[j1][i1], g[j2][i2] = g[j2][i2], g[j1][i1]
            after = [a + b for a, b in zip(contrib(i1, j1), contrib(i2, j2))]
            # a pair joining the two swapped nodes is counted twice in both, and its value is unchanged
            Sn = [S[a] + after[a] - before[a] for a in range(L)]
            obj_new = objective(Sn)
            if obj_new <= obj or float(u[2]) < math.exp(-(obj_new - obj) / T):
                S, obj = Sn, obj_new
                acc += 1
            else:
                g[j1][i1], g[j2][i2] = g[j2][i2], g[j1][i1]
        if (s + 1) % every == 0:
            T *= cooling
    return RichResult(
        payload={"grid": g, "objective": obj, "accepted": acc, "variogram": [S[a] / (2 * npair[a]) for a in range(L)]}
    )


def cheatsheet() -> str:
    return (
        "sgs_block_simulate / conditional_ensemble / pfield_simulate / lmc_conditional_simulate / collocated_cosimulate / "
        "sis_markov_bayes / snesim_simulate / annealing_simulate -> geostatistical simulation."
    )
