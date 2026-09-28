# morie.fn -- function file (rootcoder007/morie)
"""Sequential simulation (GSLIB style): normal-score transforms, sequential Gaussian simulation with simple or
ordinary kriging and collocated co-simulation, sequential indicator simulation (continuous and categorical) with
order-relation correction, realisation post-processing, leave-one-out validation and Markov-chain categorical
simulation."""

from __future__ import annotations

import math

from ._qpcore import solve, ssum
from ._richresult import RichResult
from ._rng import random_normal, random_uniform
from ._rrng_core import qnorm
from .krgsys import kriging_covariance

__all__ = [
    "normal_score",
    "back_transform",
    "sgs_simulate",
    "sis_simulate",
    "simulation_summary",
    "sgs_cross_validation",
    "transition_matrix",
    "markov_chain_simulate",
]


def _pts(a):
    return [tuple(float(v) for v in (r if isinstance(r, (list, tuple)) else [r])) for r in a]


def _d(a, b):
    return math.sqrt(ssum((x - y) ** 2 for x, y in zip(a, b)))


def _q7(s, p):
    h = (len(s) - 1) * p
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def normal_score(z) -> RichResult:
    r"""Normal-score transform ``y_i = Phi^{-1}((r_i - 1/2)/n)`` with ``r_i`` the rank (ties by input order; GSLIB ``nscore``).

    Returns the scores and the sorted transformation table used by
    :func:`back_transform`.

    References
    ----------
    Deutsch, C. V. and Journel, A. G. (1998). *GSLIB: Geostatistical Software
    Library and User's Guide*, 2nd ed. Oxford University Press.

    Examples
    --------
    >>> [round(v, 6) for v in normal_score([3.0, 1.0, 2.0]).scores]
    [0.967422, -0.967422, 0.0]
    """
    zv = [float(v) for v in z]
    n = len(zv)
    order = sorted(range(n), key=lambda i: (zv[i], i))
    y = [0.0] * n
    for r, i in enumerate(order):
        y[i] = float(qnorm((r + 0.5) / n))
    return RichResult(payload={"scores": y, "table_z": [zv[i] for i in order], "table_y": [y[i] for i in order]})


def back_transform(y, table_z, table_y):
    r"""Back-transform normal scores by linear interpolation in the ``(y, z)`` table, clamped to the data range.

    Examples
    --------
    >>> back_transform([0.0, 5.0], [1.0, 2.0, 3.0], [-1.0, 0.0, 1.0])
    [2.0, 3.0]
    """
    Y, Z = [float(v) for v in table_y], [float(v) for v in table_z]
    out = []
    for v in (float(u) for u in y):
        if v <= Y[0]:
            out.append(Z[0])
        elif v >= Y[-1]:
            out.append(Z[-1])
        else:
            k = next(i for i in range(1, len(Y)) if Y[i] >= v)
            t = (v - Y[k - 1]) / (Y[k] - Y[k - 1]) if Y[k] > Y[k - 1] else 0.0
            out.append(Z[k - 1] + t * (Z[k] - Z[k - 1]))
    return out


def _krige_node(P, V, q, model, kriging, mean, extra=None):
    """Kriging mean and variance at q from neighbours P (values V); extra = (rho, secondary value) for MM1."""
    n = len(P)
    C = [[kriging_covariance(_d(P[i], P[j]), model) for j in range(n)] for i in range(n)]
    c0 = [kriging_covariance(_d(p, q), model) for p in P]
    c00 = kriging_covariance(0.0, model)
    if extra is not None:
        rho, s0 = extra
        C = [row + [rho * c0[i]] for i, row in enumerate(C)] + [[rho * c for c in c0] + [1.0]]
        rhs = c0 + [rho]
        w = [float(v) for v in solve(C, rhs)] if n else [rho]
        vals = [v - mean for v in V] + [s0]
        return mean + ssum(a * b for a, b in zip(w, vals)), c00 - ssum(a * b for a, b in zip(w, rhs))
    if n == 0:
        return mean, c00
    if kriging == "simple":
        w = [float(v) for v in solve(C, c0)]
        return mean + ssum(a * (v - mean) for a, v in zip(w, V)), c00 - ssum(a * b for a, b in zip(w, c0))
    A = [row + [1.0] for row in C] + [[1.0] * n + [0.0]]
    sol = [float(v) for v in solve(A, c0 + [1.0])]
    w, mu = sol[:n], sol[n]
    return ssum(a * v for a, v in zip(w, V)), c00 - ssum(a * b for a, b in zip(w, c0)) - mu


def _path(m, seed, s):
    u = [float(v) for v in random_uniform(m, seed=seed, stream=3 * s)]
    idx = list(range(m))
    for t in range(m):
        j = t + int(u[t] * (m - t))
        idx[t], idx[j] = idx[j], idx[t]
    return idx


def _nearest(cands, q, nmax):
    return sorted(range(len(cands)), key=lambda i: (_d(cands[i], q), i))[:nmax]


def sgs_simulate(
    coords,
    z,
    grid,
    model,
    *,
    kriging: str = "simple",
    mean: float = 0.0,
    nmax: int = 16,
    nsim: int = 1,
    seed: int = 1,
    secondary=None,
    rho: float = 0.0,
) -> RichResult:
    r"""Sequential Gaussian simulation of a Gaussian (normal-score) variable at ``grid`` nodes, conditioned on data.

    For each realisation ``s`` the nodes are visited along a random path
    (Fisher-Yates with Philox stream ``3s`` of ``seed``); at each node the
    ``nmax`` nearest conditioning values (data first, then previously
    simulated nodes; ties by that order; nodes on a datum take its value) give a simple kriging (known
    ``mean``) or ordinary kriging mean and variance, and the node value is
    ``mean + sqrt(max(var, 0)) e`` with ``e`` the next standard normal of
    stream ``3s + 1`` (Deutsch and Journel 1998). With ``secondary`` (the
    standardised secondary value at every grid node) and
    correlation ``rho``, collocated simple cokriging under Markov model 1
    (cross-covariance ``rho C(h)``) adds the collocated secondary value to each
    system (Almeida and Journel 1994). Work in normal scores
    (:func:`normal_score`) and back-transform the results.

    References
    ----------
    Deutsch, C. V. and Journel, A. G. (1998). *GSLIB*, 2nd ed., section V.2.
    Almeida, A. S. and Journel, A. G. (1994). Joint simulation of multiple
    variables with a Markov-type coregionalization model. *Mathematical
    Geology*, 26(5), 565-588.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 2.0}
    >>> r = sgs_simulate([(0.0, 0.0)], [1.0], [(0.0, 0.0), (5.0, 0.0)], m, seed=2)
    >>> r.realizations[0][0]
    1.0
    """
    D, zv = _pts(coords), [float(v) for v in z]
    G = _pts(grid)
    m = len(G)
    sec_g = None if secondary is None else [float(v) for v in secondary]
    reals, paths = [], []
    for s in range(nsim):
        path = _path(m, seed, s)
        e = [float(v) for v in random_normal(m, seed=seed, stream=3 * s + 1)]
        cP, cV = list(D), list(zv)
        out = [0.0] * m
        for k, g in enumerate(path):
            q = G[g]
            same = next((i for i, p in enumerate(D) if _d(p, q) == 0.0), None)
            if same is not None:  # a node on a datum takes its value and is already conditioning data
                out[g] = zv[same]
                continue
            nb = _nearest(cP, q, nmax)
            extra = (rho, sec_g[g]) if sec_g is not None else None
            mu, var = _krige_node(
                [cP[i] for i in nb],
                [cV[i] for i in nb],
                q,
                model,
                "simple" if extra is not None else kriging,
                mean,
                extra,
            )
            val = mu + math.sqrt(max(var, 0.0)) * e[k]
            out[g] = val
            cP.append(q)
            cV.append(val)
        reals.append(out)
        paths.append(path)
    return RichResult(payload={"realizations": reals, "paths": paths})


def _order_relation(F):
    """GSLIB order-relation correction: clip to [0, 1], average of the upward and downward monotone passes."""
    F = [min(1.0, max(0.0, v)) for v in F]
    up = list(F)
    for k in range(1, len(up)):
        up[k] = max(up[k], up[k - 1])
    dn = list(F)
    for k in range(len(dn) - 2, -1, -1):
        dn[k] = min(dn[k], dn[k + 1])
    return [(a + b) / 2 for a, b in zip(up, dn)]


def sis_simulate(
    coords,
    z,
    grid,
    thresholds,
    models,
    *,
    nmax: int = 16,
    nsim: int = 1,
    seed: int = 1,
    categorical: bool = False,
    proportions=None,
) -> RichResult:
    r"""Sequential indicator simulation (Journel and Alabert 1989) with simple indicator kriging.

    Continuous variables: indicators ``I_k = 1{z <= t_k}`` for the sorted
    ``thresholds``, prior means their global proportions (or
    ``proportions``); at each node simple indicator kriging of every
    threshold (with ``models[k]``, or one model for all: median indicator
    kriging) gives a ccdf, corrected for order relations (clipping to
    ``[0, 1]`` and averaging upward and downward monotone passes), and a
    class ``0..K`` is drawn by inverting the ccdf with the next uniform of
    stream ``3s + 2``. Categorical variables (``categorical=True``,
    ``thresholds`` = category codes): indicators ``1{z = c_k}``, kriged
    probabilities clipped at 0 and renormalised. Simulated classes join the
    conditioning data. Returns the simulated class index per node and the
    ccdf/probabilities used at each node.

    References
    ----------
    Journel, A. G. and Alabert, F. (1989). Non-Gaussian data expansion in
    the earth sciences. *Terra Nova*, 1(2), 123-134.
    Deutsch, C. V. and Journel, A. G. (1998). *GSLIB*, 2nd ed., sections
    IV.1.9 and V.3.

    Examples
    --------
    >>> m = {"model": "Sph", "psill": 0.25, "range": 3.0}
    >>> r = sis_simulate([(0.0, 0.0), (4.0, 0.0)], [0, 1], [(0.0, 0.0), (1.0, 0.0)], [0, 1], m, categorical=True)
    >>> r.realizations[0][0]
    0
    """
    D, zv = _pts(coords), [float(v) for v in z]
    G = _pts(grid)
    T = [float(v) for v in thresholds]
    K = len(T)
    mods = models if isinstance(models, list) else [models] * K
    n = len(zv)
    if categorical:
        ind = [[1.0 if abs(v - t) < 1e-12 else 0.0 for t in T] for v in zv]
    else:
        T = sorted(T)
        ind = [[1.0 if v <= t else 0.0 for t in T] for v in zv]
    prior = (
        [float(v) for v in proportions] if proportions is not None else [ssum(r[k] for r in ind) / n for k in range(K)]
    )
    reals, probs_all = [], []
    for s in range(nsim):
        path = _path(len(G), seed, s)
        u = [float(v) for v in random_uniform(len(G), seed=seed, stream=3 * s + 2)]
        cP, cI = list(D), [list(r) for r in ind]
        out, probs = [0] * len(G), [None] * len(G)
        for k, g in enumerate(path):
            q = G[g]
            same = next((i for i, p in enumerate(D) if _d(p, q) == 0.0), None)
            if same is not None:  # a node on a datum takes its class
                row = ind[same]
                out[g] = row.index(1.0) if categorical else next((j for j, v in enumerate(row) if v == 1.0), K)
                probs[g] = list(row)
                continue
            nb = _nearest(cP, q, nmax)
            est = []
            for j in range(K):
                mu, _ = _krige_node([cP[i] for i in nb], [cI[i][j] for i in nb], q, mods[j], "simple", prior[j])
                est.append(mu)
            if categorical:
                pr = [max(0.0, v) for v in est]
                tot = ssum(pr)
                pr = [v / tot for v in pr] if tot > 0 else list(prior)
                acc, cls = 0.0, K - 1
                for j, p in enumerate(pr):
                    acc += p
                    if u[k] < acc:
                        cls = j
                        break
                code = [0.0] * K
                code[cls] = 1.0
            else:
                pr = _order_relation(est)
                cls = next((j for j, F in enumerate(pr) if u[k] <= F), K)
                code = [1.0 if cls <= j else 0.0 for j in range(K)]
            out[g], probs[g] = cls, pr
            cP.append(q)
            cI.append(code)
        reals.append(out)
        probs_all.append(probs)
    return RichResult(payload={"realizations": reals, "probabilities": probs_all, "thresholds": T})


def simulation_summary(
    realizations, *, probs=(0.1, 0.5, 0.9), threshold: float | None = None, data=None, blocks=None
) -> RichResult:
    r"""Post-process a set of realisations (rows) node by node.

    E-type mean, conditional variance (divisor ``nsim - 1``), percentile maps
    (type-7 quantiles at ``probs``), probability of exceeding ``threshold``;
    with ``data`` the histogram reproduction of each realisation (Kolmogorov-
    Smirnov distance between its values and the data); with ``blocks`` (a
    block id per node) the block-average of each realisation and the mean and
    variance of the block averages (change of support by simulation).

    Examples
    --------
    >>> s = simulation_summary([[1.0, 4.0], [3.0, 2.0]], threshold=2.5)
    >>> s.etype, s.variance, s.exceedance
    ([2.0, 3.0], [2.0, 2.0], [0.5, 0.5])
    """
    R = [[float(v) for v in r] for r in realizations]
    ns, m = len(R), len(R[0])
    cols = [[R[s][i] for s in range(ns)] for i in range(m)]
    et = [ssum(c) / ns for c in cols]
    var = [ssum((v - mu) ** 2 for v in c) / (ns - 1) if ns > 1 else float("nan") for c, mu in zip(cols, et)]
    pct = {p: [_q7(sorted(c), p) for c in cols] for p in probs}
    out = {"etype": et, "variance": var, "percentiles": {f"{p:g}": v for p, v in pct.items()}}
    if threshold is not None:
        out["exceedance"] = [sum(1 for v in c if v > threshold) / ns for c in cols]
    if data is not None:
        dz = sorted(float(v) for v in data)
        ks = []
        for r in R:
            sr = sorted(r)
            pts = sorted(set(dz) | set(sr))
            ks.append(
                max(abs(sum(1 for v in sr if v <= t) / len(sr) - sum(1 for v in dz if v <= t) / len(dz)) for t in pts)
            )
        out["ks_distance"] = ks
    if blocks is not None:
        ids = sorted(set(blocks), key=str)
        ba = [
            [ssum(r[i] for i in range(m) if blocks[i] == b) / sum(1 for x in blocks if x == b) for b in ids] for r in R
        ]
        out["block_ids"] = ids
        out["block_averages"] = ba
        out["block_mean"] = [ssum(r[j] for r in ba) / ns for j in range(len(ids))]
        out["block_variance"] = [
            ssum((r[j] - out["block_mean"][j]) ** 2 for r in ba) / (ns - 1) if ns > 1 else float("nan")
            for j in range(len(ids))
        ]
    return RichResult(payload=out)


def sgs_cross_validation(
    coords,
    z,
    model,
    *,
    kriging: str = "simple",
    mean: float = 0.0,
    nmax: int = 16,
    nsim: int = 100,
    seed: int = 1,
    level: float = 0.9,
) -> RichResult:
    r"""Leave-one-out validation of sequential Gaussian simulation.

    Each datum is simulated ``nsim`` times from the others
    (:func:`sgs_simulate` at that single location, seed ``seed + i``); the
    E-type error, the empirical coverage of the central ``level`` interval
    of the simulated values and the mean interval width are reported.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 3.0}
    >>> r = sgs_cross_validation([(0, 0), (1, 0), (2, 0), (3, 0)], [0.2, 0.5, -0.1, 0.3], m, nsim=20)
    >>> len(r.etype_error), 0.0 <= r.coverage <= 1.0
    (4, True)
    """
    P, zv = _pts(coords), [float(v) for v in z]
    err, inside, width = [], 0, []
    lo_p, hi_p = (1 - level) / 2, 1 - (1 - level) / 2
    for i in range(len(zv)):
        others = [j for j in range(len(zv)) if j != i]
        r = sgs_simulate(
            [P[j] for j in others],
            [zv[j] for j in others],
            [P[i]],
            model,
            kriging=kriging,
            mean=mean,
            nmax=nmax,
            nsim=nsim,
            seed=seed + i,
        )
        vals = sorted(rr[0] for rr in r["realizations"])
        et = ssum(vals) / nsim
        lo, hi = _q7(vals, lo_p), _q7(vals, hi_p)
        err.append(zv[i] - et)
        inside += lo <= zv[i] <= hi
        width.append(hi - lo)
    n = len(zv)
    return RichResult(
        payload={
            "etype_error": err,
            "rmse": math.sqrt(ssum(e * e for e in err) / n),
            "coverage": inside / n,
            "mean_width": ssum(width) / n,
        }
    )


def transition_matrix(sequences, categories=None) -> RichResult:
    r"""One-step transition probability matrix of categorical sequences (e.g. facies along wells; Carle and Fogg 1996).

    ``T[a][b] = n(a -> b) / n(a -> .)`` over consecutive pairs of every
    sequence, with the category proportions and the mean lengths ``1/(1 -
    T[a][a])`` of the runs.

    References
    ----------
    Carle, S. F. and Fogg, G. E. (1996). Transition probability-based
    indicator geostatistics. *Mathematical Geology*, 28(4), 453-476.

    Examples
    --------
    >>> [[round(v, 6) for v in r] for r in transition_matrix([["a", "a", "b", "b", "b", "a"]]).matrix]
    [[0.5, 0.5], [0.333333, 0.666667]]
    """
    seqs = [list(s) for s in (sequences if isinstance(sequences[0], (list, tuple)) else [sequences])]
    cats = sorted({c for s in seqs for c in s}, key=str) if categories is None else list(categories)
    ix = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    N = [[0.0] * K for _ in range(K)]
    for s in seqs:
        for a, b in zip(s, s[1:]):
            N[ix[a]][ix[b]] += 1
    M = [[v / ssum(r) if ssum(r) > 0 else 0.0 for v in r] for r in N]
    tot = sum(len(s) for s in seqs)
    props = [sum(1 for s in seqs for c in s if c == k) / tot for k in cats]
    runs = [1 / (1 - M[i][i]) if M[i][i] < 1 else float("inf") for i in range(K)]
    return RichResult(
        payload={"matrix": M, "categories": cats, "proportions": props, "mean_run_length": runs, "counts": N}
    )


def markov_chain_simulate(T, start: int, n: int, *, seed: int = 1, conditioning=None):
    r"""Simulate a categorical sequence of length ``n`` from transition matrix ``T`` (Philox uniforms, stream 0).

    ``conditioning`` maps positions to fixed categories (the chain is reset
    there). Returns category indices.

    Examples
    --------
    >>> markov_chain_simulate([[0.0, 1.0], [1.0, 0.0]], 0, 5)
    [0, 1, 0, 1, 0]
    """
    P = [[float(v) for v in r] for r in T]
    u = [float(v) for v in random_uniform(n, seed=seed, stream=0)]
    fix = {} if conditioning is None else {int(k): int(v) for k, v in dict(conditioning).items()}
    out = [fix.get(0, int(start))]
    for t in range(1, n):
        if t in fix:
            out.append(fix[t])
            continue
        row = P[out[-1]]
        acc, nxt = 0.0, len(row) - 1
        for j, p in enumerate(row):
            acc += p
            if u[t] < acc:
                nxt = j
                break
        out.append(nxt)
    return out


def cheatsheet() -> str:
    return "normal_score / sgs_simulate / sis_simulate / simulation_summary / markov_chain_simulate -> simulation."
