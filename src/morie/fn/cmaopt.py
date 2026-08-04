# morie.fn -- tail3 batch (rootcoder007/morie)
"""CMA-ES, the covariance matrix adaptation evolution strategy.

Source consulted: Hansen, N. & Ostermeier, A. (2001). Completely derandomized
self-adaptation in evolution strategies.  *Evolutionary Computation* 9(2),
159-195.  The update equations and the default strategy parameters are those
of the author's own reference listing ``purecmaes.m``, reproduced in Hansen,
The CMA Evolution Strategy: A Tutorial (arXiv:1604.00772), whose lines carry
the equation numbers quoted below:

    lambda = 4 + floor(3 log N),  mu = floor(lambda/2)
    w_i    = log(mu + 1/2) - log i,  normalised,  mueff = 1 / sum w_i^2
    cc     = (4 + mueff/N) / (N + 4 + 2 mueff/N)
    cs     = (mueff + 2) / (N + mueff + 5)
    c1     = 2 / ((N + 1.3)^2 + mueff)
    cmu    = min(1 - c1, 2 (mueff - 2 + 1/mueff) / ((N + 2)^2 + mueff))
    damps  = 1 + 2 max(0, sqrt((mueff - 1)/(N + 1)) - 1) + cs

    x_k   = m + sigma C^{1/2} z_k                                     (eq. 40)
    m     <- sum_i w_i x_{i:lambda}                                   (eq. 42)
    ps    <- (1 - cs) ps + sqrt(cs (2 - cs) mueff) C^{-1/2} (m - m_old)/sigma
                                                                      (eq. 43)
    sigma <- sigma exp((cs/damps)(||ps||/chiN - 1))                   (eq. 44)
    pc    <- (1 - cc) pc + hsig sqrt(cc (2 - cc) mueff) (m - m_old)/sigma
                                                                      (eq. 45)
    C     <- (1 - c1 - cmu) C + c1 (pc pc' + (1 - hsig) cc (2 - cc) C)
             + cmu sum_i w_i y_i y_i'                                 (eq. 47)

Two deliberate departures from the listing, both required for the run to be
reproducible and for the R mirror to follow the identical trajectory:

* the mutation vectors z_k are supplied by the caller as ``Z`` rather than
  drawn internally;
* the square root of C is taken to be the symmetric root C^{1/2} = B D B'
  rather than the factor B D used by ``purecmaes.m``.  The two give the same
  sampling distribution because B is orthogonal, but B alone is not unique --
  eigenvector signs are arbitrary and, when C has repeated eigenvalues (as it
  does at start-up, where C = I), so is the whole eigenbasis.  The symmetric
  root is invariant to both, and it is the form in which equations (40), (43)
  and (45) are written in the first place.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["cma_es"]


def _sym_roots(C, N):
    """Symmetric square root and inverse square root of a PSD matrix."""
    vals, vecs = np.linalg.eigh(np.asarray(C, dtype=float))
    lam = [max(0.0, float(vals[k])) for k in range(N)]
    half = [float(np.sqrt(v)) for v in lam]
    ihalf = [1.0 / h if h > 0.0 else 0.0 for h in half]
    root = [[0.0] * N for _ in range(N)]
    iroot = [[0.0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            s1 = 0.0
            s2 = 0.0
            for k in range(N):
                p = float(vecs[r, k]) * float(vecs[c, k])
                s1 += p * half[k]
                s2 += p * ihalf[k]
            root[r][c] = s1
            iroot[r][c] = s2
    return root, iroot


def _matvec(M, v, N):
    return [sum(M[r][c] * v[c] for c in range(N)) for r in range(N)]


def cma_es(f, x0, sigma=0.5, Z=None, lam=None, iters=10):
    """Run CMA-ES for a fixed number of generations.

    Parameters
    ----------
    f : callable
        Objective to minimise, taking a length-N sequence.
    x0 : array-like
        Initial distribution mean.
    sigma : float
        Initial step size.
    Z : array-like
        Standard normal mutation vectors, ``iters * lam`` rows of length N,
        grouped generation by generation.  Required.
    lam : int, optional
        Population size; defaults to ``4 + floor(3 log N)``.
    iters : int
        Number of generations.

    Returns
    -------
    RichResult
        estimate (best objective seen), fbest, xbest, xmean, sigma, C, evals,
        generations, n, method.

    References
    ----------
    Hansen & Ostermeier (2001), Evolutionary Computation 9(2), 159-195;
    purecmaes.m as listed in arXiv:1604.00772.
    """
    xmean = [float(v) for v in np.atleast_1d(np.asarray(x0, dtype=float)).ravel()]
    N = len(xmean)
    if lam is None:
        lam = int(4 + int(3.0 * float(np.log(float(N)))))
    lam = int(lam)
    mu = int(lam // 2)
    wraw = [float(np.log(mu + 0.5)) - float(np.log(float(i + 1))) for i in range(mu)]
    wsum = sum(wraw)
    w = [v / wsum for v in wraw]
    mueff = 1.0 / sum(v * v for v in w)
    Nf = float(N)
    cc = (4.0 + mueff / Nf) / (Nf + 4.0 + 2.0 * mueff / Nf)
    cs = (mueff + 2.0) / (Nf + mueff + 5.0)
    c1 = 2.0 / ((Nf + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((Nf + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, float(np.sqrt((mueff - 1.0) / (Nf + 1.0))) - 1.0) + cs
    chiN = Nf**0.5 * (1.0 - 1.0 / (4.0 * Nf) + 1.0 / (21.0 * Nf * Nf))
    pc = [0.0] * N
    ps = [0.0] * N
    C = [[1.0 if r == c else 0.0 for c in range(N)] for r in range(N)]
    sig = float(sigma)
    Zm = np.atleast_2d(np.asarray(Z, dtype=float))
    fbest = float("inf")
    xbest = list(xmean)
    counteval = 0
    for g in range(int(iters)):
        root, iroot = _sym_roots(C, N)
        arz = []
        ary = []
        arx = []
        fit = []
        for k in range(lam):
            z = [float(Zm[g * lam + k, j]) for j in range(N)]
            y = _matvec(root, z, N)
            x = [xmean[r] + sig * y[r] for r in range(N)]
            arz.append(z)
            ary.append(y)
            arx.append(x)
            fv = float(f(x))
            fit.append(fv)
            counteval += 1
            if fv < fbest:
                fbest = fv
                xbest = list(x)
        order = sorted(range(lam), key=lambda k: (fit[k], k))
        xold = list(xmean)
        xmean = [sum(w[i] * arx[order[i]][r] for i in range(mu)) for r in range(N)]
        delta = [(xmean[r] - xold[r]) / sig for r in range(N)]
        cinvd = _matvec(iroot, delta, N)
        ps = [(1.0 - cs) * ps[r] + float(np.sqrt(cs * (2.0 - cs) * mueff)) * cinvd[r] for r in range(N)]
        psn = float(np.sqrt(sum(v * v for v in ps)))
        thresh = float(np.sqrt(1.0 - (1.0 - cs) ** (2.0 * counteval / lam)))
        hsig = 1.0 if psn / thresh / chiN < 1.4 + 2.0 / (Nf + 1.0) else 0.0
        pc = [(1.0 - cc) * pc[r] + hsig * float(np.sqrt(cc * (2.0 - cc) * mueff)) * delta[r] for r in range(N)]
        Cn = [[0.0] * N for _ in range(N)]
        for r in range(N):
            for c in range(N):
                Cn[r][c] = (1.0 - c1 - cmu) * C[r][c] + c1 * (
                    pc[r] * pc[c] + (1.0 - hsig) * cc * (2.0 - cc) * C[r][c]
                )
        for i in range(mu):
            y = ary[order[i]]
            for r in range(N):
                for c in range(N):
                    Cn[r][c] += cmu * w[i] * y[r] * y[c]
        for r in range(N):
            for c in range(r + 1, N):
                v = 0.5 * (Cn[r][c] + Cn[c][r])
                Cn[r][c] = v
                Cn[c][r] = v
        C = Cn
        sig = sig * float(np.exp((cs / damps) * (psn / chiN - 1.0)))
    return RichResult(
        payload={
            "estimate": float(fbest),
            "fbest": float(fbest),
            "xbest": np.asarray(xbest, dtype=float),
            "xmean": np.asarray(xmean, dtype=float),
            "sigma": float(sig),
            "C": np.asarray(C, dtype=float),
            "evals": int(counteval),
            "generations": int(iters),
            "n": N,
            "method": "CMA-ES (Hansen & Ostermeier 2001)",
        }
    )


# CANONICAL TEST
# >>> # a sphere started at its optimum with zero mutations never moves
# >>> Z = [[0.0, 0.0] for _ in range(12)]
# >>> r = cma_es(lambda v: v[0] ** 2 + v[1] ** 2, [0.0, 0.0], 0.5, Z, lam=4, iters=3)
# >>> assert abs(r["fbest"]) < 1e-30
# >>> assert r["evals"] == 12


def cheatsheet():
    return "cmaopt(f, x0, sigma, Z, lam, iters): CMA-ES with supplied noise."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
cmaes = cma_es
