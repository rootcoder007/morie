# morie.fn -- function file (rootcoder007/morie)
"""Determine the number of factors: parallel analysis, Velicer's MAP, Kaiser, scree (OC, AF), variance, AIC, BIC."""

import math

from ._mlfa import corr_matrix, eigh_desc, fa_statistic, mlfa_fit, to_corr
from ._rng import random_normal


def _ssum(it):
    # plain left-to-right summation: sum() of floats is compensated from Python 3.12 on, which
    # would make results depend on the Python version and differ from the R arm
    s = 0.0
    for v in it:
        s += v
    return s


__all__ = ["efa_nfactors"]


def _quantile7(v, q):
    s = sorted(v)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])


def _map(R, ev, V, power):
    p = len(R)
    denom = p * (p - 1)
    out = [_ssum(R[i][j] ** power for i in range(p) for j in range(p) if i != j) / denom]
    for m in range(1, p - 1):
        A = [[V[i][k] * math.sqrt(max(ev[k], 0.0)) for k in range(m)] for i in range(p)]
        C = [[R[i][j] - _ssum(A[i][k] * A[j][k] for k in range(m)) for j in range(p)] for i in range(p)]
        d = [math.sqrt(C[i][i]) for i in range(p)]
        out.append(_ssum((C[i][j] / (d[i] * d[j])) ** power for i in range(p) for j in range(p) if i != j) / denom)
    return out


def efa_nfactors(data, *, method="parallel", nsim=100, seed=42, quantile=0.95, threshold=0.7, max_factors=None):
    r"""Number of factors to retain from the correlation matrix of ``data``.

    method:

    * ``"parallel"`` (Horn 1965): simulate ``nsim`` n x p standard-normal data sets
      (morie Philox normals, stream i for set i), take the ``quantile`` of each
      ordered correlation eigenvalue, and retain the leading eigenvalues that
      exceed their threshold.
    * ``"map"`` (Velicer 1976): average squared partial correlation after
      partialling out the first m principal components (loadings V_m sqrt(lambda_m)),
      m = 0, ..., p - 2 (m = 0 is the average squared correlation); retain the
      minimising m. ``"map4"`` uses fourth powers (Velicer, Eaton & Fava 2000).
    * ``"kaiser"``: eigenvalues greater than 1.
    * ``"scree"``: optimal coordinates (Raiche et al. 2013, nFactors::nScree):
      the last i with lambda_i at least the value predicted by the line through
      (i + 1, lambda_{i+1}) and (p, lambda_p), and lambda_i >= 1.
      ``"af"``: acceleration factor, argmax_j (lambda_{j+1} - 2 lambda_j + lambda_{j-1}) - 1.
    * ``"variance"``: smallest m whose leading eigenvalues explain ``threshold`` of the variance.
    * ``"aic"``, ``"bic"``: maximum-likelihood factor models (factanal's objective)
      for m = 1, 2, ... while the degrees of freedom stay non-negative;
      chi2 - 2 df and chi2 - df log n, minimised over m.

    Parameters
    ----------
    data : n x p numeric data (rows with non-finite values are dropped)
    method : str
    nsim : int
    seed : int
    quantile : float
    threshold : float
    max_factors : int, optional

    Returns
    -------
    dict
        n_factors, method, eigenvalues and the criterion values (threshold, map_values,
        predicted, acceleration, cumulative, aic_values / bic_values).

    References
    ----------
    Horn, J. L. (1965). Psychometrika 30, 179-185.
    Velicer, W. F. (1976). Psychometrika 41, 321-327.
    Velicer, W. F., Eaton, C. A. & Fava, J. L. (2000). In Problems and Solutions in Human Assessment, 41-71.
    Raiche, G., Walls, T. A., Magis, D., Riopel, M. & Blais, J.-G. (2013). Methodology 9, 23-29.

    Examples
    --------
    >>> X = [[i, 2 * i + (i % 3), 3 * i - (i % 2), (7 * i) % 5] for i in range(20)]
    >>> efa_nfactors(X, method="kaiser")["n_factors"] >= 1
    True
    """
    if hasattr(data, "to_numpy"):
        data = data.to_numpy()
    if hasattr(data, "tolist"):
        data = data.tolist()
    X = [[float(v) for v in row] for row in data]
    X = [row for row in X if all(math.isfinite(v) for v in row)]
    n, p = len(X), len(X[0])
    if n < 3 or p < 2:
        raise ValueError("need at least 3 complete rows and 2 variables")
    R = to_corr(corr_matrix(X))
    ev, V = eigh_desc(R)
    out = {"method": method, "eigenvalues": ev}
    if method == "parallel":
        sims = []
        for i in range(nsim):
            z = random_normal(n * p, seed=seed, stream=i)
            Z = [[z[r * p + c] for c in range(p)] for r in range(n)]
            sims.append(eigh_desc(to_corr(corr_matrix(Z)))[0])
        thr = [_quantile7([s[j] for s in sims], quantile) for j in range(p)]
        k = 0
        while k < p and ev[k] > thr[k]:
            k += 1
        out.update(n_factors=k, threshold=thr)
    elif method in ("map", "map4"):
        vals = _map(R, ev, V, 2 if method == "map" else 4)
        out.update(n_factors=min(range(len(vals)), key=lambda m: vals[m]), map_values=vals)
    elif method == "kaiser":
        out.update(n_factors=sum(v > 1 for v in ev))
    elif method in ("scree", "af"):
        k = p
        pred = [None] * p
        for i in range(1, p - 1):
            pred[i - 1] = ev[i] + (ev[p - 1] - ev[i]) / (p - 1 - i) * (-1)
        nc = 0
        for i in range(p - 2):
            if ev[i] >= pred[i] and ev[i] >= 1:
                nc = i + 1
            else:
                break
        af = [None] + [ev[j + 1] - 2 * ev[j] + ev[j - 1] for j in range(1, p - 1)] + [None]
        cand = [j for j in range(1, p - 1) if ev[j - 1] >= 1] or list(
            range(1, p - 1)
        )  # nScree: af only where lambda_{j-1} >= 1
        best = max(cand, key=lambda j: af[j])
        k = nc if method == "scree" else best
        out.update(n_factors=k, predicted=pred, acceleration=af)
    elif method == "variance":
        tot = _ssum(ev)
        cum = []
        s = 0.0
        for v in ev:
            s += v
            cum.append(s / tot)
        out.update(n_factors=next(m + 1 for m, c in enumerate(cum) if c >= threshold - 1e-12), cumulative=cum)
    elif method in ("aic", "bic"):
        vals = []
        top = max_factors or p - 1
        for m in range(1, top + 1):
            if ((p - m) ** 2 - p - m) < 0:
                break
            st, dof = fa_statistic(mlfa_fit(R, m)["objective"], n, p, m)
            vals.append(st - 2 * dof if method == "aic" else st - dof * math.log(n))
        if not vals:
            raise ValueError("no factor model with non-negative degrees of freedom")
        out.update(n_factors=1 + min(range(len(vals)), key=lambda j: vals[j]), **{f"{method}_values": vals})
    else:
        raise ValueError(
            'method must be "parallel", "map", "map4", "kaiser", "scree", "af", "variance", "aic" or "bic"'
        )
    return out


def cheatsheet():
    return "efa_nfactors: number of factors by parallel analysis, MAP, Kaiser, scree (OC/AF), variance, AIC, BIC."


# compact alias per ledger/NAMING.md
efanfactors = efa_nfactors
