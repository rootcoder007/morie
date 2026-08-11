"""CMIP multi-model ensemble weighting (Knutti et al. 2017)."""

import math

from ._richresult import RichResult

__all__ = ["caCMIP", "cmip_ensemble", "cmipensemble"]


def _rms(a, b):
    n = len(a)
    return math.sqrt(sum((float(x) - float(y)) ** 2
                         for x, y in zip(a, b)) / n)


def caCMIP(models, obs, sigma_d, sigma_s, projections=None):
    """
    Performance-and-independence weighted CMIP ensemble mean.

    Implements the model-projection weighting scheme of Knutti et
    al. (2017), Eq. 1: with D_i the RMS distance of model i's
    historical field to observations and S_ij the RMS distance
    between models i and j,

        w_i = exp(-D_i^2 / sigma_d^2)
              / (1 + sum_{j != i} exp(-S_ij^2 / sigma_s^2)),

    normalized to sum to one.  The numerator downweights poor
    performance (Gaussian in distance-to-obs); the denominator is the
    "effective repetition of a model": as stated in the paper, a
    model with no close neighbours has denominator ~ 1, while two
    identical models each get half weight, so duplicating a model
    does not change the result.  The weighted projection ensemble
    mean is sum_i w_i x_i.

    Sources
    -------
    Knutti, R., Sedlacek, J., Sanderson, B. M., Lorenz, R., Fischer,
    E. M. & Eyring, V. (2017). A climate model projection weighting
    scheme accounting for performance and interdependence.
    *Geophysical Research Letters*, 44, 1909-1918, Eq. 1 and
    surrounding text (local copy
    fetched-wave3/knutti-2017-model-weighting-grl44.pdf).
    Sanderson, B. M., Knutti, R. & Caldwell, P. (2015). A
    representative democracy to reduce interdependency in a
    multimodel ensemble. *Journal of Climate*, 28, 5171-5194 (the
    scheme's basis, Eqs. 10-16, as cited by Knutti et al.).

    Parameters
    ----------
    models : sequence of sequences
        Per-model historical fields (equal lengths), compared to obs.
    obs : sequence of float
        Observed field on the same grid/length.
    sigma_d, sigma_s : float
        Performance and similarity radii (same units as the fields).
    projections : sequence of float, optional
        One scalar projection per model to average; defaults to each
        model's own field mean.

    Returns
    -------
    RichResult
        Keys: estimate (weighted projection mean), weights,
        unweighted_mean, d (distances to obs), n_models,
        effective_n (1 / sum w_i^2).
    """
    mods = [[float(v) for v in m] for m in models]
    ob = [float(v) for v in obs]
    m_count = len(mods)
    if m_count < 1:
        raise ValueError("need at least one model")
    if any(len(mm) != len(ob) for mm in mods):
        raise ValueError("all models must match obs length")
    sd = float(sigma_d)
    ss = float(sigma_s)
    if sd <= 0 or ss <= 0:
        raise ValueError("sigma_d and sigma_s must be positive")
    if projections is None:
        proj = [sum(mm) / len(mm) for mm in mods]
    else:
        proj = [float(v) for v in projections]
        if len(proj) != m_count:
            raise ValueError("projections must have one value per model")
    d = [_rms(mm, ob) for mm in mods]
    s = [[0.0] * m_count for _ in range(m_count)]
    for i in range(m_count):
        for j in range(i + 1, m_count):
            s[i][j] = s[j][i] = _rms(mods[i], mods[j])
    w = []
    for i in range(m_count):
        num = math.exp(-d[i] ** 2 / sd ** 2)
        den = 1.0 + sum(math.exp(-s[i][j] ** 2 / ss ** 2)
                        for j in range(m_count) if j != i)
        w.append(num / den)
    tot = sum(w)
    if tot <= 0:
        raise ValueError("all weights vanished; increase sigma_d")
    w = [x / tot for x in w]
    est = sum(wi * xi for wi, xi in zip(w, proj))
    return RichResult(payload={
        "estimate": est,
        "weights": w,
        "unweighted_mean": sum(proj) / m_count,
        "d": d,
        "n_models": m_count,
        "effective_n": 1.0 / sum(x * x for x in w),
        "method": "Knutti et al. (2017) Eq. 1 weighting",
    })


def cmip_ensemble(models, weights):
    """Plain weighted ensemble mean (stub-era signature kept)."""
    mods = [float(v) for v in models]
    ws = [float(v) for v in weights]
    if len(mods) != len(ws):
        raise ValueError("models and weights must have equal length")
    tot = sum(ws)
    if tot <= 0:
        raise ValueError("weights must sum to a positive value")
    est = sum(w * x for w, x in zip(ws, mods)) / tot
    return RichResult(payload={
        "estimate": est, "n": len(mods),
        "method": "weighted ensemble mean",
    })


# compact alias per ledger/NAMING.md
cmipensemble = cmip_ensemble


def cheatsheet():
    return "caCMIP: Knutti 2017 performance+independence CMIP weighting"
