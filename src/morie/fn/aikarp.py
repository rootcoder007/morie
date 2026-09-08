# morie.fn -- function file (rootcoder007/morie)
"""AIC order selection for an autoregression."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['aicar', 'aic_ar_order', 'aicarorder']


def aicar(x, max_p=10, demean=True):
    """AIC order selection for an autoregression.

    The per-observation normalisation printed here is a rescaling of the usual -2 log L + 2k and picks the same order; the unnormalised T log sigmahat^2 + 2(p+1) is returned as well so the value can be compared with software that uses the other convention. The innovation variances come from the Levinson-Durbin recursion on the sample autocovariances, so the whole table costs one pass.


    Formula: AIC(p) = log(sigmahat_p^2) + 2(p+1)/T, sigmahat_p^2 from the Levinson-Durbin recursion

    Parameters
    ----------
    x : array-like
        The series.
    max_p : int
        Largest order considered.
    demean : bool
        Subtract the sample mean first.

    Returns
    -------
    RichResult
        ``p``, ``aic``, ``aic_unnormalised``, ``sigma2``, ``pacf``, ``n``.

    References
    ----------
    Akaike (1973), Information theory and an extension of the maximum
    likelihood principle, in Petrov and Csaki (eds), 2nd International
    Symposium on Information Theory.  Not held locally; AIC = -2 log L +
    2k and its AR(p) specialisation via the Levinson-Durbin innovation
    variance are the standard published forms.
    """
    x = C.vec(x)
    T = len(x)
    P = int(max_p)
    if T < P + 2:
        raise ValueError("series too short for max_p")
    mu = sum(x) / T if demean else 0.0
    z = [v - mu for v in x]
    g = [sum(z[t] * z[t - k] for t in range(k, T)) / T for k in range(P + 1)]
    if g[0] <= 0:
        raise ValueError("series has zero variance")
    sig = [g[0]]
    phi = []
    pacf = []
    for k in range(1, P + 1):
        num = g[k] - sum(phi[j] * g[k - 1 - j] for j in range(k - 1))
        kk = num / sig[k - 1]
        pacf.append(kk)
        newphi = [phi[j] - kk * phi[k - 2 - j] for j in range(k - 1)] + [kk]
        phi = newphi
        sig.append(sig[k - 1] * (1.0 - kk * kk))
    aic = [math.log(sig[p]) + 2.0 * (p + 1) / T if sig[p] > 0 else float("inf")
           for p in range(P + 1)]
    unn = [T * math.log(sig[p]) + 2.0 * (p + 1) if sig[p] > 0 else float("inf")
           for p in range(P + 1)]
    best = min(range(P + 1), key=lambda p: aic[p])
    return RichResult(payload={
        "p": best, "aic": aic, "aic_unnormalised": unn, "sigma2": sig,
        "pacf": pacf, "n": T, "method": "AIC order selection for AR(p)"})


aic_ar_order = aicar
aicarorder = aicar


def cheatsheet():
    return "aikarp: AIC order selection for an autoregression."
