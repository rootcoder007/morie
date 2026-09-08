# morie.fn -- tail3 batch (rootcoder007/morie)
"""Hasselmann stochastic climate model: AR(1) red-noise fit.

Source consulted: Hasselmann, K. (1976). Stochastic climate models, Part I.
Theory.  *Tellus* 28(6), 473-485 (doi 10.3402/tellusa.v28i6.11316).  Slow
climate variables are driven by fast weather fluctuations that act as white
noise; without feedback the response is a random walk whose variance grows
without bound, and with stabilising linear feedback it becomes a stationary
first-order Markov process with a red variance spectrum.  Discretely,

    x_t = phi x_{t-1} + eps_t,   eps_t ~ white noise

with phi estimated by the lag-one autocorrelation (Yule-Walker), the
innovation variance sigma2_eps = var(x) (1 - phi^2), the decorrelation time
tau = -dt / log(phi), and the red spectrum

    S(f) = sigma2_eps dt / ( 1 - 2 phi cos(2 pi f dt) + phi^2 ) .
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ar1_climate"]


def ar1_climate(x, phi=None, dt=1.0, freq=None):
    """Fit and summarise the AR(1) red-noise climate model.

    Parameters
    ----------
    x : array-like
        Time series of the slow (climate) variable.
    phi : float, optional
        Lag-one coefficient; estimated by Yule-Walker when omitted.
    dt : float
        Sampling interval.
    freq : array-like, optional
        Frequencies at which to report the red spectrum.

    Returns
    -------
    RichResult
        estimate (phi), phi, tau, sigma2_eps, var, spectrum, n, method.

    References
    ----------
    Hasselmann (1976), Tellus 28(6), 473-485.
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = int(xs.size)
    m = float(np.mean(xs))
    c0 = sum((float(xs[i]) - m) ** 2 for i in range(n)) / n
    c1 = sum((float(xs[i]) - m) * (float(xs[i - 1]) - m) for i in range(1, n)) / n
    ph = c1 / c0 if phi is None and c0 > 0.0 else (float(phi) if phi is not None else float("nan"))
    varx = c0 * n / (n - 1) if n > 1 else float("nan")
    s2 = c0 * (1.0 - ph * ph)
    tau = -float(dt) / float(np.log(ph)) if 0.0 < ph < 1.0 else float("inf")
    if freq is None:
        fv = np.asarray([0.0, 0.125, 0.25, 0.375, 0.5], dtype=float)
    else:
        fv = np.atleast_1d(np.asarray(freq, dtype=float)).ravel()
    spec = []
    for i in range(int(fv.size)):
        w = 2.0 * float(np.pi) * float(fv[i]) * float(dt)
        spec.append(s2 * float(dt) / (1.0 - 2.0 * ph * float(np.cos(w)) + ph * ph))
    return RichResult(
        payload={
            "estimate": float(ph),
            "phi": float(ph),
            "tau": float(tau),
            "sigma2_eps": float(s2),
            "var": float(varx),
            "c0": float(c0),
            "c1": float(c1),
            "spectrum": np.asarray(spec, dtype=float),
            "freq": fv,
            "n": n,
            "method": "AR(1) stochastic climate model (Hasselmann 1976)",
        }
    )


# CANONICAL TEST
# >>> # a pure alternating series has lag-1 autocorrelation near -1
# >>> r = ar1_climate([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
# >>> assert r["phi"] < -0.7
# >>> # white-noise-like input with phi supplied leaves sigma2 = c0 (1 - phi^2)
# >>> r2 = ar1_climate([1.0, -1.0, 1.0, -1.0], phi=0.0)
# >>> assert abs(r2["sigma2_eps"] - 1.0) < 1e-12


def cheatsheet():
    return "ar1cl(x, phi, dt): AR(1) red-noise climate fit + spectrum."


# compact alias per ledger/NAMING.md (registered in _lazy_map.json)
ar1climate = ar1_climate
