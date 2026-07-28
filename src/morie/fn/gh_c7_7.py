# morie.fn -- function file (rootcoder007/morie)
"""Spectral density estimation consistency via Whittle likelihood."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ghosal_spec_dens_con"]


def ghosal_spec_dens_con(x, spectral_density=None, grid=None):
    r"""Whittle-likelihood spectral density estimation and its
    consistency (Ghosal Sec. 7.3.3):

    .. math:: \log L_W(f) = -\sum_j\Big[\log f(\omega_j)
              + \frac{I(\omega_j)}{f(\omega_j)}\Big].

    The Whittle likelihood replaces the exact Gaussian likelihood of
    a stationary series, whose covariance determinant costs
    :math:`O(n^3)`, with a sum over Fourier frequencies. What makes
    that legitimate is that the periodogram ordinates
    :math:`I(\omega_j)` are ASYMPTOTICALLY INDEPENDENT exponentials
    with mean :math:`f(\omega_j)` -- dependence in the time domain
    becomes independence in the frequency domain, and a
    nonparametric prior on f becomes tractable.

    It is an approximation, and the module says so: ``exact`` is
    False. Posterior consistency under it is a genuine theorem
    rather than a corollary of the i.i.d. theory, because the
    likelihood is not the true one.

    With no ``spectral_density`` supplied this returns the
    periodogram and the Whittle log-likelihood of a smoothed
    reference, so the object is computed rather than described.

    Parameters
    ----------
    x : array-like
        A stationary time series.
    spectral_density : callable, optional
        ``f(omega)``, strictly positive. A smoothed periodogram is
        used otherwise.
    grid : array-like, optional
        Frequencies to report on.

    Returns
    -------
    RichResult
        keys: ``freqs``, ``periodogram``, ``spectral_density``,
        ``whittle_loglik``, ``exact`` (False),
        ``periodogram_independence``, ``n``, ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 7.3.3 (spectral density
    estimation); Whittle estimation also at Sec. 9.5.2 and 10.4.6.
    """
    from ._ghosal import whittle_loglik

    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 8:
        raise ValueError(f"need at least 8 observations, got {xv.size}.")
    n = xv.size
    fft = np.fft.rfft(xv - xv.mean())
    per = (np.abs(fft) ** 2) / n
    w = np.fft.rfftfreq(n, d=1.0) * 2 * np.pi
    keep = (w > 0) & (w < np.pi)
    w0, per0 = w[keep], per[keep]
    if spectral_density is None:
        # a smoothed periodogram, floored so the likelihood exists
        k = max(3, per0.size // 10)
        pad = np.r_[per0[:k][::-1], per0, per0[-k:][::-1]]
        sm = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
        sm = sm[k:k + per0.size]
        floor = max(float(np.mean(per0)) * 1e-3, 1e-12)
        ref = np.maximum(sm, floor)

        def sd(v):
            return float(np.interp(v, w0, ref))
    else:
        sd = spectral_density
    ll, wout, perout = whittle_loglik(xv, sd)
    g = wout if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    return RichResult(payload={
        "freqs": g,
        "periodogram": perout if grid is None else np.interp(g, wout, perout),
        "spectral_density": np.array([float(sd(v)) for v in g]),
        "whittle_loglik": ll, "exact": False,
        "periodogram_independence": "asymptotically independent exponentials "
                                    "with mean f(omega_j)",
        "n": int(n),
        "method": "Whittle likelihood (Sec. 7.3.3); an approximation, and consistency is its own theorem"})


def cheatsheet():
    return "gh_c7_7: time-domain dependence becomes frequency-domain independence -- that is the trick"
