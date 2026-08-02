# morie.fn -- function file (rootcoder007/morie)
"""GP density estimation contraction rate via concentration function."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_gp_dens_crt"]


def ghosal_gp_dens_crt(x, s=None, n=None, kernel="squared_exponential"):
    r"""Contraction rate for Gaussian-process density estimation
    (Ghosal Sec. 11.3.1).

    The density is built as :math:`f = e^{\psi}/\int e^{\psi}` with
    :math:`\psi` a Gaussian process, which forces positivity and
    integration to one automatically -- the reason the exponential
    link is used rather than a prior on f directly.

    The rate follows from the CONCENTRATION FUNCTION

    .. math:: \varphi_{\psi_0}(\varepsilon)
              = \inf_{h:\|h-\psi_0\|_\infty<\varepsilon}
                \tfrac12\|h\|^2_{\mathbb H}
              - \log \Pr\big(\|\psi\|_\infty < \varepsilon\big),

    solved for :math:`\varphi_{\psi_0}(\varepsilon_n)
    \le n\varepsilon_n^2`. Its two terms are the whole story: a
    reproducing-kernel-Hilbert-space approximation term and a small-ball
    probability term, and the rate is where they balance.

    The kernel choice then decides the answer. A Matern process with
    smoothness matched to :math:`\psi_0` attains the minimax
    :math:`n^{-s/(2s+1)}`. A SQUARED-EXPONENTIAL process is far too
    smooth -- its sample paths are analytic -- so for a merely
    s-smooth truth it contracts only at a LOGARITHMIC rate unless its
    length-scale is itself given a prior, and the rescaled version
    recovers a polynomial rate. That contrast is returned rather than
    buried.

    Parameters
    ----------
    x : array-like
        Observations; used for the sample size.
    s : float, optional
        Smoothness of the log-density; 1.0 otherwise.
    n : int, optional
        Sample size.
    kernel : {"squared_exponential", "matern", "rescaled_se"}
        Which prior is used.

    Returns
    -------
    RichResult
        keys: ``n``, ``smoothness``, ``kernel``, ``rate``,
        ``minimax_rate``, ``attains_minimax``, ``rate_kind``,
        ``link``, ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 11.3.1 (density estimation) and
    Ch. 11 (the concentration function); van der Vaart and van Zanten.
    """
    from ._ghosal import minimax_rate

    xv = np.asarray(x, dtype=float).ravel()
    nn = int(xv.size) if n is None else int(n)
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    sv = 1.0 if s is None else float(s)
    if sv <= 0:
        raise ValueError(f"smoothness must be positive, got {sv}.")
    if kernel not in ("squared_exponential", "matern", "rescaled_se"):
        raise ValueError("kernel must be 'squared_exponential', 'matern' "
                         "or 'rescaled_se'.")
    mm = minimax_rate(nn, sv)
    if kernel == "matern":
        rate, kind, attains = mm, "polynomial (minimax)", True
    elif kernel == "rescaled_se":
        rate = mm * float(np.log(nn)) ** ((sv + 1.0) / (2.0 * sv + 1.0))
        kind, attains = "polynomial up to a log factor", False
    else:
        rate = float(np.log(nn) ** (-sv))
        kind, attains = "LOGARITHMIC", False
    return RichResult(payload={
        "n": nn, "smoothness": sv, "kernel": kernel, "rate": float(rate),
        "minimax_rate": mm, "attains_minimax": attains, "rate_kind": kind,
        "ratio_to_minimax": float(rate / mm),
        "link": "f = exp(psi) / int exp(psi): positivity and normalisation for free",
        "driver": "the concentration function: RKHS approximation + small-ball probability",
        "method": "GP density contraction (Sec. 11.3.1); the kernel's smoothness decides the rate"})


def cheatsheet():
    return "gh_c11_4: a squared-exponential GP is TOO smooth -- logarithmic rate unless rescaled"
