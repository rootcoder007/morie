# morie.fn -- function file (rootcoder007/morie)
"""Deconvolution density estimate."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hrz_deconvolution", "horowitz_deconvolution_density"]


def hrz_deconvolution(W, sigma_eps, grid=None, h=None, error="normal"):
    r"""Deconvolution kernel density estimate (Horowitz Sec. 5.1):

    .. math:: \hat f_U(u) = \frac{1}{2\pi}\int e^{-i\tau u}
              \frac{\psi_W(\tau)}{\psi_\epsilon(\tau)}\, d\tau,

    recovering the density of U from contaminated observations
    :math:`W = U + \epsilon`. Dividing by the error characteristic
    function is the whole idea and also the whole difficulty: for a
    NORMAL error :math:`\psi_\epsilon` decays like
    :math:`e^{-\sigma^2\tau^2/2}`, so the ratio explodes and the
    rate collapses to a LOGARITHMIC :math:`(\log n)^{-s}` -- the
    supersmooth case. For an ordinary-smooth error (Laplace) it decays
    polynomially and the rate stays polynomial. A damping kernel in
    the integrand is therefore mandatory, not optional.

    Parameters
    ----------
    W : array-like
        Contaminated observations.
    sigma_eps : float > 0
        Error scale.
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Damping bandwidth. The default is set from the noise-
        amplification criterion rather than a generic rate: the
        estimator variance carries :math:`\int |\hat K/\psi_\epsilon|^2`,
        so the cut-off :math:`T = 1/h` may only grow as fast as that
        integral divided by ``n`` stays bounded. Normal error gives
        :math:`e^{\sigma^2 T^2}/n = O(1)`, i.e.
        :math:`h = \sigma/\sqrt{\log n}`; Laplace error gives
        :math:`T^5/n = O(1)`, i.e. :math:`h = n^{-1/5}`. A fixed
        :math:`n^{-1/8}` (the previous default) cuts off far too
        early -- at ``n = 3000, sigma = 0.4`` it returns 0.275 for a
        standard normal density at 0 against the true 0.399, worse
        than not deconvolving at all.
    error : {"normal", "laplace"}
        Error type; sets the rate regime.

    Returns
    -------
    RichResult
        keys: ``grid``, ``density``, ``bandwidth``, ``regime``
        (supersmooth/ordinary smooth), ``rate_note``, ``n``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 5, Sec. 5.1 (deconvolution in a model of
    measurement error).
    """
    W = np.asarray(W, dtype=float).ravel()
    n = W.size
    if n < 8:
        raise ValueError(f"need at least 8 observations, got {n}.")
    s = float(sigma_eps)
    if s <= 0:
        raise ValueError(f"sigma_eps must be positive, got {s}.")
    if error not in ("normal", "laplace"):
        raise ValueError("error must be 'normal' or 'laplace'.")
    if h is None:
        # cut-off from the variance criterion, see the h docstring
        hh = s / np.sqrt(np.log(n)) if error == "normal" else n ** (-0.2)
        hh = float(hh)
    else:
        hh = float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    g = np.linspace(W.min(), W.max(), 200) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))

    # sinc-kernel Fourier transform: compactly supported in tau, which
    # is what keeps 1/psi_eps from being evaluated where it vanishes
    T = 1.0 / hh
    tau = np.linspace(-T, T, 2001)
    psi_W = np.exp(1j * np.outer(tau, W)).mean(axis=1)
    if error == "normal":
        psi_e = np.exp(-0.5 * s**2 * tau**2)
        regime = "supersmooth"
        note = "psi_eps decays exponentially: rate is logarithmic, (log n)^{-s}"
    else:
        psi_e = 1.0 / (1.0 + s**2 * tau**2)
        regime = "ordinary smooth"
        note = "psi_eps decays polynomially: rate stays polynomial, n^{-r}"
    damp = (1.0 - (tau / T) ** 2) ** 3  # vanishes at the cut-off
    integrand = psi_W / psi_e * damp
    dens = np.array([
        float(np.real(np.trapezoid(integrand * np.exp(-1j * tau * u), tau)) / (2 * np.pi))
        for u in g
    ])
    return RichResult(payload={"grid": g, "density": dens, "bandwidth": hh,
                               "regime": regime, "rate_note": note, "n": int(n),
                               "method": "Fourier deconvolution with a compact damping kernel"})


def cheatsheet():
    return "hrzdeconv: normal error => LOGARITHMIC rate; damping is mandatory"


#: Catalogue alias for :func:`hrz_deconvolution`.
horowitz_deconvolution_density = hrz_deconvolution
