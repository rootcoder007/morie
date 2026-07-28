# morie.fn -- function file (rootcoder007/morie)
"""Nadaraya kernel distribution function estimator (1964)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_kdfe"]


def fauzi_kdfe(x, grid=None, h=None):
    r"""Nadaraya's kernel distribution function estimator
    (Fauzi Eq. 2.2):

    .. math:: \hat F_h(x) = \frac1n\sum_{i=1}^{n}
              W\!\left(\frac{x - X_i}{h}\right),
              \qquad x \in \mathbb R,

    with :math:`W` the INTEGRATED kernel
    :math:`W(u) = \int_{-\infty}^{u}K(v)dv`.

    Smoothing a distribution function uses the kernel's integral,
    not the kernel, and that changes the bias term: it carries
    :math:`f'` where the density estimator carries :math:`f''`, and
    it is :math:`h^2\mu_2(K)f'(x)/2 + o(h^2)`.

    The estimator is worth having over the empirical df for a reason
    that is easy to state and easy to forget: the empirical df is a
    step function, so it is a poor estimate of a CONTINUOUS
    distribution, cannot be inverted smoothly for quantiles, and has
    no density to differentiate. The smoothed version fixes all
    three, and is monotone by construction because :math:`W` is.

    Parameters
    ----------
    x : array-like
        Sample.
    grid : array-like, optional
        Evaluation points.
    h : float, optional
        Bandwidth.

    Returns
    -------
    RichResult
        keys: ``grid``, ``F_hat``, ``F_empirical``, ``bandwidth``,
        ``monotone``, ``bias_term``, ``uses_integrated_kernel``
        (True), ``n``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Eq. (2.2) and Sec. 5.3.2;
    Nadaraya (1964); Azzalini, A. (1981), "A note on the estimation
    of a distribution function and quantiles by a kernel method",
    *Biometrika* 68:326-328 (reference [9] of the book).
    """
    from ._fauzi import kdfe_bandwidth, kernel_W

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    # n^{-1/3}, NOT the n^{-1/5} density rule. Sec. 5.3.2 of the book
    # is explicit: "Azzalini in [9] recommended a bandwidth of
    # c n^{-1/3} for the estimation of the distribution function",
    # and the book's own simulations use h_n = n^{-1/3}. The reason
    # is that the leading variance term of a df estimator is
    # F(1-F)/n - (h/n) f(x) r, so h enters the variance at order h/n
    # rather than 1/(nh); minimising that against the h^4 bias gives
    # a cube root. Using the density rule oversmooths badly enough
    # that the estimator loses to the empirical df it is meant to
    # improve on.
    hh = kdfe_bandwidth(xv) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    g = np.linspace(xv.min() - 3 * hh, xv.max() + 3 * hh, 200) \
        if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    F = kernel_W((g[:, None] - xv[None, :]) / hh).sum(axis=1) / n
    emp = np.array([float(np.mean(xv <= v)) for v in g])
    return RichResult(payload={
        "grid": g, "F_hat": F, "F_empirical": emp, "bandwidth": hh,
        "bandwidth_rate": "n^{-1/3} (Azzalini), not the n^{-1/5} density rule",
        "monotone": bool(np.all(np.diff(F) >= -1e-12)),
        "bias_term": "h^2 mu_2(K) f'(x)/2 + o(h^2): f PRIME, not f double prime",
        "uses_integrated_kernel": True,
        "why_over_edf": "the empirical df is a step function: not continuous, "
                        "not smoothly invertible, and has no density",
        "n": int(n),
        "method": "Nadaraya KDFE (2.2); smooths with W = integral of K, so the bias carries f'"})


def cheatsheet():
    return "fzkdfe: smooth with the kernel's INTEGRAL -- bias carries f', not f''"
