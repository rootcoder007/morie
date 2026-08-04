# morie.fn -- function file (rootcoder007/morie)
"""AR model order selection by Akaike's information criterion."""

import math as _math

from ._richresult import RichResult

__all__ = ["rangayyan_ar_order_aic", "aicorder"]


def rangayyan_ar_order_aic(prediction_errors, n_samples, window="hamming"):
    r"""Akaike order selection for an AR model, Rangayyan eq. (7.60).

    .. math:: I(P) = \log \varepsilon_P + \frac{2P}{N_e}

    where :math:`\varepsilon_P` is the total squared prediction error of
    the order-:math:`P` model and :math:`N_e` is the EFFECTIVE number of
    data points after windowing -- the book gives :math:`N_e = 0.4N` for
    a Hamming window.  The chosen order is the one minimising
    :math:`I(P)`.

    This is not the textbook :math:`N\log\sigma^2 + 2p` form: Rangayyan
    normalises by the effective count, which is what windowed data
    actually supplies, and takes the log of the error directly.  The
    placeholder docstring for this module stated the textbook form; the
    book is followed here.

    Parameters
    ----------
    prediction_errors : sequence
        Total squared prediction error for orders 1, 2, ..., P_max.
    n_samples : int
        Number of data samples N.
    window : str or float
        "hamming" (N_e = 0.4 N), "rectangular"/"none" (N_e = N), or a
        float giving the effective-sample fraction directly.

    Returns
    -------
    RichResult
        ``order`` (minimising order), ``criterion`` (I(P) per order),
        ``n_effective``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*, 3rd ed.
    Wiley-IEEE Press, eq. (7.60).  Akaike, H. (1974). A new look at the
    statistical model identification. *IEEE Trans. Automatic Control*
    19(6), 716-723.
    """
    eps = [float(v) for v in prediction_errors]
    if not eps:
        raise ValueError("need at least one prediction error")
    if any(v <= 0 for v in eps):
        raise ValueError("prediction errors must be positive")
    n = int(n_samples)
    if n <= 0:
        raise ValueError("n_samples must be positive")

    if isinstance(window, str):
        frac = {"hamming": 0.4, "rectangular": 1.0, "none": 1.0}.get(
            window.lower())
        if frac is None:
            raise ValueError("unknown window %r" % window)
    else:
        frac = float(window)
        if not (0.0 < frac <= 1.0):
            raise ValueError("effective-sample fraction must be in (0, 1]")
    n_eff = frac * n
    if n_eff <= 0:
        raise ValueError("effective sample size must be positive")

    crit = [_math.log(e) + 2.0 * (p + 1) / n_eff for p, e in enumerate(eps)]
    best = min(range(len(crit)), key=lambda i: crit[i])
    return RichResult(
        title="Akaike order selection (Rangayyan eq. 7.60)",
        summary_lines=[("order", best + 1), ("min I(P)", crit[best])],
        payload={"order": best + 1, "criterion": crit,
                 "n_effective": n_eff,
                 "method": "Rangayyan (2024) eq. (7.60)"},
    )


aicorder = rangayyan_ar_order_aic


def cheatsheet():
    return "aicorder: I(P) = log eps_P + 2P/Ne, Ne = 0.4N for Hamming"
