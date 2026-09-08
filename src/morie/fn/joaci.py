# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adaptive conformal inference.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 17 p. 519; Gibbs and Candes (2021) NeurIPS 34, arXiv:2106.00170
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["aci", "joseph_adaptive_conformal_inference"]

_METHOD = "Adaptive conformal inference"


def aci(inside, alpha=0.1, gamma=0.01):
    """Adaptive conformal inference.

    Adaptive conformal inference, ch. 17 p. 519.

    The book's own online update, quoted from p. 519:

        err_t = 1 if Y_t is outside Chat(alpha_t), else 0
        alpha_{t+1} = alpha_t + gamma (alpha - err_t)

    with alpha_1 = alpha.  ``inside`` is the sequence of coverage
    outcomes (True when the observation fell inside the interval), so
    the routine is a pure recursion over data the caller supplies.

    -- the method is Gibbs, I. and Candes, E. (2021), "Adaptive
    Conformal Inference Under Distribution Shift", NeurIPS 34
    (arXiv:2106.00170), which the book cites as its Reference 13.

    Parameters
    ----------
    inside : as documented for the shelf core
        See ``morie.fn._joseph.aci``.
    alpha : as documented for the shelf core
        See ``morie.fn._joseph.aci``.
    gamma : as documented for the shelf core
        See ``morie.fn._joseph.aci``.

    Returns
    -------
    result : RichResult
        Payload keys: final, empirical, minalpha, maxalpha.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 17 p. 519; Gibbs and Candes (2021) NeurIPS 34, arXiv:2106.00170
    """
    res = _core.aci(inside=inside, alpha=alpha, gamma=gamma)
    return RichResult(
        title=_METHOD,
        summary_lines=[("final", res["final"]), ("empirical", res["empirical"]), ("minalpha", res["minalpha"]), ("maxalpha", res["maxalpha"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_adaptive_conformal_inference = aci


def cheatsheet():
    return "aci: Adaptive conformal inference"
