# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conformalized quantile regression.

Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 17 pp. 514-515; Romano, Patterson and Candes (2019) NeurIPS 32, arXiv:1905.03222
"""

from . import _joseph as _core

from ._richresult import RichResult

__all__ = ["cqr", "joseph_conformalized_quantile_regression"]

_METHOD = "Conformalized quantile regression"


def cqr(callo, calhi, caly, lo, hi, alpha=0.1):
    """Conformalized quantile regression.

    Conformalized quantile regression, ch. 17 pp. 514-515.

    The book's own non-conformity score, quoted from p. 514:

        s(x, y) = max{ yhat_t^{alpha/2} - y , y - yhat_t^{1-(alpha/2)} }

    The conformal quantile of those calibration scores is then added to
    both ends of the test interval.  The rank used is the standard
    finite-sample one, ceil((n+1)(1-alpha))/n, which is what delivers
    the coverage guarantee the section is about.

    -- the method is Romano, Y., Patterson, E. and Candes, E. (2019),
    "Conformalized Quantile Regression", NeurIPS 32 (arXiv:1905.03222),
    which the book cites as its Reference 11.

    Parameters
    ----------
    callo : as documented for the shelf core
        See ``morie.fn._joseph.cqr``.
    calhi : as documented for the shelf core
        See ``morie.fn._joseph.cqr``.
    caly : as documented for the shelf core
        See ``morie.fn._joseph.cqr``.
    lo : as documented for the shelf core
        See ``morie.fn._joseph.cqr``.
    hi : as documented for the shelf core
        See ``morie.fn._joseph.cqr``.
    alpha : as documented for the shelf core
        See ``morie.fn._joseph.cqr``.

    Returns
    -------
    result : RichResult
        Payload keys: qhat, k, meanwidth, widening.

    References
    ----------
    Joseph, M. and Tackes, J. (2024). Modern Time Series Forecasting with Python, 2nd ed. Packt, ch. 17 pp. 514-515; Romano, Patterson and Candes (2019) NeurIPS 32, arXiv:1905.03222
    """
    res = _core.cqr(callo=callo, calhi=calhi, caly=caly, lo=lo, hi=hi, alpha=alpha)
    return RichResult(
        title=_METHOD,
        summary_lines=[("qhat", res["qhat"]), ("k", res["k"]), ("meanwidth", res["meanwidth"]), ("widening", res["widening"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
joseph_conformalized_quantile_regression = cqr


def cheatsheet():
    return "cqr: Conformalized quantile regression"
