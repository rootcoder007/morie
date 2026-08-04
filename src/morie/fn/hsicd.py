# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hilbert-Schmidt independence criterion.

Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 13 p. 354 (formula not printed there)
"""

from . import _molak as _core

from ._richresult import RichResult

__all__ = ["hsicstat", "hsic_independence"]

_METHOD = "Hilbert-Schmidt independence criterion"


def hsicstat(a, b, sigma_a=None, sigma_b=None, threshold=0.01):
    """Hilbert-Schmidt independence criterion.

    Hilbert-Schmidt independence criterion for an ANM residual test.

    The corpus copy only CALLS gCastle's ``hsic_test`` (ch. 13, p. 354)
    and prints no formula; the book's own citation for the criterion is
    Gretton et al. (2007).  The estimator here is the existing biased
    V-statistic in ``morie.fn.anmod.hsic``, ``tr(K H L H)/n^2`` with
    RBF Gram matrices and the median-heuristic bandwidth, reused rather
    than reimplemented.  ``threshold`` is a caller-supplied cutoff --
    no null distribution is simulated, so nothing here is random.

    Parameters
    ----------
    a : as documented for the shelf core
        See ``morie.fn._molak.hsicstat``.
    b : as documented for the shelf core
        See ``morie.fn._molak.hsicstat``.
    sigma_a : as documented for the shelf core
        See ``morie.fn._molak.hsicstat``.
    sigma_b : as documented for the shelf core
        See ``morie.fn._molak.hsicstat``.
    threshold : as documented for the shelf core
        See ``morie.fn._molak.hsicstat``.

    Returns
    -------
    result : RichResult
        Payload keys: hsic, nhsic, independent.

    References
    ----------
    Molak, A., Causal Inference and Discovery in Python, Packt (corpus copy: 2023 first edition), ch. 13 p. 354 (formula not printed there)
    """
    res = _core.hsicstat(a=a, b=b, sigma_a=sigma_a, sigma_b=sigma_b, threshold=threshold)
    return RichResult(
        title=_METHOD,
        summary_lines=[("hsic", res["hsic"]), ("nhsic", res["nhsic"]), ("independent", res["independent"])],
        payload=dict(res, method=_METHOD),
    )


# legacy spelling from the extraction pipeline -- kept working per
# ledger/NAMING.md ("renames always leave the old spelling working")
hsic_independence = hsicstat


def cheatsheet():
    return "hsicstat: Hilbert-Schmidt independence criterion"
