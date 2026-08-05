# morie.fn -- function file (rootcoder007/morie)
"""Population attributable fraction (PAF).

DUPLICATE: Levin's PAF is already implemented in ``attfr`` (public name
``attributable_fraction``).  Per ledger/wave2/DUPMAP.tsv this module
aliases it, only reordering the arguments to the (pe, RR) form the
module's own signature declares.
"""

from .attfr import attributable_fraction as _af
from ._richresult import RichResult

__all__ = ["population_attributable"]


def population_attributable(pe, RR, se_RR=None, alpha=0.05):
    """Levin's population attributable fraction.

    Formula: ``PAF = pe (RR - 1) / (pe (RR - 1) + 1)``.

    Alias of :func:`morie.fn.attfr.attributable_fraction` with the
    argument order of this module's signature.

    Parameters
    ----------
    pe : float
        Prevalence of exposure in the population, in [0, 1].
    RR : float
        Relative risk, strictly positive.
    se_RR : float, optional
        Standard error of ``RR``; supplying it adds a delta-method CI.
    alpha : float, default 0.05
        Two-sided CI level.

    Returns
    -------
    RichResult
        ``estimate`` (the PAF), ``se``, ``ci_lower``, ``ci_upper``,
        ``pe``, ``RR``.

    References
    ----------
    Levin, M. L. (1953).  The occurrence of lung cancer in man.  Acta
    Unio Internationalis Contra Cancrum, 9(3), 531--541.
    """
    res = _af(float(RR), float(pe), alpha=float(alpha), se_RR=se_RR)
    return RichResult(payload={
        "estimate": float(res.estimate),
        "se": float(res.se) if res.se is not None else float("nan"),
        "ci_lower": float(res.ci_lower) if res.ci_lower is not None else float("nan"),
        "ci_upper": float(res.ci_upper) if res.ci_upper is not None else float("nan"),
        "pe": float(pe), "RR": float(RR),
        "method": "Levin population attributable fraction"})


def cheatsheet():
    return "pareff: Population attributable fraction (alias of attfr)"


populationattributable = population_attributable
