# morie.fn -- function file (rootcoder007/morie)
"""Unified mixed-model per-SNP association test."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["gwasmlm", "gwaslinear", "gwas_linear"]


def gwasmlm(y, X, snp, Vinv):
    """Unified mixed-model per-SNP association test.

    Per-SNP test of b = 0 in y = X mu + b snp + Z u + e.

    The unified mixed model absorbs population structure and kinship
    into the covariance of u; conditioning on it, the SNP effect is a
    generalised least squares coefficient.  ``Vinv`` is the inverse of
    the fitted total covariance, supplied by the caller so the variance
    components are estimated once rather than per SNP -- which is the
    computational point of the method.  The reported statistic is the
    Wald t on b.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Unified mixed-model per-SNP association test", payload=_c.gwasmlm(y=y, X=X, snp=snp, Vinv=Vinv))


gwas_linear = gwasmlm


def cheatsheet():
    return "gwasl1: Unified mixed-model per-SNP association test"


# compact alias per ledger/NAMING.md (pre-existing spelling, kept working)
gwaslinear = gwasmlm
