# morie.fn -- function file (rootcoder007/morie)
"""ARE values at the uniform distribution."""

from ._gb_are import ARE_TABLE, efficacy_are
from ._richresult import RichResult

__all__ = ["gibbons_are_unif"]


def gibbons_are_unif(distribution="uniform", cdf=None):
    r"""Exact asymptotic relative efficiencies at the uniform
    distribution (Gibbons Table 13.3.1, PDF-verified, printed
    p. 492).

    The signed-rank test is fully efficient (ARE = 1) against the t
    at the uniform; the sign test achieves the Hodges-Lehmann lower
    bound 1/3 here.

    The table values are also RE-DERIVED here from the density via
    the efficacy integrals, and both are returned -- the derivation
    agreeing with the table is the check that neither was copied
    wrong.

    Parameters
    ----------
    distribution : str
        Accepted for interface compatibility; must name this module's
        distribution.
    cdf : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``wilcoxon_vs_t``, ``sign_vs_t``, ``sign_vs_wilcoxon``
        (exact), ``derived`` (the same three from the efficacy
        integrals), ``distribution``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Table 13.3.1.
    """
    if distribution not in ("uniform",):
        raise ValueError(
            f"this module carries the uniform case, got {distribution!r}."
        )
    from morie.fn import _array_core as np
    from . import _stats_core as stats

    dens = {
        "uniform": lambda x: stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        "normal": stats.norm.pdf,
        "logistic": stats.logistic.pdf,
        "double_exponential": stats.laplace.pdf,
    }["uniform"]
    derived = efficacy_are(dens)
    exact = ARE_TABLE["uniform"]
    return RichResult(
        payload={
            **exact,
            "derived": {k: derived[k] for k in exact},
            "distribution": "uniform",
            "method": "Gibbons Table 13.3.1 + efficacy re-derivation",
        }
    )


def cheatsheet():
    return "gb_ar6: Table 13.3.1 at the uniform; re-derived from the density"


# compact alias per ledger/NAMING.md
gibbonsareunif = gibbons_are_unif
