# morie.fn -- function file (rootcoder007/morie)
"""ARE values at the double_exponential distribution."""

from ._gb_are import ARE_TABLE, efficacy_are
from ._richresult import RichResult

__all__ = ["gibbons_are_dbl_exp"]


def gibbons_are_dbl_exp(distribution="double_exponential", cdf=None):
    r"""Exact asymptotic relative efficiencies at the double_exponential
    distribution (Gibbons Table 13.3.1, PDF-verified, printed
    p. 492).

    Heavy tails reverse the ranking: the sign test BEATS the t test
    (ARE = 2) and beats the signed-rank test too (4/3), because the
    median is the MLE location for the Laplace.

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
    if distribution not in ("double_exponential",):
        raise ValueError(
            f"this module carries the double_exponential case, got {distribution!r}."
        )
    import numpy as np
    from scipy import stats

    dens = {
        "uniform": lambda x: stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        "normal": stats.norm.pdf,
        "logistic": stats.logistic.pdf,
        "double_exponential": stats.laplace.pdf,
    }["double_exponential"]
    derived = efficacy_are(dens)
    exact = ARE_TABLE["double_exponential"]
    return RichResult(
        payload={
            **exact,
            "derived": {k: derived[k] for k in exact},
            "distribution": "double_exponential",
            "method": "Gibbons Table 13.3.1 + efficacy re-derivation",
        }
    )


def cheatsheet():
    return "gb_are3: Table 13.3.1 at the double_exponential; re-derived from the density"
