# morie.fn -- function file (rootcoder007/morie)
"""ARE values at the logistic distribution."""

from ._gb_are import ARE_TABLE, efficacy_are
from ._richresult import RichResult

__all__ = ["gibbons_are_logistic"]


def gibbons_are_logistic(distribution="logistic", cdf=None):
    r"""Exact asymptotic relative efficiencies at the logistic
    distribution (Gibbons Table 13.3.1, PDF-verified, printed
    p. 492).

    The logistic is the Wilcoxon's home game: ARE = pi^2/9 = 1.097 > 1,
    the signed-rank test strictly beats the t.

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
    if distribution not in ("logistic",):
        raise ValueError(
            f"this module carries the logistic case, got {distribution!r}."
        )
    import numpy as np
    from scipy import stats

    dens = {
        "uniform": lambda x: stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
        "normal": stats.norm.pdf,
        "logistic": stats.logistic.pdf,
        "double_exponential": stats.laplace.pdf,
    }["logistic"]
    derived = efficacy_are(dens)
    exact = ARE_TABLE["logistic"]
    return RichResult(
        payload={
            **exact,
            "derived": {k: derived[k] for k in exact},
            "distribution": "logistic",
            "method": "Gibbons Table 13.3.1 + efficacy re-derivation",
        }
    )


def cheatsheet():
    return "gb_ar7: Table 13.3.1 at the logistic; re-derived from the density"
