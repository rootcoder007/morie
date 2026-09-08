# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Bayesian ridge for genomic prediction -- the same method as brreg.

The specification here is "u_j ~ N(0, sigma^2) with conjugate prior",
citing Park and Casella (2008).  A normal prior with a common variance
on every marker effect, combined with a normal likelihood, has the
ridge estimator as its posterior mode,

    u_hat = (X'X + lambda I)^{-1} X'y,   lambda = sigma_e^2 / sigma_u^2,

which is RR-BLUP.  That is precisely what ``morie.fn.brreg``
("Bayesian ridge regression (RR-BLUP) for marker effects") already
implements, with both R arms present and verified in parity.

There is therefore exactly one implementation: this module delegates.
Note on the citation -- Park and Casella (2008) is the Bayesian *lasso*
(a Laplace prior), not the ridge; the ridge with a conjugate normal
prior is the Lindley and Smith (1972) / Hoerl and Kennard (1970)
construction.  The stub's attribution is repeated here only to record
that it was checked and is wrong for the formula the stub states; the
formula, not the label, is what has been implemented.

Recorded in ledger/wave2/DUPMAP.tsv as baysrr -> brreg.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

from .brreg import bayesian_ridge_regression as _ridge

__all__ = ["bayes_ridge"]


def bayes_ridge(y, M=None, lam=None):
    """Ridge / RR-BLUP marker effects.

    Parameters
    ----------
    y : array-like
        The n phenotypes.
    M : array-like
        The n-by-p marker matrix.
    lam : float, optional
        Ridge parameter sigma_e^2 / sigma_u^2; the brreg default when
        omitted.

    Returns
    -------
    u : the marker effects
    lam : the ridge parameter used
    """
    if M is None:
        raise ValueError("bayes_ridge: a marker matrix is required")
    r = _ridge(M, y, lam)
    beta = list(r["beta"])
    lm = r["lam"] if "lam" in r else r["lambda"]
    return RichResult(
        title="Bayesian ridge (RR-BLUP)",
        summary_lines=[("p", len(beta)), ("lam", lm)],
        payload={
            "u": beta,
            "beta": beta,
            "estimate": beta[0] if beta else float("nan"),
            "lam": lm,
            "n": r["n"],
            "p": r["p"],
            "method": "posterior mode of a conjugate normal prior = ridge; shared implementation with morie.fn.brreg",
        },
    )


def cheatsheet():
    return "baysrr: Bayesian ridge (shares brreg's implementation)"


# compact alias per ledger/NAMING.md
bayesridge = bayes_ridge
