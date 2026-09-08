# morie.fn -- function file (rootcoder007/morie)
"""Continuous-shrinkage polygenic effects."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["csshrink", "prscs", "prs_cs"]


def csshrink(beta_hat, D, psi, n, sigma2=1.0):
    """Continuous-shrinkage polygenic effects.

    Posterior mean of the SNP effects under a continuous shrinkage prior.

    beta_post = (D + Psi^-1)^-1 beta_hat, with D the LD matrix and
    Psi = diag(psi) the local shrinkage variances.  Continuous shrinkage
    is what separates PRS-CS from a fixed-threshold score: small effects
    are pulled hard toward zero while large ones are left nearly alone,
    so no p-value cutoff has to be chosen.  ``psi`` and ``sigma2`` are
    supplied rather than sampled, so the result is the conditional
    posterior mean and is reproducible.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Continuous-shrinkage polygenic effects", payload=_c.csshrink(beta_hat=beta_hat, D=D, psi=psi, n=n, sigma2=sigma2))


prs_cs = csshrink


def cheatsheet():
    return "prsccs: Continuous-shrinkage polygenic effects"


# compact alias per ledger/NAMING.md (pre-existing spelling, kept working)
prscs = csshrink
