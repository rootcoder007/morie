# morie.fn -- function file (rootcoder007/morie)
"""Targeted maximum likelihood estimation of the ATE."""

import numpy as np

from ._richresult import RichResult
from ._tmle import tmle_ate as _tmle_ate

__all__ = ["tmle_ate"]


def tmle_ate(y, D, X, trunc=0.01, g=None):
    r"""Targeted maximum likelihood estimate of the average treatment effect.

    Thin front end over the shared engine in :mod:`morie.fn._tmle`,
    which does the three steps: fit :math:`\bar Q(A, W)` and
    :math:`g(W)`, fluctuate :math:`\bar Q` along the clever covariate

    .. math:: H(A, W) = \frac{A}{g(W)} - \frac{1 - A}{1 - g(W)},

    then take the plug-in mean of :math:`\bar Q^*(1, W) -
    \bar Q^*(0, W)`.

    Two properties separate TMLE from the AIPW estimator with the same
    influence function. It is a SUBSTITUTION estimator -- the answer is
    the parameter evaluated at a fitted distribution, so a risk
    difference can never come back outside :math:`[-1, 1]` however
    badly the nuisance models behave, which one-step corrections can
    and do. And the targeting step makes the empirical mean of the
    efficient influence function zero by construction, so the standard
    error comes from the same object that certifies the estimate: the
    returned ``eif_mean`` is a check that the targeting converged, not
    a formality.

    Positivity is where this estimator actually fails. Propensities
    near 0 or 1 make :math:`H` explode, and truncating them trades a
    little bias for a lot of variance. ``n_truncated`` and the
    propensity range are returned so the trade is visible rather than
    silent.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome, binary or bounded continuous.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    trunc : float
        Propensity truncation bound.
    g : array-like, optional
        Known propensities; skips the internal fit. Use this when the
        design assigned treatment, since a fitted propensity would
        add noise to a quantity that is known.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci``, ``ey1``, ``ey0``, ``epsilon``,
        ``eif_mean``, ``propensity``, ``n_truncated``,
        ``positivity_warning``, ``substitution``.

    References
    ----------
    van der Laan and Rubin (2006), *International Journal of
    Biostatistics* 2(1), Article 11.
    Gruber and van der Laan (2010), *IJB* 6(1), Article 26 (the
    logistic fluctuation for bounded continuous outcomes).
    van der Laan and Rose (2011), *Targeted Learning*, chapters 4-5.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> W = rng.normal(size=(4000, 2))
    >>> p = 1 / (1 + np.exp(-(0.8 * W[:, 0])))
    >>> A = (rng.uniform(size=4000) < p).astype(float)
    >>> y = 0.3 + 0.2 * W[:, 0] + 0.25 * A + rng.normal(scale=0.1, size=4000)
    >>> bool(abs(tmle_ate(y, A, W)["estimate"] - 0.25) < 0.05)
    True
    """
    out = _tmle_ate(y, D, X, trunc=trunc, g=g)
    gv = np.asarray(out["g"], dtype=float)
    n_trunc = int(np.sum((gv <= trunc + 1e-15) | (gv >= 1 - trunc - 1e-15)))
    eif = np.asarray(out["eif"], dtype=float)
    return RichResult(
        payload={
            "estimate": float(out["ate"]),
            "se": float(out["se"]),
            "ci": tuple(out["ci"]),
            "ey1": float(out["ey1"]),
            "ey0": float(out["ey0"]),
            "epsilon": float(out["epsilon"]),
            "eif": eif,
            "eif_mean": float(np.mean(eif)),
            "targeting_note": (
                "the fluctuation step sets the empirical mean of the "
                "efficient influence function to zero; eif_mean is the "
                "check that it did"
            ),
            "propensity": {
                "min": float(gv.min()),
                "max": float(gv.max()),
                "mean": float(gv.mean()),
            },
            "n_truncated": n_trunc,
            "trunc": float(trunc),
            "positivity_warning": (
                "%d of %d propensities hit the truncation bound; the clever "
                "covariate is unbounded as g approaches 0 or 1, so these "
                "observations would otherwise dominate the estimate"
                % (n_trunc, gv.size)
                if n_trunc else None
            ),
            "substitution": (
                "TMLE is a plug-in: the estimate is the parameter evaluated "
                "at a fitted distribution and cannot leave the parameter "
                "space, which one-step corrections can"
            ),
            "propensity_supplied": g is not None,
            "n": int(out["n"]),
            "method": "Targeted maximum likelihood estimation of the ATE",
        }
    )


def cheatsheet():
    return (
        "tmleat: TMLE for the ATE -- doubly robust, substitution, with the "
        "influence-function check and positivity diagnostics"
    )
