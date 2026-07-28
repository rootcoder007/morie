# morie.fn -- function file (rootcoder007/morie)
"""Random common cause refutation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["random_common_cause_refutation"]


def random_common_cause_refutation(estimator, y, d, X, n_sims=50, seed=0,
                                   tol=0.1):
    r"""Add an independent random covariate; the estimate should not move.

    A random common cause is independent of everything, so conditioning
    on it cannot change any causal quantity. If the estimate DOES move,
    the estimator is unstable with respect to irrelevant covariates --
    a property of the fitting procedure, not of the data.

    This is a FALSIFICATION test and its logic runs one way only.
    Passing means the estimator survived one way of being wrong; it is
    not evidence that the identifying assumptions hold, and no number
    of refutations passed can supply that. The test is cheap precisely
    because it does not attempt the hard question.

    A pass is also not free of information about power. If the
    estimator would not move for ANY covariate -- because it ignores
    covariates altogether -- it passes trivially. ``sensitivity_check``
    reports how much the estimate moves when a covariate genuinely
    correlated with the outcome is added instead, which is the control
    that makes the pass meaningful.

    Parameters
    ----------
    estimator : callable
        ``estimator(y, d, X) -> float``.
    y, d, X : array-like
    n_sims : int
    seed : int
    tol : float
        Relative change treated as a failure.

    Returns
    -------
    RichResult
        ``original``, ``refuted_mean``, ``refuted_sd``,
        ``relative_change``, ``passed``, ``sensitivity_check``,
        ``p_value``.

    References
    ----------
    Molak (2023), *Causal Inference and Discovery in Python*, chapter 7,
    refutation tests. Sharma and Kiciman (2020), DoWhy, arXiv:2011.04216.

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn.drblr import doubly_robust_ate
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 2))
    >>> d = (rng.uniform(size=300) < 0.5).astype(float)
    >>> y = 2.0 * d + rng.normal(size=300)
    >>> f = lambda yy, dd, XX: doubly_robust_ate(yy, dd, XX)["estimate"]
    >>> bool(random_common_cause_refutation(f, y, d, X, n_sims=5)["passed"])
    True
    """
    if not callable(estimator):
        raise ValueError("estimator must be callable.")
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    orig = float(estimator(yv, dv, Xa))
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_sims)):
        noise = rng.normal(size=(n, 1))
        vals.append(float(estimator(yv, dv, np.hstack([Xa, noise]))))
    vals = np.asarray(vals)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    denom = max(abs(orig), 1e-12)
    rel = float(abs(mean - orig) / denom)
    # a covariate that genuinely predicts the outcome, as the control
    ctrl = rng.normal(size=n) + yv / max(float(np.std(yv)), 1e-12)
    sens = float(abs(
        float(estimator(yv, dv, np.hstack([Xa, ctrl[:, None]]))) - orig
    ) / denom)
    z = float((mean - orig) / (sd / np.sqrt(vals.size))) if sd > 0 else 0.0
    import math
    return RichResult(
        payload={
            "estimate": mean,
            "original": orig,
            "refuted_mean": mean,
            "refuted_sd": sd,
            "refuted_values": vals,
            "relative_change": rel,
            "passed": bool(rel < tol),
            "tolerance": float(tol),
            "z": z,
            "p_value": float(math.erfc(abs(z) / math.sqrt(2.0))),
            "sensitivity_check": sens,
            "sensitivity_note": (
                "how far the estimate moves when a covariate genuinely "
                "correlated with the outcome is added; if this is also near "
                "zero the estimator ignores covariates and the refutation "
                "passed trivially"
            ),
            "falsification_note": (
                "passing means the estimator survived one way of being "
                "wrong; it is not evidence that the identifying assumptions "
                "hold, and no number of refutations can supply that"
            ),
            "n_sims": int(n_sims),
            "n": int(n),
            "method": "Random common cause refutation",
        }
    )


def cheatsheet():
    return (
        "rcaus: add an independent covariate and check the estimate holds, "
        "with a control that catches a trivial pass"
    )
