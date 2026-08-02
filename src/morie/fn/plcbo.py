# morie.fn -- function file (rootcoder007/morie)
"""Placebo (permutation) refutation test for a causal estimate."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["placebo_refutation"]


def placebo_refutation(estimator, y, treatment, n_simulations=500, seed=0, **kwargs):
    r"""Replace the treatment with a random placebo and re-estimate.

    A causal estimate should collapse to zero when the treatment
    assignment is replaced by an independent permutation of itself: any
    remaining effect is coming from the estimator or the covariates,
    not the treatment. The placebo distribution also supplies a
    randomisation p-value

    .. math:: p = \frac{1 + \#\{|\hat\tau^{placebo}|
              \ge |\hat\tau|\}}{1 + B},

    which needs no distributional assumption.

    Parameters
    ----------
    estimator : callable
        ``estimator(y, treatment, **kwargs)`` returning a float or a
        mapping containing one of ``estimate``, ``ate``, ``att``.
    y : array-like, shape (n,)
        Outcome.
    treatment : array-like, shape (n,)
        Observed treatment; permuted to build the placebo.
    n_simulations : int, default 500
        Number of placebo draws.
    seed : int, default 0
        RNG seed.
    **kwargs :
        Passed through to ``estimator`` unchanged.

    Returns
    -------
    RichResult
        keys: ``estimate`` (on the real treatment), ``placebo_mean``,
        ``placebo_sd``, ``p_value``, ``passes`` (True when the real
        estimate is extreme relative to the placebos at the 5% level),
        ``placebo`` (all draws), ``n_simulations``, ``method``.

    References
    ----------
    Sharma, A. & Kiciman, E. (2020). DoWhy: an end-to-end library for
    causal inference. arXiv:2011.04216. (placebo-treatment refuter)

    Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic
    control methods for comparative case studies. *JASA*, 105(490),
    493-505. (placebo/permutation inference for a single treated unit)
    """
    if not callable(estimator):
        raise ValueError("estimator must be callable.")
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(treatment).ravel()
    if t.size != y.size:
        raise ValueError("y and treatment must have equal length.")
    B = int(n_simulations)
    if B < 1:
        raise ValueError(f"n_simulations must be at least 1, got {B}.")

    def scalar(r):
        if isinstance(r, dict):
            for k in ("estimate", "ate", "att"):
                if k in r:
                    return float(r[k])
            raise ValueError("estimator result has no estimate/ate/att key.")
        return float(r)

    real = scalar(estimator(y, t, **kwargs))
    rng = np.random.default_rng(seed)
    draws = np.array([scalar(estimator(y, rng.permutation(t), **kwargs)) for _ in range(B)])

    p = float((1 + np.sum(np.abs(draws) >= abs(real))) / (1 + B))
    return RichResult(
        payload={
            "estimate": real,
            "placebo_mean": float(draws.mean()),
            "placebo_sd": float(draws.std(ddof=1)) if B > 1 else float("nan"),
            "p_value": p,
            "passes": bool(p < 0.05),
            "placebo": draws,
            "n_simulations": B,
            "method": "Placebo-treatment refutation (permutation randomisation test)",
        }
    )


def cheatsheet():
    return "plcbo: permute the treatment B times; p = (1 + #{|placebo| >= |real|}) / (1 + B)"
