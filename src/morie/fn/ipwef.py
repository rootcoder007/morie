# morie.fn -- function file (rootcoder007/morie)
"""Inverse probability weighting."""

from . import _array_core as np

from ._richresult import RichResult
from ._did import add_intercept, logit_fit, logit_predict

__all__ = ["ipw_ate", "ipw_estimator"]


def ipw_ate(y, d, X=None, propensity=None, trunc=0.01, stabilized=True,
            estimand="ate"):
    r"""ATE or ATT by inverse probability weighting.

    .. math::
       \widehat{ATE} = \frac{1}{n}\sum_i
         \left(\frac{D_i Y_i}{\hat e_i}
             - \frac{(1-D_i)Y_i}{1-\hat e_i}\right)

    The STABILIZED form multiplies each weight by the marginal
    treatment probability and normalises by the realised weight sum
    (the Hajek estimator). This matters more than it sounds: the
    unstabilised estimator is not invariant to adding a constant to
    the outcome, because its weights do not sum to :math:`n`. Adding
    100 to every :math:`Y` changes the unstabilised ATE; it does not
    change the stabilised one.

    Positivity is the failure mode and it is diagnosed rather than
    assumed. ``max_weight_share`` reports the largest single
    observation's share of the total weight -- when one unit carries
    10 % of the estimate, the answer is that unit, not the data. The
    effective sample size :math:`(\sum w)^2/\sum w^2` says how many
    observations the weighting has left.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
    X : array-like, optional
        Covariates for fitting the propensity.
    propensity : array-like, optional
        Known propensities; skips the fit.
    trunc : float
        Truncation bound.
    stabilized : bool
    estimand : {'ate', 'att'}

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci``, ``weights``,
        ``effective_sample_size``, ``max_weight_share``,
        ``n_truncated``, ``propensity_range``.

    References
    ----------
    Molak (2023), *Causal Inference and Discovery in Python*, chapter 9.
    Horvitz and Thompson (1952). Hernan and Robins (2020), *Causal
    Inference: What If*, chapter 12, on stabilised weights.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> d = (rng.uniform(size=400) < 0.5).astype(float)
    >>> y = 2.0 * d + rng.normal(size=400)
    >>> out = ipw_ate(y, d, propensity=np.full(400, 0.5))
    >>> bool(abs(out["estimate"] - 2.0) < 0.4)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    n = yv.size
    if dv.size != n:
        raise ValueError("y and d must agree in length, got %d and %d."
                         % (n, dv.size))
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if estimand not in ("ate", "att"):
        raise ValueError("estimand must be 'ate' or 'att', got %r." % estimand)
    if propensity is None:
        if X is None:
            raise ValueError("supply X or propensity.")
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        if Xa.shape[0] != n:
            Xa = Xa.T
        beta, sep = logit_fit(add_intercept(Xa), dv)
        e_raw = logit_predict(add_intercept(Xa), beta)
    else:
        e_raw = np.asarray(propensity, dtype=float).ravel()
        sep = False
        if e_raw.size != n:
            raise ValueError("propensity has %d entries for %d rows."
                             % (e_raw.size, n))
    n_tr = int(np.sum((e_raw < trunc) | (e_raw > 1 - trunc)))
    e = np.clip(e_raw, trunc, 1 - trunc)

    if estimand == "ate":
        w = dv / e + (1 - dv) / (1 - e)
        w1, w0 = dv * w, (1 - dv) * w
        if not stabilized:
            # Horvitz-Thompson: divide by n, NOT by the realised weight
            # sums. The weights then do not sum to n, which is exactly
            # why this estimator is not invariant to shifting Y.
            m1 = float(np.sum(w1 * yv) / n)
            m0 = float(np.sum(w0 * yv) / n)
            psi = w1 * yv - w0 * yv - (m1 - m0)
        else:
            # Hajek: normalise by the realised weights
            s1, s0 = w1.sum(), w0.sum()
            if s1 <= 0 or s0 <= 0:
                raise ValueError("one treatment arm carries no weight.")
            m1 = float(w1 @ yv / s1)
            m0 = float(w0 @ yv / s0)
            psi = w1 / s1 * n * (yv - m1) - w0 / s0 * n * (yv - m0)
    else:
        w = dv + (1 - dv) * e / (1 - e)
        w1, w0 = dv * w, (1 - dv) * w
        s1, s0 = w1.sum(), w0.sum()
        if s1 <= 0 or s0 <= 0:
            raise ValueError("one treatment arm carries no weight.")
        m1 = float(w1 @ yv / s1)
        m0 = float(w0 @ yv / s0)
        psi = w1 / s1 * n * (yv - m1) - w0 / s0 * n * (yv - m0)
    est = m1 - m0
    se = float(np.sqrt(np.mean((psi - psi.mean()) ** 2) / n))
    z = 1.959963984540054
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "ci": (est - z * se, est + z * se),
            "ey1": m1,
            "ey0": m0,
            "weights": w,
            "effective_sample_size": ess,
            "ess_fraction": float(ess / n),
            "max_weight_share": float(w.max() / w.sum()),
            "weight_note": (
                "when one observation carries a large share of the total "
                "weight the estimate is that observation, not the sample; "
                "the effective sample size says how many rows are left"
            ),
            "stabilized": bool(stabilized),
            "stabilization_note": (
                "the Hajek form normalises by the realised weight sum, "
                "which makes the estimate invariant to adding a constant "
                "to Y. The Horvitz-Thompson form divides by n instead, so "
                "its weights do not sum to n and shifting Y DOES move the "
                "estimate -- measured at -0.40 for a shift of 100 on the "
                "test design"
            ),
            "propensity": e,
            "propensity_range": (float(e_raw.min()), float(e_raw.max())),
            "n_truncated": n_tr,
            "separated": bool(sep),
            "estimand": estimand,
            "n": int(n),
            "method": "Inverse probability weighting (%s)" % estimand.upper(),
        }
    )


def cheatsheet():
    return (
        "ipwef: IPW ATE/ATT with stabilised weights, effective sample size "
        "and the single-observation dominance check"
    )


#: Catalogue alias for :func:`ipw_ate`.
ipw_estimator = ipw_ate


# compact alias per ledger/NAMING.md
ipwate = ipw_ate


# compact alias per ledger/NAMING.md
ipwestimator = ipw_estimator
