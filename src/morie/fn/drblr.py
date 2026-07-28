# morie.fn -- function file (rootcoder007/morie)
"""Doubly robust (AIPW) ATE estimator."""

import numpy as np

from ._richresult import RichResult
from ._did import add_intercept, logit_fit, logit_predict, ols_fit

__all__ = ["doubly_robust_ate"]


def doubly_robust_ate(y, d, X, propensity=None, mu1=None, mu0=None,
                      trunc=0.01):
    r"""Augmented IPW: consistent if EITHER nuisance model is right.

    .. math::
       \hat\tau = \frac1n\sum_i\Big[
         \hat\mu_1(X_i) - \hat\mu_0(X_i)
         + \frac{D_i\{Y_i-\hat\mu_1(X_i)\}}{\hat e_i}
         - \frac{(1-D_i)\{Y_i-\hat\mu_0(X_i)\}}{1-\hat e_i}\Big]

    Double robustness is often stated as "two chances to be right",
    which oversells it. What the property actually guarantees is
    consistency when one model is correct; it says nothing about the
    STANDARD ERROR, which is valid only if the model that happens to be
    correct is also well-behaved. And when BOTH models are misspecified
    -- the usual case -- the AIPW estimator can be worse than either
    the outcome-regression or the IPW estimator alone, because the
    augmentation term amplifies a bad propensity rather than repairing
    it.

    Both single-model estimates are returned for exactly that reason:
    when ``regression_only`` and ``ipw_only`` disagree sharply, at
    least one model is wrong, and the doubly robust number sitting
    between them is not evidence that it has fixed anything.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
    X : array-like, shape (n, p)
    propensity, mu1, mu0 : array-like, optional
        Supply fitted nuisances to skip the internal linear fits.
    trunc : float

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci``, ``regression_only``,
        ``ipw_only``, ``model_disagreement``, ``eif``,
        ``propensity_range``, ``n_truncated``.

    References
    ----------
    Molak (2023), *Causal Inference and Discovery in Python*, chapter 10.
    Robins, Rotnitzky and Zhao (1994), *JASA* 89:846-866.
    Kang and Schafer (2007), *Statistical Science* 22:523-539, on the
    both-wrong case.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 2))
    >>> d = (rng.uniform(size=600) < 0.5).astype(float)
    >>> y = 2.0 * d + X[:, 0] + rng.normal(size=600)
    >>> bool(abs(doubly_robust_ate(y, d, X)["estimate"] - 2.0) < 0.3)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    if dv.size != n or Xa.shape[0] != n:
        raise ValueError("y, d and X must agree in their first dimension.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if min(int(dv.sum()), int((1 - dv).sum())) < 2:
        raise ValueError("need at least 2 observations in each arm.")

    B = add_intercept(Xa)
    if propensity is None:
        beta, sep = logit_fit(B, dv)
        e_raw = logit_predict(B, beta)
    else:
        e_raw = np.asarray(propensity, dtype=float).ravel()
        sep = False
        if e_raw.size != n:
            raise ValueError("propensity has %d entries for %d rows."
                             % (e_raw.size, n))
    n_tr = int(np.sum((e_raw < trunc) | (e_raw > 1 - trunc)))
    e = np.clip(e_raw, trunc, 1 - trunc)

    if mu1 is None:
        mu1 = B @ ols_fit(B[dv == 1], yv[dv == 1])
    else:
        mu1 = np.asarray(mu1, dtype=float).ravel()
    if mu0 is None:
        mu0 = B @ ols_fit(B[dv == 0], yv[dv == 0])
    else:
        mu0 = np.asarray(mu0, dtype=float).ravel()

    aug = (dv * (yv - mu1) / e - (1 - dv) * (yv - mu0) / (1 - e))
    psi = mu1 - mu0 + aug
    est = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    reg = float(np.mean(mu1 - mu0))
    ipw = float(np.mean(dv * yv / e) - np.mean((1 - dv) * yv / (1 - e)))
    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "ci": (est - z * se, est + z * se),
            "eif": psi,
            "eif_mean_centered": float(np.mean(psi) - est),
            "regression_only": reg,
            "ipw_only": ipw,
            "model_disagreement": float(abs(reg - ipw)),
            "disagreement_note": (
                "when the outcome-regression and IPW estimates disagree "
                "sharply at least one model is wrong; a doubly robust number "
                "sitting between them is not evidence it repaired either"
            ),
            "double_robust_note": (
                "consistency holds if EITHER nuisance model is correct, but "
                "the standard error is valid only if the correct one is also "
                "well behaved; with both wrong the augmentation can make "
                "this worse than either single-model estimate"
            ),
            "augmentation_mean": float(np.mean(aug)),
            "propensity": e,
            "propensity_range": (float(e_raw.min()), float(e_raw.max())),
            "n_truncated": n_tr,
            "separated": bool(sep),
            "n": int(n),
            "method": "Doubly robust (AIPW) ATE",
        }
    )


def cheatsheet():
    return (
        "drblr: AIPW ATE with both single-model estimates exposed, since "
        "their disagreement is the real diagnostic"
    )
