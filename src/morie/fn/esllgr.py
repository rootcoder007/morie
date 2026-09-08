# morie.fn -- function file (rootcoder007/morie)
"""Logistic regression -- ESL Sec 4.4."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from .esliwls import esl_iwls

__all__ = ["esl_logistic_reg"]


def esl_logistic_reg(X, y, newdata=None, threshold=0.5, **kwargs):
    r"""Fit a logistic regression and classify.

    .. math::
        \log\frac{P(Y=1\mid X)}{1 - P(Y=1\mid X)} = \beta_0 + X^\top\beta .

    Fitting is IRLS via :func:`~morie.fn.esliwls.esl_iwls`; this adds the
    prediction path -- fitted probabilities, a class rule at ``threshold``,
    and the confusion counts on the training data.

    The default 0.5 threshold minimises error rate only under equal
    misclassification costs and roughly balanced classes. On imbalanced data
    it will happily predict the majority class everywhere, so treat it as a
    starting point rather than a property of the model.

    Parameters
    ----------
    X : array-like
        Design matrix ``(n, p)``.
    y : array-like
        Binary 0/1 response.
    newdata : array-like, optional
        Points to predict at. Defaults to ``X``.
    threshold : float
        Probability cut-off for the class rule, in (0, 1).
    **kwargs
        Passed to :func:`~morie.fn.esliwls.esl_iwls`.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``p_value``, ``prob``, ``class_``, ``odds_ratio``,
        ``loglik``, ``deviance``, plus training ``accuracy`` and
        ``confusion``.

    References
    ----------
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> X = rng.normal(size=(3000, 1))
    >>> y = (rng.random(3000) < 1 / (1 + np.exp(-(0.3 + 2.0 * X[:, 0])))).astype(float)
    >>> r = esl_logistic_reg(X, y)
    >>> bool(abs(r["beta"][1] - 2.0) < 0.2)
    True

    The odds ratio is the exponentiated coefficient, which is how a logistic
    fit is normally reported.

    >>> bool(abs(r["odds_ratio"][1] - np.exp(r["beta"][1])) < 1e-12)
    True

    Probabilities stay in the unit interval and the classifier beats chance.

    >>> bool(r["prob"].min() >= 0 and r["prob"].max() <= 1)
    True
    >>> bool(r["accuracy"] > 0.6)
    True

    >>> esl_logistic_reg([[1.0], [2.0]], [0.0, 1.0], threshold=1.5)
    Traceback (most recent call last):
        ...
    ValueError: threshold must be in (0, 1)
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    fit = esl_iwls(X, y, family="binomial", **kwargs)
    beta = fit["beta"]
    Z = np.atleast_2d(np.asarray(X if newdata is None else newdata, dtype=float))
    if kwargs.get("add_intercept", True):
        Z = np.column_stack([np.ones(Z.shape[0]), Z])
    if Z.shape[1] != beta.size:
        raise ValueError(f"newdata gives {Z.shape[1]} columns but beta has {beta.size}")
    prob = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -500, 500)))
    cls = (prob >= threshold).astype(int)
    yv = np.asarray(y, dtype=float).ravel()
    if newdata is None:
        acc = float(np.mean(cls == yv))
        conf = np.array([[int(np.sum((yv == a) & (cls == b))) for b in (0, 1)] for a in (0, 1)])
    else:
        acc, conf = float("nan"), None
    return RichResult(
        title="Logistic regression",
        summary_lines=[("n", int(yv.size)), ("loglik", fit["loglik"]), ("accuracy", acc)],
        warnings=list(fit.get("warnings", []) or []),
        payload={
            "beta": beta, "se": fit["se"], "z": fit["z"], "p_value": fit["p_value"],
            "odds_ratio": np.exp(beta),
            "prob": prob, "class_": cls, "threshold": float(threshold),
            "accuracy": acc, "confusion": conf,
            "loglik": fit["loglik"], "deviance": fit["deviance"],
            "converged": fit["converged"], "separated": fit["separated"],
            "method": "esl_logistic_reg",
        },
    )


def cheatsheet():
    return "esllgr: logistic regression via IRLS; odds_ratio = exp(beta), and 0.5 is a cost assumption not a default truth"


# compact alias per ledger/NAMING.md
esllogisticreg = esl_logistic_reg
