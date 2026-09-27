"""Classification by linear regression of an indicator response matrix (ESL sec 4.2)."""

from ._richresult import RichResult
from .eslmol import esl_multi_output_ls

__all__ = ["esl_indicator_regression"]


def esl_indicator_regression(X, g, query=None):
    r"""Fit :math:`\hat Y = X(X^TX)^{-1}X^TY` to the class indicators and classify by the largest fit.

    Each class k gets an indicator column :math:`Y_k`; the fitted vector of a
    point is :math:`\hat f(x)^T = (1, x^T)\hat B` and the rule is
    :math:`\hat G(x) = \arg\max_k \hat f_k(x)` (ESL eqs 4.3-4.6), which is the
    same as the closest target :math:`t_k` in eq 4.5. The fits sum to one
    for every x but need not lie in [0, 1]; with K >= 3 classes can be
    masked.

    Parameters
    ----------
    X : N x p nested sequence
    g : sequence of N class labels
    query : M x p nested sequence, optional
        Points to classify (default the training X).

    Returns
    -------
    RichResult
        ``prediction``, ``fitted`` (rows of query), ``coefficients``
        ((p + 1) x K), ``classes``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 4.2.
    """
    labels = list(g)
    classes = sorted(set(labels), key=repr)
    if len(classes) < 2:
        raise ValueError("need at least two classes")
    Y = [[1.0 if lab == c else 0.0 for c in classes] for lab in labels]
    B = esl_multi_output_ls(X, Y)["coefficients"]
    Q = X if query is None else query
    fit = [[B[0][k] + sum(float(v) * B[j + 1][k] for j, v in enumerate(r)) for k in range(len(classes))] for r in Q]
    pred = [classes[max(range(len(classes)), key=lambda k: (f[k], -k))] for f in fit]
    return RichResult(
        title="Indicator-matrix regression",
        summary_lines=[("classes", classes)],
        payload={"prediction": pred, "fitted": fit, "coefficients": B, "classes": classes},
    )


def cheatsheet():
    return "eslind: regress class indicators on (1, x); classify to argmax_k of the fitted row"
