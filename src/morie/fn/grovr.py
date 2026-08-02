# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-vs-Rest: K binary classifiers, predict the argmax of their scores."""

from . import _array_core as np

from ._richresult import RichResult
from .grsig import geron_sigmoid

__all__ = ["geron_one_vs_rest", "train_logreg"]

_METHOD = "One-vs-Rest multiclass reduction"


def train_logreg(X, y, eta=0.5, n_iter=400, l2=0.0):
    """Batch-gradient-descent logistic regression, intercept prepended.

    Deterministic (no sampling), so two calls on the same data give the
    same weights. Shared with :mod:`morie.fn.grovo` as the default binary
    base learner.
    """
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    w = np.zeros(A.shape[1])
    for _ in range(int(n_iter)):
        p = np.asarray(geron_sigmoid(A @ w)["sigma"], dtype=float).ravel()
        grad = A.T @ (p - y) / A.shape[0]
        if l2:
            grad[1:] += l2 * w[1:]
        w -= eta * grad
    return w


def _scores(w, X):
    return np.hstack([np.ones((X.shape[0], 1)), X]) @ w


def geron_one_vs_rest(X, y, base_fit=None, eta=0.5, n_iter=400):
    r"""Train one "class k against everything else" classifier per class.

    .. math::
        \hat y = \arg\max_k \mathrm{score}_k(x)
        \quad \text{over the } K \text{ OvR classifiers}

    K classifiers, each on the full dataset.  Compared with One-vs-One
    (:mod:`morie.fn.grovo`) that is far fewer models but each sees every
    instance, so OvR is the right choice when training cost grows
    linearly in ``m`` and the wrong one when it grows worse than that.
    Its structural weakness is here too: each binary problem is
    imbalanced (1 class against K-1), and the K scores are calibrated
    independently, so comparing them by argmax is a convention, not a
    probability statement.

    ``base_fit`` is caller-supplied and its contract is enforced: it must
    accept ``(X, y_binary)`` and return a callable producing one score
    per row.  The default is the deterministic logistic regression in
    :func:`train_logreg`.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like of int, shape (m,)
        Labels ``0 .. K-1``; at least two classes.
    base_fit : callable, optional
    eta, n_iter : float, int, optional
        Default learner's step size and iteration count.

    Returns
    -------
    RichResult
        Payload keys ``predictions``, ``scores`` (m x K),
        ``n_classifiers``, ``accuracy``, ``coefficients``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 3, Multiclass (OvR) section.

    Examples
    --------
    Three well-separated 1-D clusters: three classifiers, all training
    points recovered.

    >>> X = [[0.0], [0.5], [5.0], [5.5], [10.0], [10.5]]
    >>> y = [0, 0, 1, 1, 2, 2]
    >>> r = geron_one_vs_rest(X, y)
    >>> r["n_classifiers"]
    3
    >>> r["predictions"]
    [0, 0, 1, 1, 2, 2]
    >>> r["accuracy"]
    1.0
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    yv = np.asarray(y).ravel()
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"X must be a non-empty (m, n) matrix, got shape {A.shape}.")
    if yv.size != A.shape[0]:
        raise ValueError(f"y has {yv.size} labels but X has {A.shape[0]} rows.")
    if not np.all(yv == np.round(np.asarray(yv, dtype=float))):
        raise ValueError("y must hold integer class labels.")
    yv = yv.astype(int)
    classes = np.unique(yv)
    if classes.size < 2:
        raise ValueError(f"OvR needs at least 2 classes, got {classes.size}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")

    S = np.zeros((A.shape[0], classes.size))
    coefs = []
    for j, c in enumerate(classes):
        yb = (yv == c).astype(float)
        if base_fit is None:
            w = train_logreg(A, yb, eta=eta, n_iter=n_iter)
            S[:, j] = _scores(w, A)
            coefs.append(w.tolist())
        else:
            if not callable(base_fit):
                raise ValueError(f"base_fit must be callable, got {type(base_fit).__name__}.")
            model = base_fit(A, yb)
            if not callable(model):
                raise ValueError("base_fit(X, y) must return a callable scorer.")
            s = np.asarray(model(A), dtype=float).ravel()
            if s.size != A.shape[0]:
                raise ValueError(
                    f"base classifier for class {int(c)} returned {s.size} scores "
                    f"for {A.shape[0]} rows."
                )
            if not np.all(np.isfinite(s)):
                raise ValueError(f"base classifier for class {int(c)} returned non-finite scores.")
            S[:, j] = s
            coefs.append(None)

    pred = classes[np.argmax(S, axis=1)]
    acc = float(np.mean(pred == yv))

    return RichResult(
        title="One-vs-Rest",
        summary_lines=[("Classifiers", int(classes.size)), ("Training accuracy", acc)],
        payload={
            "predictions": pred.astype(int).tolist(),
            "scores": S.tolist(),
            "classes": classes.astype(int).tolist(),
            "n_classifiers": int(classes.size),
            "accuracy": acc,
            "coefficients": coefs,
            "estimate": pred.astype(int).tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grovr: K classifiers, each class vs the rest, argmax of scores; K models, each on all m rows"
