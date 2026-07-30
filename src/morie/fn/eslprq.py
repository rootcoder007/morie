# morie.fn -- function file (rootcoder007/morie)
"""Learning vector quantization -- Kohonen (1989), ESL Sec 13.2.2."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_prototype_lvq"]


def esl_prototype_lvq(X, y, n_prototypes=2, eta=0.1, n_epochs=50, newdata=None, seed=0):
    r"""Fit class prototypes by LVQ1 and classify by nearest prototype.

    Prototypes are initialised within each class, then for each training
    point the nearest prototype is pulled *toward* it when their labels agree
    and pushed *away* when they do not:

    .. math::
        m^* \leftarrow m^* \pm \eta\,(x_i - m^*),

    with the sign positive on a match. The repulsion is the whole difference
    from k-means-per-class. Its effect is to drive prototypes *out of* the
    contested region: where the classes overlap, a prototype is repeatedly
    pushed away from the opposing class's points, so the fitted prototypes
    end up further apart than the class centroids are. They are positioned to
    win the nearest-prototype vote, not to summarise their class.

    ESL flags the cost: LVQ is defined by an algorithm rather than by an
    objective function, so there is no quantity guaranteed to decrease and
    convergence is not assured. The learning rate is therefore decayed to
    zero, which is what forces the iteration to settle.

    Parameters
    ----------
    X : array-like
        Predictors ``(n, p)``.
    y : array-like
        Class labels; any hashable values.
    n_prototypes : int
        Prototypes *per class*.
    eta : float
        Initial learning rate, in (0, 1].
    n_epochs : int
        Passes over the data.
    newdata : array-like, optional
        Points to classify. Defaults to ``X``.
    seed : int
        Seed for initialisation and presentation order.

    Returns
    -------
    RichResult
        ``prototypes``, ``prototype_class``, ``class_``, ``accuracy``,
        ``classes``.

    References
    ----------
    Kohonen, T. (1989). *Self-Organization and Associative Memory* (3rd ed.).
        Springer.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = np.r_[rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))]
    >>> y = np.r_[np.zeros(100), np.ones(100)]
    >>> r = esl_prototype_lvq(X, y, n_prototypes=2, seed=1)
    >>> bool(r["accuracy"] > 0.95)
    True

    Where the classes overlap the repulsion actually fires, driving the
    prototypes out of the contested region and further apart than the class
    centroids -- prototypes are placed to win the vote, not to summarise.

    >>> rng2 = np.random.default_rng(3)
    >>> Xo = np.r_[rng2.normal(-0.7, 1, (150, 2)), rng2.normal(0.7, 1, (150, 2))]
    >>> yo = np.r_[np.zeros(150), np.ones(150)]
    >>> ro = esl_prototype_lvq(Xo, yo, n_prototypes=1, eta=0.1, seed=1)
    >>> P, pc = ro["prototypes"], ro["prototype_class"]
    >>> sep = np.linalg.norm(P[pc == 0].mean(0) - P[pc == 1].mean(0))
    >>> cent = np.linalg.norm(Xo[yo == 0].mean(0) - Xo[yo == 1].mean(0))
    >>> bool(sep > cent)
    True

    >>> esl_prototype_lvq(X, y, n_prototypes=0)
    Traceback (most recent call last):
        ...
    ValueError: n_prototypes must be at least 1
    """
    if n_prototypes < 1:
        raise ValueError("n_prototypes must be at least 1")
    if not 0.0 < eta <= 1.0:
        raise ValueError("eta must be in (0, 1]")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    yr = np.asarray(y).ravel()
    n = X.shape[0]
    if yr.size != n:
        raise ValueError(f"X has {n} rows but y has {yr.size}")
    classes = np.unique(yr)
    rng = np.random.default_rng(seed)

    protos, pclass = [], []
    for c in classes:
        idx = np.flatnonzero(yr == c)
        if idx.size < n_prototypes:
            raise ValueError(
                f"class {c!r} has {idx.size} observations, fewer than "
                f"n_prototypes={n_prototypes}"
            )
        protos.append(X[rng.choice(idx, n_prototypes, replace=False)])
        pclass.append(np.full(n_prototypes, c))
    M = np.vstack(protos).astype(float)
    mc = np.concatenate(pclass)

    for ep in range(n_epochs):
        lr = eta * (1.0 - ep / n_epochs)
        for i in rng.permutation(n):
            j = int(np.argmin(((M - X[i]) ** 2).sum(1)))
            sign = 1.0 if mc[j] == yr[i] else -1.0
            M[j] += sign * lr * (X[i] - M[j])

    Z = X if newdata is None else np.atleast_2d(np.asarray(newdata, dtype=float))
    if Z.shape[1] != X.shape[1]:
        raise ValueError(f"newdata has {Z.shape[1]} columns but X has {X.shape[1]}")
    pred = mc[np.argmin(((Z[:, None] - M[None]) ** 2).sum(-1), axis=1)]
    train = mc[np.argmin(((X[:, None] - M[None]) ** 2).sum(-1), axis=1)]
    return RichResult(
        title="Learning vector quantization",
        summary_lines=[("n", n), ("prototypes/class", int(n_prototypes)),
                       ("accuracy", float(np.mean(train == yr)))],
        payload={
            "prototypes": M, "prototype_class": mc,
            "class_": pred, "accuracy": float(np.mean(train == yr)),
            "classes": classes, "n_prototypes": int(n_prototypes),
            "method": "esl_prototype_lvq",
        },
    )


def cheatsheet():
    return "eslprq: LVQ1 pushes wrong-class prototypes AWAY, pulling them onto the boundary; no objective function"
