"""Flexible discriminant analysis by optimal scoring with a linear regression step (ESL sec 12.5)."""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["esl_fda"]


def esl_fda(X, g, query=None, basis=None):
    r"""Optimal scoring (ESL Alg. 12.5.1) with least squares on the basis ``h(x)``.

    1. Regress the N x K class-indicator matrix Y on (1, h(x)), giving
       :math:`\hat Y = S Y` and fitted functions :math:`\eta^*(x)`.
    2. Optimal scores: the eigen-decomposition of :math:`Y^T\hat Y/N` relative to
       :math:`D_\pi = Y^TY/N`, eigenvectors normalised :math:`\Theta^TD_\pi\Theta = I`;
       the leading constant score is dropped.
    3. Discriminant functions :math:`\eta(x) = \Theta^T\eta^*(x)`; classify to
       the closest class centroid with weights :math:`1/(\alpha_\ell^2(1-\alpha_\ell^2))`,
       :math:`\alpha_\ell^2` the eigenvalues (as ``mda::fda``).

    With the linear basis (the default, h(x) = x) this is linear discriminant
    analysis; pass ``basis`` (a callable returning the expanded row) for
    polynomial or spline FDA.

    Parameters
    ----------
    X : N x p nested sequence
    g : sequence of N labels
    query : M x p nested sequence, optional
    basis : callable, optional
        ``basis(row) -> list`` expansion applied to X and query.

    Returns
    -------
    RichResult
        ``eigenvalues`` (the K - 1 non-trivial alpha^2), ``scores`` (Theta,
        K x (K - 1)), ``variates`` (discriminant functions at query),
        ``centroids``, ``prediction``, ``classes``.

    References
    ----------
    Hastie, T., Tibshirani, R. & Buja, A. (1994). Flexible discriminant
    analysis by optimal scoring. JASA 89, 1255-1270.
    """
    h = (lambda r: [float(v) for v in r]) if basis is None else basis
    rows = [h(r) for r in X]
    labels = list(g)
    classes = sorted(set(labels), key=repr)
    N, K = len(rows), len(classes)
    if len(labels) != N or K < 2:
        raise ValueError("need X and g of equal length and at least two classes")
    H = np.asarray([[1.0] + r for r in rows])
    Y = np.asarray([[1.0 if lab == c else 0.0 for c in classes] for lab in labels])
    B = np.linalg.lstsq(H, Y, rcond=None)[0]
    Yhat = H @ B
    pi = [float(v) / N for v in Y.sum(axis=0)]
    Dm = np.diag(np.asarray([1 / math.sqrt(v) for v in pi]))
    Msym = Dm @ (Y.T @ Yhat / N) @ Dm
    Msym = (Msym + Msym.T) / 2
    w, V = np.linalg.eigh(Msym)
    order = sorted(range(K), key=lambda k: -float(w[k]))
    Theta = Dm @ V[:, order]
    vals = [float(w[k]) for k in order][1:]
    Th = Theta[:, 1:]
    fit_tr = (Yhat @ Th).tolist()
    cent = [
        [
            sum(fit_tr[i][q] for i in range(N) if labels[i] == c) / sum(1 for lab in labels if lab == c)
            for q in range(K - 1)
        ]
        for c in classes
    ]
    wts = [1 / (a * (1 - a)) if 0 < a < 1 else 0.0 for a in vals]
    Qrows = rows if query is None else [h(r) for r in query]
    etas = (np.asarray([[1.0] + r for r in Qrows]) @ B @ Th).tolist()
    pred = [
        classes[min(range(K), key=lambda k: (sum(wl * (e[q] - cent[k][q]) ** 2 for q, wl in enumerate(wts)), k))]
        for e in etas
    ]
    return RichResult(
        title="Flexible discriminant analysis",
        summary_lines=[("eigenvalues", vals)],
        payload={
            "eigenvalues": vals,
            "scores": Th.tolist(),
            "variates": etas,
            "centroids": cent,
            "prediction": pred,
            "classes": classes,
        },
    )


def cheatsheet():
    return (
        "eslfda: regress Y on h(x); eigen of Y'Yhat/N vs D_pi (Theta' D Theta = I); eta = Theta' eta*; nearest centroid"
    )
