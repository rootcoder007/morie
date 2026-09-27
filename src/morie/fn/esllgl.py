"""Local multinomial logistic regression (ESL sec 6.5)."""

import math

from . import _array_core as np
from ._richresult import RichResult
from .eslmnl import esl_multinomial_logit
from .eslnnk import _KERNELS, _kernel_weights

__all__ = ["esl_local_logistic"]


def esl_local_logistic(X, g, x0, lambda_, kernel="gaussian"):
    r"""Class probabilities at :math:`x_0` from a kernel-weighted local linear logit.

    ESL eqs 6.18-6.20: maximise
    :math:`\sum_i K_\lambda(x_0, x_i)\{\beta_{g_i0}(x_0) + \beta_{g_i}(x_0)^T(x_i - x_0)
    - \log[1 + \sum_k \exp(\beta_{k0}(x_0) + \beta_k(x_0)^T(x_i - x_0))]\}`
    with the last class as baseline; because the design is centred at
    :math:`x_0`, :math:`\hat\Pr(G=j|x_0)` is the softmax of the intercepts
    :math:`\hat\beta_{j0}(x_0)`. Kernel weights use :math:`\|x_i - x_0\|/\lambda`.

    Parameters
    ----------
    X : n x p nested sequence
    g : sequence of n labels
    x0 : m x p nested sequence
        Target points.
    lambda_ : float
        Bandwidth, > 0.
    kernel : {"gaussian", "epanechnikov", "tri-cube"}

    Returns
    -------
    RichResult
        ``prob`` (one row per target, classes in sorted order),
        ``intercepts``, ``classes``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 6.5.
    """
    rows = [[float(v) for v in r] for r in X]
    labels = list(g)
    if len(labels) != len(rows) or lambda_ <= 0 or kernel not in _KERNELS:
        raise ValueError("need X and g of equal length, lambda > 0 and a known kernel")
    probs, inter, classes = [], [], None
    for t in x0:
        t = [float(v) for v in t]
        d = [[a - b for a, b in zip(r, t)] for r in rows]
        w = [
            float(v)
            for v in _kernel_weights(np.asarray([math.sqrt(sum(v * v for v in r)) / lambda_ for r in d]), kernel)
        ]
        fit = esl_multinomial_logit(d, labels, query=[[0.0] * len(t)], weights=w)
        classes = fit["classes"]
        probs.append(fit["prob"][0])
        inter.append([c[0] for c in fit["coefficients"]])
    return RichResult(
        title="Local logistic regression",
        summary_lines=[("classes", classes)],
        payload={"prob": probs, "intercepts": inter, "classes": classes},
    )


def cheatsheet():
    return "esllgl: kernel-weighted multinomial logit on (x - x0); Pr(G=j|x0) = softmax of the intercepts"
