# morie.fn -- function file (rootcoder007/morie)
"""Cross-validation risk bound."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["cvbnd"]


def cvbnd(
    cv_risks,
    n: int,
    *,
    n_folds: int = 5,
    delta: float = 0.05,
    loss_bound: float = 1.0,
) -> dict[str, Any]:
    r"""
    Hoeffding confidence bound for a cross-validated risk.

    Every observation is held out exactly once, so the cross-validated risk
    is an average of ``n`` held-out losses. If the loss takes values in
    ``[0, B]``, Hoeffding's inequality gives

    .. math::

        P\!\left(|R_{cv} - R| > t\right) \le 2\exp\!\left(-\frac{2nt^2}{B^2}\right),

    so with probability at least ``1 - delta`` the risk lies within
    ``t = B sqrt(log(2/delta) / (2 n))`` of ``R_cv``. ``B`` is the range of
    the LOSS FUNCTION (1 for misclassification or any loss in [0, 1]) and
    must be known in advance: the spread of the fold risks, which this
    function used to plug in, is a random quantity that shrinks with the
    fold size and does not bound anything.

    :param cv_risks: Per-fold risk estimates, shape (n_folds,).
    :param n: Total sample size (number of held-out losses).
    :param n_folds: Number of CV folds (reported only).
    :param delta: Confidence parameter. Default 0.05.
    :param loss_bound: Upper bound B of the loss (losses in [0, B]). Default 1.
    :return: Dict with ``cv_risk``, ``cv_se``, ``upper_bound``,
        ``lower_bound``, ``bound_width``, ``loss_bound``, ``n``, ``delta``.
    :raises ValueError: If cv_risks is empty or an argument is out of range.

    References
    ----------
    Hoeffding, W. (1963). Probability inequalities for sums of bounded
    random variables. JASA 58, 13-30.
    Kosorok, M.R. (2008). Introduction to Empirical Processes and
    Semiparametric Inference. Springer, Ch. 22.

    Examples
    --------
    >>> r = cvbnd([0.10, 0.14, 0.12, 0.08, 0.11], n=500)
    >>> round(r["bound_width"], 12)
    0.060736146191
    """
    risks = [float(v) for v in (cv_risks.tolist() if hasattr(cv_risks, "tolist") else cv_risks)]
    if len(risks) == 0:
        raise ValueError("cv_risks must be non-empty.")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1).")
    if not loss_bound > 0:
        raise ValueError("loss_bound must be positive.")
    if n < 1:
        raise ValueError("n must be at least 1.")
    k = len(risks)
    cv_risk = sum(risks) / k
    cv_se = math.sqrt(sum((v - cv_risk) ** 2 for v in risks) / (k - 1) / k) if k > 1 else float("nan")
    width = float(loss_bound) * math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    return {
        "cv_risk": cv_risk,
        "cv_se": cv_se,
        "upper_bound": cv_risk + width,
        "lower_bound": cv_risk - width,
        "bound_width": width,
        "loss_bound": float(loss_bound),
        "n": n,
        "delta": delta,
    }


def cheatsheet() -> str:
    return "cvbnd({cv_risks, n}) -> Cross-validation risk bound."
