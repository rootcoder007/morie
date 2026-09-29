"""Ensemble variance."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _is2d(a):
    rows = a.tolist() if hasattr(a, "tolist") else list(a)
    return bool(rows) and hasattr(rows[0], "__len__")


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def ensemble_variance(segments):
    r"""Pointwise ensemble variance ``s^2(n) = (1/(M - 1)) sum_k (y_k(n) - ybar(n))^2`` across ``M`` sweeps (Bessel-corrected, unbiased; needs ``M >= 2``); ``mean_var`` is its average over ``n``.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> ensemble_variance([[1.0, 2.0, 3.0], [3.0, 2.0, 5.0]]).value
    [2.0, 0.0, 2.0]
    """
    rows = (
        [_vec(r) for r in (segments.tolist() if hasattr(segments, "tolist") else segments)]
        if _is2d(segments)
        else [_vec(segments)]
    )
    M, N = len(rows), len(rows[0])
    if M < 2:
        raise ValueError("the ensemble variance needs at least two sweeps")
    var = []
    for j in range(N):
        m = math.fsum(r[j] for r in rows) / M
        var.append(math.fsum((r[j] - m) ** 2 for r in rows) / (M - 1))
    return DescriptiveResult(
        name="ensemble_variance", value=var, extra={"M": M, "N": N, "mean_var": math.fsum(var) / N}
    )


ensrv = ensemble_variance


def cheatsheet() -> str:
    return "ensemble_variance(segments) -> (1/(M-1)) sum_k (y_k(n) - ybar(n))^2."
