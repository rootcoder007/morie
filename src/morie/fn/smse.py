"""Mean squared error."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def mean_squared_error(x, x_hat):
    r"""Mean squared error ``(1/N) sum_n (x(n) - xhat(n))^2`` between a reference and an estimate of equal length.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> mean_squared_error([1.0, 2.0, 3.0], [1.5, 2.0, 2.0]).value
    0.4166666666666667
    """
    a, b = _vec(x), _vec(x_hat)
    if len(a) != len(b) or not a:
        raise ValueError("x and x_hat must be non-empty and of equal length")
    mse = math.fsum((u - w) ** 2 for u, w in zip(a, b)) / len(a)
    return DescriptiveResult(name="mean_squared_error", value=mse, extra={"mse": mse, "n": len(a)})


smse = mean_squared_error


def cheatsheet() -> str:
    return "mean_squared_error(x, x_hat) -> (1/N) sum (x - xhat)^2."
