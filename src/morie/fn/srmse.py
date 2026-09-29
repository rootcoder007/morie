"""Root mean squared error."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def root_mean_squared_error(x, x_hat):
    r"""Root mean squared error ``sqrt((1/N) sum_n (x(n) - xhat(n))^2)``.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> round(root_mean_squared_error([1.0, 2.0, 3.0], [1.5, 2.0, 2.0]).value, 12)
    0.645497224368
    """
    a, b = _vec(x), _vec(x_hat)
    if len(a) != len(b) or not a:
        raise ValueError("x and x_hat must be non-empty and of equal length")
    mse = math.fsum((u - w) ** 2 for u, w in zip(a, b)) / len(a)
    rmse = math.sqrt(mse)
    return DescriptiveResult(name="root_mean_squared_error", value=rmse, extra={"rmse": rmse, "mse": mse, "n": len(a)})


srmse = root_mean_squared_error


def cheatsheet() -> str:
    return "root_mean_squared_error(x, x_hat) -> sqrt(MSE)."
