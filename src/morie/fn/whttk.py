"""Whittaker smoother (penalised least squares)."""

from __future__ import annotations

from . import _array_core as np

from ._containers import DescriptiveResult


def whittaker_smooth(
    y: np.ndarray,
    lambda_: float = 100.0,
    d: int = 2,
) -> DescriptiveResult:
    r"""
    Whittaker smoother (penalised least squares).

    Minimises :math:`\\|y - z\\|^2 + \\lambda \\|D^d z\\|^2`
    where *D* is the *d*-th order difference matrix.

    :param y: 1-D signal array.
    :param lambda_: Smoothing penalty (larger = smoother). Default 100.
    :param d: Order of the difference penalty. Default 2.
    :return: DescriptiveResult with smoothed array in extra.
    :raises ValueError: If y is too short or lambda_ <= 0.

    References
    ----------
    Eilers, P. H. C. (2003). A perfect smoother. Analytical Chemistry,
    75(14), 3631--3636. doi:10.1021/ac034173t
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < d + 1:
        raise ValueError(f"y must be 1-D with length >= {d + 1}.")
    if lambda_ <= 0.0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")

    n = len(y)
    # d-th order difference matrix rows, built as dense lists
    D = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(d):
        D = [[D[i + 1][j] - D[i][j] for j in range(n)]
             for i in range(len(D) - 1)]
    W = [[(1.0 if i == j else 0.0)
          + lambda_ * sum(D[r][i] * D[r][j] for r in range(len(D)))
          for j in range(n)] for i in range(n)]
    z = np.linalg.solve(np.asarray(W), y)

    residual = y - z
    rmse = float(np.sqrt(np.mean(residual**2)))

    return DescriptiveResult(
        name="Whittaker Smoother",
        value=rmse,
        extra={
            "smoothed": z,
            "residuals": residual,
            "rmse": rmse,
            "lambda": lambda_,
            "difference_order": d,
            "n": n,
        },
    )


whttk = whittaker_smooth


def cheatsheet() -> str:
    return "whittaker_smooth({}) -> Whittaker smoother."
