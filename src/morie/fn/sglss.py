"""Squared error loss function."""

from __future__ import annotations

from . import _array_core as np
from ._containers import SpatialResult


def squared_error_loss(
    predicted: np.ndarray,
    observed: np.ndarray,
) -> SpatialResult:
    r"""Compute element-wise squared error loss.

    .. math::

        L_i = (Z(s_i) - \hat{Z}(s_i))^2

    Parameters
    ----------
    predicted : np.ndarray
        Predicted values.
    observed : np.ndarray
        Observed values.

    Returns
    -------
    SpatialResult
        ``statistic`` is mean squared error.
        ``extra`` has ``losses``, ``total_loss``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::


    Examples
    --------
    >>> r = squared_error_loss([1.0, 2.0, 4.0], [1.5, 2.0, 3.0])
    >>> r.statistic, r.extra["total_loss"]
    (0.4166666666666667, 1.25)
    """
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    obs = np.asarray(observed, dtype=np.float64).ravel()
    losses = (obs - pred) ** 2
    return SpatialResult(
        name="squared_error_loss",
        statistic=float(np.mean(losses)),
        p_value=None,
        extra={"losses": losses, "total_loss": float(np.sum(losses))},
    )


sglss = squared_error_loss


def cheatsheet() -> str:
    return "squared_error_loss({}) -> Squared error loss function."
