"""Median distance from the origin to the nearest of N uniform points in the unit ball (ESL eq 2.24)."""

from ._richresult import RichResult

__all__ = ["esl_median_nn_radius"]


def esl_median_nn_radius(N, p):
    r"""Median of the distance from the origin to its nearest neighbour among N points uniform in the p-ball.

    ESL eq 2.24: :math:`d(p, N) = (1 - (1/2)^{1/N})^{1/p}`, the curse of
    dimensionality in one number (N = 500, p = 10 gives about 0.52: the
    nearest point is more than halfway to the boundary). The distance is
    measured in units of the ball's radius, so the ball volume :math:`v_p`
    cancels (ESL Ex. 13 writes it with :math:`v_p^{-1/p}` for radius-r balls).

    Parameters
    ----------
    N : int
        Number of points, >= 1.
    p : int
        Dimension, >= 1.

    Returns
    -------
    RichResult
        ``median_radius``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 2.5.
    """
    N, p = int(N), int(p)
    if N < 1 or p < 1:
        raise ValueError("need N >= 1 and p >= 1")
    r = (1 - 0.5 ** (1 / N)) ** (1 / p)
    return RichResult(
        title="Median nearest-neighbour radius", summary_lines=[("median_radius", r)], payload={"median_radius": r}
    )


def cheatsheet():
    return "eslmnr: (1 - (1/2)^(1/N))^(1/p), ESL 2.24"
