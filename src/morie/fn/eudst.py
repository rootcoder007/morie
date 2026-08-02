# morie.fn -- function file (rootcoder007/morie)
"""Quadratic Euclidean utility for the spatial voting model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["euclidean_utility"]


def euclidean_utility(ideal_point, policy_position):
    r"""Quadratic spatial utility.

    .. math:: U_i(x_j) = -\|x_i^* - x_j\|^2
              = -\sum_d (x_{id} - x_{jd})^2.

    Accepts a single voter against a single policy, or arrays: with
    ``ideal_point`` of shape (n, k) and ``policy_position`` of shape
    (m, k), the result is the (n, m) utility matrix.

    Parameters
    ----------
    ideal_point : array-like, shape (k,) or (n, k)
        Voter ideal point(s).
    policy_position : array-like, shape (k,) or (m, k)
        Policy location(s).

    Returns
    -------
    RichResult
        keys: ``utility`` (scalar or (n, m)), ``distance`` (the
        corresponding Euclidean distances), ``method``.

    References
    ----------
    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Ch. 1 (the spatial model and
    quadratic utility), p. 1.
    """
    x = np.atleast_2d(np.asarray(ideal_point, dtype=float))
    p = np.atleast_2d(np.asarray(policy_position, dtype=float))
    if x.shape[1] != p.shape[1]:
        raise ValueError(
            f"dimension mismatch: ideal points have {x.shape[1]} coordinates, "
            f"policies have {p.shape[1]}."
        )
    diff = x[:, None, :] - p[None, :, :]
    sq = (diff**2).sum(axis=2)
    scalar = np.ndim(ideal_point) <= 1 and np.ndim(policy_position) <= 1

    return RichResult(
        payload={
            "utility": float(-sq[0, 0]) if scalar else -sq,
            "distance": float(np.sqrt(sq[0, 0])) if scalar else np.sqrt(sq),
            "method": "Quadratic Euclidean spatial utility",
        }
    )


def cheatsheet():
    return "eudst: U = -||x* - x||^2, scalar or (n, m) matrix"
