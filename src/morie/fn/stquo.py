# morie.fn -- function file (rootcoder007/morie)
"""Status quo vs proposal in the spatial voting model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["status_quo_spatial"]


def status_quo_spatial(ideal_points, status_quo, proposal):
    r"""Majority vote between a proposal and the status quo.

    Voter i supports the proposal exactly when it gives higher
    quadratic utility, i.e. lies closer to their ideal point:

    .. math:: \#\{i : \|x_i^* - x_{prop}\| < \|x_i^* - x_{sq}\|\}
              > n/2.

    Also reports each voter's cutting-plane side and the margin -- in
    one dimension the winner is simply the alternative closer to the
    median voter, which the tests check against the median directly.

    Parameters
    ----------
    ideal_points : array-like, shape (n,) or (n, k)
        Voter ideal points.
    status_quo : array-like, shape (k,) or scalar
        Status quo location.
    proposal : array-like, shape (k,) or scalar
        Proposal location.

    Returns
    -------
    RichResult
        keys: ``passes`` (strict majority for the proposal),
        ``votes_for``, ``votes_against``, ``margin``, ``supports``
        (n, boolean), ``indifferent`` (count of exact ties), ``n``,
        ``method``.

    References
    ----------
    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Ch. 1 (proximity voting between
    a proposal and the status quo), p. 1.
    """
    X = np.atleast_2d(np.asarray(ideal_points, dtype=float))
    if X.shape[0] == 1 and np.ndim(ideal_points) == 1:
        X = X.T  # a 1-D list of scalar ideal points is n voters in 1 dim
    sq = np.atleast_1d(np.asarray(status_quo, dtype=float))
    pr = np.atleast_1d(np.asarray(proposal, dtype=float))
    if sq.shape != pr.shape or sq.size != X.shape[1]:
        raise ValueError("status_quo and proposal must share the ideal points' dimension.")

    d_sq = np.sqrt(((X - sq) ** 2).sum(axis=1))
    d_pr = np.sqrt(((X - pr) ** 2).sum(axis=1))
    supports = d_pr < d_sq
    ties = np.isclose(d_pr, d_sq)
    votes_for = int(supports.sum())
    votes_against = int((~supports & ~ties).sum())
    n = X.shape[0]

    return RichResult(
        payload={
            "passes": bool(votes_for > n / 2),
            "votes_for": votes_for,
            "votes_against": votes_against,
            "margin": votes_for - votes_against,
            "supports": supports,
            "indifferent": int(ties.sum()),
            "n": int(n),
            "method": "Spatial majority vote: proposal vs status quo (quadratic utility)",
        }
    )


def cheatsheet():
    return "stquo: proposal passes iff closer than the status quo for a strict majority"
