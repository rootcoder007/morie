# morie.fn -- function file (rootcoder007/morie)
"""Joint null distribution of the two run counts."""

from math import comb

from ._richresult import RichResult

__all__ = ["gibbons_runs_joint_dist"]


def gibbons_runs_joint_dist(r1, r2, n1, n2):
    r"""Theorem 3.2.1 (PDF-verified, printed p. 77): under
    randomness,

    .. math:: f_{R_1,R_2}(r_1, r_2) = \frac{c \binom{n_1-1}{r_1-1}
              \binom{n_2-1}{r_2-1}}{\binom{n_1+n_2}{n_1}},

    where c = 2 if r_1 = r_2 and c = 1 if |r_1 - r_2| = 1; the
    probability is zero for any other (r_1, r_2), because runs of the
    two types must alternate.

    Parameters
    ----------
    r1, r2 : int
        Run counts of type 1 and type 2.
    n1, n2 : int
        Numbers of elements of each type.

    Returns
    -------
    RichResult
        keys: ``pmf``, ``c``, ``feasible``, ``r1``, ``r2``, ``n1``,
        ``n2``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Theorem 3.2.1.
    """
    r1, r2, n1, n2 = int(r1), int(r2), int(n1), int(n2)
    if n1 < 1 or n2 < 1:
        raise ValueError("n1 and n2 must be at least 1.")
    if not (1 <= r1 <= n1 and 1 <= r2 <= n2):
        raise ValueError("run counts must lie in 1..n of their type.")
    if abs(r1 - r2) > 1:
        # alternation makes this arrangement impossible, not merely rare
        return RichResult(
            payload={"pmf": 0.0, "c": 0, "feasible": False, "r1": r1,
                     "r2": r2, "n1": n1, "n2": n2,
                     "method": "Runs joint pmf (Gibbons Theorem 3.2.1)"}
        )
    c = 2 if r1 == r2 else 1
    pmf = c * comb(n1 - 1, r1 - 1) * comb(n2 - 1, r2 - 1) / comb(n1 + n2, n1)
    return RichResult(
        payload={"pmf": float(pmf), "c": c, "feasible": True, "r1": r1,
                 "r2": r2, "n1": n1, "n2": n2,
                 "method": "Runs joint pmf (Gibbons Theorem 3.2.1)"}
    )


def cheatsheet():
    return "gb321: c C(n1-1,r1-1)C(n2-1,r2-1)/C(n,n1); zero unless |r1-r2| <= 1"
