# morie.fn -- function file (rootcoder007/morie)
"""Agreement score matrix from roll-call votes."""

from ._richresult import RichResult
from .agrmt import agreement_score

__all__ = ["agreement_score_matrix"]


def agreement_score_matrix(vote_matrix):
    """Pairwise agreement matrix; front-end to :mod:`morie.fn.agrmt`.

    ``A_ij`` = proportion of shared (both non-missing) votes on which
    legislators i and j voted the same way (Armstrong Sec. 3.2.2,
    p. 88). Returns the matrix directly rather than the descriptive
    wrapper.

    References
    ----------
    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Sec. 3.2.2 (Agreement Scores),
    p. 88.
    """
    out = agreement_score(vote_matrix)
    return RichResult(
        payload={
            "matrix": out.value["agreement_matrix"],
            "n_shared_votes": out.extra["n_shared_votes"],
            "mean_agreement": out.extra["mean_agreement"],
            "n": out.extra["n_legislators"],
            "method": "Agreement score matrix (Armstrong Sec 3.2.2 p.88)",
        }
    )


def cheatsheet():
    return "agrsc: A_ij = shared votes cast the same way / shared votes (front-end to agrmt)"
