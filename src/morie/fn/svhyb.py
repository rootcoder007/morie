"""Hybrid proximity-valence model.

Category: Spatial
"""

from . import _array_core as np
from ._containers import DescriptiveResult


def svhyb(voter, candidates=None, saliences=None):
    """Hybrid proximity-valence model.

    Parameters
    ----------
    voter : array-like
        Voter position in issue space.
    candidates : array-like, optional
        Candidate positions (n_candidates x n_dims).
    saliences : array-like, optional
        Issue salience weights.

    Returns
    -------
    DescriptiveResult
    """
    voter = np.asarray(voter, dtype=float)
    candidates = np.zeros((1, len(voter))) if candidates is None else np.asarray(candidates, dtype=float)
    saliences = np.ones(len(voter)) / len(voter) if saliences is None else np.asarray(saliences, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    dists = np.array([float(np.sum(saliences * (voter - c) ** 2)) for c in candidates])
    utilities = np.exp(-0.5 * dists)
    stat = float(np.max(utilities))
    return DescriptiveResult(
        name="svhyb",
        value=stat,
        extra={"n_candidates": len(candidates), "utilities": utilities.tolist()},
    )


short = "svhyb"
alias = "svhyb"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
svhyb = svhyb


def cheatsheet() -> str:
    return "svhyb({}) -> Hybrid proximity-valence model."
