# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 8.7: BERTScore recall."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch8_bertscore_recall"]


def _sim_matrix(x, xhat, normalize):
    X = np.atleast_2d(np.asarray(x, dtype=float))
    Y = np.atleast_2d(np.asarray(xhat, dtype=float))
    if X.size == 0 or Y.size == 0:
        raise ValueError("both token-embedding matrices must be "
                         "non-empty.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"embedding widths differ: {X.shape[1]} vs {Y.shape[1]}.")
    if normalize:
        nx = np.linalg.norm(X, axis=1)
        ny = np.linalg.norm(Y, axis=1)
        if np.any(nx == 0) or np.any(ny == 0):
            raise ValueError("a zero token embedding has no direction; "
                             "its cosine similarity is undefined.")
        X = X / nx[:, None]
        Y = Y / ny[:, None]
    return X, Y, X @ Y.T


def kamath_ch8_bertscore_recall(x, xhat, normalize=False):
    r"""R_BERT = (1/|x|) sum_{x_i in x} max_j <x_i, xhat_j>.

    ``x`` is the REFERENCE token-embedding matrix (one row per token)
    and ``xhat`` the candidate's. The book writes a plain inner
    product because BERTScore pre-normalizes its embeddings; pass
    ``normalize=True`` to make the inner products cosines here.
    ``greedy_match`` records the reference-to-candidate alignment.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 8, Eq 8.7, printed
    p. 325.

    Examples
    --------
    >>> out = kamath_ch8_bertscore_recall([[1.0, 0.0], [0.0, 1.0]],
    ...                                   [[1.0, 0.0]])
    >>> out["estimate"]            # (1 + 0) / 2
    0.5
    """
    X, Y, S = _sim_matrix(x, xhat, normalize)
    best = S.max(axis=1)
    return RichResult(payload={
        "estimate": float(best.mean()),
        "per_token": [float(v) for v in best],
        "greedy_match": [int(j) for j in S.argmax(axis=1)],
        "n": int(X.shape[0]),
        "method": "BERTScore recall (Kamath Eq 8.7)"})


def cheatsheet():
    return "km119: mean over reference tokens of best candidate match"
