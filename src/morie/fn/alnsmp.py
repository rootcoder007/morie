# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Skip-gram with negative sampling (Mikolov et al. 2013;
Alammar Ch 2)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["alammar_negative_sampling_skipgram"]


def alammar_negative_sampling_skipgram(center_vec, context_vec,
                                       negative_vecs, V=None):
    """L = -log sigma(v_c . v_w) - sum_i log sigma(-v_c . v_ni).

    References: Alammar and Grootendorst, Ch 2; Mikolov et al. (2013).
    """
    c = np.atleast_1d(np.asarray(center_vec, dtype=float))
    w = np.atleast_1d(np.asarray(context_vec, dtype=float))
    N = np.atleast_2d(np.asarray(negative_vecs, dtype=float))
    if c.shape != w.shape or N.shape[1] != len(c):
        raise ValueError("all vectors must share one dimension.")

    def logsig(z):
        # stable log sigmoid
        return -np.logaddexp(0.0, -z)

    pos = float(logsig(np.dot(c, w)))
    negs = [float(logsig(-np.dot(c, N[i]))) for i in range(N.shape[0])]
    loss = -(pos + sum(negs))
    return RichResult(payload={
        "estimate": loss, "positive_logsig": pos,
        "negative_logsigs": negs, "k": N.shape[0], "n": len(c),
        "method": "Skip-gram negative sampling (Mikolov et al. 2013)"})


def cheatsheet():
    return "alnsmp: -log sig(c.w) - sum log sig(-c.n_i), stable logsigmoid"
