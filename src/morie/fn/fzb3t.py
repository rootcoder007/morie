# morie.fn -- function file (rootcoder007/morie)
"""b_3(t) coefficient in the boundary-free bias expansion."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_b3_coefficient"]


def fauzi_b3_coefficient(t, f_X, f_X_prime=None, S_X=None, transform="log"):
    r"""The :math:`b_3(t)` bias coefficient (Fauzi Eq. 4.21):

    .. math:: b_3(t) = [g'(g^{-1}(t))]^2 f_X(t) - g''(g^{-1}(t))\\,S_X(t),

    It governs the bias of the SECOND cumulative survival estimator (4.19). Note the MINUS sign and the survival function where :math:`b_2` has an integral -- the two estimators are mirror images and their bias constants differ accordingly.

    These coefficients are what the bijective transformation buys.
    The transformation does not remove the bias; it makes the bias
    O(h^2) EVERYWHERE, boundary included, and expresses its constant
    through derivatives of :math:`g`. Remark 4.5 is explicit that the
    naive kernel method's bias degrades from :math:`h^2` to
    :math:`h` or even to :math:`O(1)` in the boundary region, which
    is the analytic reason the transformed estimators outperform it.

    Parameters
    ----------
    t : array-like
        Evaluation points inside the support.
    f_X : array-like
        Density at ``t``.
    f_X_prime : array-like, optional
        Density derivative at ``t``, where the coefficient needs it.
    S_X : array-like, optional
        Survival function at ``t``, where the coefficient needs it.
    transform : {"log", "identity"}
        The bijection supplying ``g'`` and ``g''``.

    Returns
    -------
    RichResult
        keys: ``t``, ``b_3``, ``g_prime``, ``g_double_prime``,
        ``bias_order``, ``transform``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Eq. (4.21). Transcribed from the PDF.
    """
    from ._fauzi import boundary_free_transform

    tv = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    fx = np.atleast_1d(np.asarray(f_X, dtype=float)).ravel()
    if fx.size != tv.size:
        raise ValueError(f"f_X has {fx.size} entries for {tv.size} points.")
    if np.any(fx < 0):
        raise ValueError("a density must be non-negative.")
    tr = boundary_free_transform(transform)
    lo, hi = tr["support"]
    if np.any(tv <= lo) or np.any(tv >= hi):
        raise ValueError("t must lie strictly inside the support.")
    zt = tr["g_inv"](tv)
    gp = tr["dg"](zt)
    gpp = tr["d2g"](zt)
    if S_X is None:
        raise ValueError("b_3 needs the survival function S_X.")
    sx = np.atleast_1d(np.asarray(S_X, dtype=float)).ravel()
    if sx.size != tv.size:
        raise ValueError(f"S_X has {sx.size} entries for {tv.size}.")
    if np.any((sx < 0) | (sx > 1)):
        raise ValueError("S_X must lie in [0, 1].")
    b3 = gp ** 2 * fx - gpp * sx

    return RichResult(payload={
        "t": tv, "b_3": b3, "g_prime": gp, "g_double_prime": gpp,
        "bias_order": "O(h^2) everywhere, including the boundary region",
        "contrast": "the naive kernel estimator degrades to O(h) or O(1) "
                    "at the boundary (Remark 4.5)",
        "transform": tr["name"],
        "method": "b_3 from Eq. (4.21); the transformation makes the bias constant computable"})


def cheatsheet():
    return "fzb3t: b_3 is why the transformed bias stays O(h^2) at the edge"
