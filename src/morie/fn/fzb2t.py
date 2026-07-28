# morie.fn -- function file (rootcoder007/morie)
"""b_2(t) coefficient in the boundary-free bias expansion."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_b2_coefficient"]


def fauzi_b2_coefficient(t, f_X, f_X_prime=None, S_X=None, transform="log"):
    r"""The :math:`b_2(t)` bias coefficient (Fauzi Eq. 4.15):

    .. math:: b_2(t) = [g'(g^{-1}(t))]^2 f_X(t) + \\int_{g^{-1}(t)}^{\\infty} g''(x)g'(x)f_X(g(x))\\,dx,

    It governs the bias of the FIRST cumulative survival estimator (4.12), and it carries an integral tail that the other two do not.

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
        keys: ``t``, ``b_2``, ``g_prime``, ``g_double_prime``,
        ``bias_order``, ``transform``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Eq. (4.15). Transcribed from the PDF.
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
    # the integral tail is evaluated by quadrature on the
    # transformed scale, where g'' g' f_X(g(.)) is smooth
    b2 = np.empty(tv.size)
    for i, zv in enumerate(zt):
        zz = np.linspace(zv, zv + 12.0, 400)
        vals = tr["d2g"](zz) * tr["dg"](zz) * np.interp(
            tr["g"](zz), tv, fx, left=0.0, right=0.0)
        b2[i] = gp[i] ** 2 * fx[i] + float(np.trapezoid(vals, zz))

    return RichResult(payload={
        "t": tv, "b_2": b2, "g_prime": gp, "g_double_prime": gpp,
        "bias_order": "O(h^2) everywhere, including the boundary region",
        "contrast": "the naive kernel estimator degrades to O(h) or O(1) "
                    "at the boundary (Remark 4.5)",
        "transform": tr["name"],
        "method": "b_2 from Eq. (4.15); the transformation makes the bias constant computable"})


def cheatsheet():
    return "fzb2t: b_2 is why the transformed bias stays O(h^2) at the edge"
