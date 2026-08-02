# morie.fn -- function file (rootcoder007/morie)
"""IRT scale linking: Haebara method."""

from __future__ import annotations

from . import _array_core as np
from ._sci_core import minimize

from ._richresult import RichResult

__all__ = ["equating_haebara"]


def equating_haebara(a_ref, b_ref, a_focal, b_focal, n_quad=41, theta_range=4.0):
    r"""Link two IRT calibrations by the Haebara criterion.

    Separate calibrations of the same items on different groups produce
    parameters on different scales. A linear transform
    :math:`\theta^* = A\theta + B` restores comparability, with items
    transformed as :math:`a^* = a/A` and :math:`b^* = Ab + B`.

    Haebara minimises the summed squared difference between **item-level**
    response curves at each ability:

    .. math::
        F(A, B) = \sum_\theta w(\theta)
            \sum_j \left[P_j(\theta) - P_j^*(\theta)\right]^2 .

    Because the discrepancy is accumulated per item, cancellation between
    items is impossible: an item fitting badly in one direction cannot be
    offset by another fitting badly in the other. Stocking-Lord sums the
    curves first and so permits exactly that cancellation, which is the
    substantive difference between the two.

    Characteristic-curve methods like this one dominate the moment methods
    (mean-mean, mean-sigma) because they use the **whole response function**
    rather than two summary statistics, so a few badly estimated items move
    the solution far less.

    Linking assumes the items are genuinely the same and function identically
    in both groups. Items with DIF violate that and should be removed from the
    anchor set before linking rather than after, since a single badly
    functioning anchor item distorts the transform for every other item.

    Parameters
    ----------
    a_ref, b_ref : array-like
        Discrimination and difficulty on the reference scale.
    a_focal, b_focal : array-like
        The same items calibrated on the focal scale.
    n_quad : int
        Quadrature points over the ability range.
    theta_range : float
        Ability range, symmetric about zero.

    Returns
    -------
    RichResult
        ``A``, ``B``, ``criterion``, ``a_transformed``, ``b_transformed``,
        ``converged``.

    References
    ----------
        Haebara, T. (1980). Equating logistic ability scales by a weighted least
        squares method. *Japanese Psychological Research*, 22(3), 144-149.

    Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and
        Linking* (3rd ed.). Springer.

    Examples
    --------
    A known linear transform is recovered.

    >>> import numpy as np
    >>> a = np.array([1.0, 1.2, 0.8, 1.5, 0.9])
    >>> b = np.array([-1.0, 0.0, 0.5, 1.0, -0.5])
    >>> A_true, B_true = 1.3, 0.4
    >>> r = equating_haebara(a, b, a * A_true, (b - B_true) / A_true)
    >>> bool(abs(r["A"] - A_true) < 0.02 and abs(r["B"] - B_true) < 0.02)
    True

    The transform maps the focal parameters back onto the reference scale.

    >>> bool(np.max(np.abs(r["b_transformed"] - b)) < 0.02)
    True

    Identical calibrations need no transform.

    >>> ident = equating_haebara(a, b, a, b)
    >>> bool(abs(ident["A"] - 1.0) < 0.01 and abs(ident["B"]) < 0.01)
    True
    """
    a_r = np.atleast_1d(np.asarray(a_ref, dtype=float)).ravel()
    b_r = np.atleast_1d(np.asarray(b_ref, dtype=float)).ravel()
    a_f = np.atleast_1d(np.asarray(a_focal, dtype=float)).ravel()
    b_f = np.atleast_1d(np.asarray(b_focal, dtype=float)).ravel()
    if not (a_r.size == b_r.size == a_f.size == b_f.size):
        raise ValueError("all four parameter vectors must have the same length")
    if a_r.size == 0:
        raise ValueError("need at least one anchor item")
    if np.any(a_r <= 0) or np.any(a_f <= 0):
        raise ValueError("discriminations must be positive")

    th = np.linspace(-theta_range, theta_range, int(n_quad))
    w = np.exp(-0.5 * th**2)
    w /= w.sum()

    def P(a, b, t):
        return 1.0 / (1.0 + np.exp(-1.7 * np.outer(a, t - 0) + 1.7 * np.outer(a, b)[:, :1] * 0
                                   - 0.0)) if False else \
            1.0 / (1.0 + np.exp(-1.7 * a[:, None] * (t[None, :] - b[:, None])))

    def crit(par):
        A, B = par
        if A <= 0:
            return 1e12
        a_star = a_f / A
        b_star = A * b_f + B
        Pr = P(a_r, b_r, th)
        Pf = P(a_star, b_star, th)
        d = ((Pr - Pf) ** 2).sum(axis=0)
        return float(np.sum(w * d))

    res = minimize(crit, np.array([1.0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
    A, B = float(res.x[0]), float(res.x[1])
    return RichResult(
        title="Haebara linking",
        summary_lines=[("items", int(a_r.size)), ("A", A), ("B", B),
                       ("criterion", float(res.fun))],
        warnings=["remove anchor items showing DIF BEFORE linking; one badly "
                  "functioning anchor distorts the transform for every item"],
        payload={
            "A": A, "B": B, "criterion": float(res.fun),
            "a_transformed": a_f / A, "b_transformed": A * b_f + B,
            "n_items": int(a_r.size), "converged": bool(res.success),
            "method": "equating_haebara",
        },
    )


def cheatsheet():
    return "eqhae: Haebara uses the WHOLE response curve, not two moments; drop DIF anchors before linking"
