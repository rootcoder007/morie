# morie.fn -- function file (rootcoder007/morie)
"""IRT scale linking: Stocking-Lord method."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._richresult import RichResult

__all__ = ["equating_stocking_lord"]


def equating_stocking_lord(a_ref, b_ref, a_focal, b_focal, n_quad=41, theta_range=4.0):
    r"""Link two IRT calibrations by the Stocking-Lord criterion.

    Separate calibrations of the same items on different groups produce
    parameters on different scales. A linear transform
    :math:`\theta^* = A\theta + B` restores comparability, with items
    transformed as :math:`a^* = a/A` and :math:`b^* = Ab + B`.

    Stocking-Lord minimises the squared difference between **test
    characteristic curves** -- the summed response functions:

    .. math::
        F(A, B) = \sum_\theta w(\theta)
            \left[\sum_j P_j(\theta) - \sum_j P_j^*(\theta)\right]^2 .

    Summing before differencing is the whole distinction from Haebara, and it
    cuts both ways. It targets the quantity that actually matters for scoring
    -- expected total score -- so it is the right criterion when the test is
    reported as a total. But it permits **cancellation**: items misfitting in
    opposite directions offset inside the sum, so a transform can look
    excellent at test level while individual items are poorly aligned.

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
        Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in
        item response theory. *Applied Psychological Measurement*, 7(2),
        201-210.

    Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and
        Linking* (3rd ed.). Springer.

    Examples
    --------
    A known linear transform is recovered.

    >>> import numpy as np
    >>> a = np.array([1.0, 1.2, 0.8, 1.5, 0.9])
    >>> b = np.array([-1.0, 0.0, 0.5, 1.0, -0.5])
    >>> A_true, B_true = 1.3, 0.4
    >>> r = equating_stocking_lord(a, b, a * A_true, (b - B_true) / A_true)
    >>> bool(abs(r["A"] - A_true) < 0.05 and abs(r["B"] - B_true) < 0.05)
    True

    The two criteria agree closely when every item is well behaved.

    >>> from morie.fn.eqhae import equating_haebara
    >>> h = equating_haebara(a, b, a * A_true, (b - B_true) / A_true)
    >>> bool(abs(r["A"] - h["A"]) < 0.05)
    True

    Its criterion is computed on the summed curve, so it is not comparable in
    magnitude to Haebara's item-level one.

    >>> bool(r["criterion"] >= 0)
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
        d = (Pr.sum(axis=0) - Pf.sum(axis=0)) ** 2
        return float(np.sum(w * d))

    res = minimize(crit, np.array([1.0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
    A, B = float(res.x[0]), float(res.x[1])
    return RichResult(
        title="Stocking-Lord linking",
        summary_lines=[("items", int(a_r.size)), ("A", A), ("B", B),
                       ("criterion", float(res.fun))],
        warnings=["remove anchor items showing DIF BEFORE linking; one badly "
                  "functioning anchor distorts the transform for every item"],
        payload={
            "A": A, "B": B, "criterion": float(res.fun),
            "a_transformed": a_f / A, "b_transformed": A * b_f + B,
            "n_items": int(a_r.size), "converged": bool(res.success),
            "method": "equating_stocking_lord",
        },
    )


def cheatsheet():
    return "eqsl: Stocking-Lord uses the WHOLE response curve, not two moments; drop DIF anchors before linking"
