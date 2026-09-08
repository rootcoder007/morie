# morie.fn -- function file (rootcoder007/morie)
"""Longitudinal mediation (cross-lagged panel)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["longitudinal_mediation"]


def longitudinal_mediation(x, m, y):
    r"""Cross-lagged panel mediation over three waves.

    Estimates the lagged paths with autoregressive controls:

    .. math::
        M_2 &= a M_1' \ldots \; a\, X_1 + s_M M_1, \\
        Y_3 &= b\, M_2 + c' X_1 + s_Y Y_2,

    so the indirect effect :math:`ab` is the effect of wave-1 X on
    wave-2 M and of wave-2 M on wave-3 Y, each net of the mediator's
    and outcome's own earlier level. Controlling the autoregressive
    paths is what distinguishes a cross-lagged estimate from a
    cross-sectional one: without them, stable between-person
    differences masquerade as an over-time effect.

    Parameters
    ----------
    x, m, y : array-like, shape (n, T) with T >= 3
        Panel measurements of treatment, mediator and outcome; column
        t is wave t.

    Returns
    -------
    RichResult
        keys: ``a``, ``b``, ``indirect``, ``direct``, ``total``,
        ``ar_m``, ``ar_y`` (autoregressive coefficients),
        ``n``, ``n_waves``, ``method``.

    References
    ----------
    Cole, D. A. & Maxwell, S. E. (2003). Testing mediational models
    with longitudinal data: questions and tips in the use of
    structural equation modeling. *Journal of Abnormal Psychology*,
    112(4), 558-577.
    """
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or m.shape != x.shape or y.shape != x.shape:
        raise ValueError("x, m, y must be (n, T) arrays of the same shape.")
    n, T = x.shape
    if T < 3:
        raise ValueError(f"need at least 3 waves, got {T}.")
    if n < 8:
        raise ValueError(f"need at least 8 units, got {n}.")

    def ols(cols, t):
        D = np.column_stack([np.ones(n), *cols])
        b, *_ = np.linalg.lstsq(D, t, rcond=None)
        return b

    bm = ols([x[:, 0], m[:, 0]], m[:, 1])
    a, ar_m = float(bm[1]), float(bm[2])
    by = ols([m[:, 1], x[:, 0], y[:, 1]], y[:, 2])
    b, cprime, ar_y = float(by[1]), float(by[2]), float(by[3])

    return RichResult(
        payload={
            "a": a,
            "b": b,
            "indirect": a * b,
            "direct": cprime,
            "total": cprime + a * b,
            "ar_m": ar_m,
            "ar_y": ar_y,
            "n": int(n),
            "n_waves": int(T),
            "method": "Cross-lagged panel mediation (X1 -> M2 -> Y3, autoregressive controls)",
        }
    )


def cheatsheet():
    return "longMd: a = X1->M2 | M1; b = M2->Y3 | Y2; indirect = ab"
