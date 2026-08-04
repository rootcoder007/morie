# morie.fn -- function file (rootcoder007/morie)
"""Variance of the modified gamma kernel density estimator (Theorem 1.5)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["mgkvar", "fauzi_thm1_5_consistency_mgkde"]


def mgkvar(varh, var4h, cov, n=None, boundary=False):
    r"""Variance of the modified gamma kernel density estimator (Theorem 1.5).

    Theorem 1.5:

    .. math:: \mathrm{Var}[\tilde f_X(x)] = 4\mathrm{Var}[A_h(x)]
              + \mathrm{Var}[A_{4h}(x)] - 4\mathrm{Cov}[A_h(x),A_{4h}(x)]
              + o(n^{-1}h^{-1/4}).

    Not a new calculation -- it is the variance of the linear combination
    :math:`2A_h - A_{4h}`, which the proof reaches by showing
    :math:`J_h/J_{4h} = 1 + O(\sqrt h)` so the nonlinear ratio in (1.14)
    linearises. Because it is a linear combination, the ORDERS do not
    change: :math:`O(n^{-1}h^{-1/4})` in the interior and
    :math:`O(n^{-1}h^{-3/4})` at the boundary.

    Combining with Theorem 1.3, the MSE is
    :math:`O(h^2) + O(n^{-1}h^{-1/4})` in the interior, optimised at
    :math:`h = O(n^{-4/9})` for a rate of :math:`O(n^{-8/9})`; at the
    boundary :math:`O(h^2)+O(n^{-1}h^{-3/4})`, optimised at
    :math:`h=O(n^{-4/11})` for :math:`O(n^{-8/11})`. Both beat Chen's
    :math:`O(n^{-4/5})` and :math:`O(n^{-2/3})`. Those rates are returned
    as ``hopt`` and ``mserate`` so the claim is checkable, not decorative.

    Parameters
    ----------
    varh : float
        ``Var[A_h(x)]``, e.g. from :func:`morie.fn.fzt11.gkrawbv`.
    var4h : float
        ``Var[A_{4h}(x)]``.
    cov : float
        ``Cov[A_h(x), A_{4h}(x)]``, e.g. from :func:`morie.fn.fzt14.gkcov`.
    n : int, optional
        Sample size; only used to evaluate the optimal-bandwidth rates.
    boundary : bool, default False
        Report the boundary-region rates instead of the interior ones.

    Returns
    -------
    RichResult
        Keys ``variance``, ``hopt``, ``mserate``, ``region``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 1.5, Eqs. (1.18)-(1.19).
    """
    var_t = 4.0 * float(varh) + float(var4h) - 4.0 * float(cov)
    if boundary:
        pow_h, pow_mse, region = -4.0 / 11.0, -8.0 / 11.0, "boundary"
    else:
        pow_h, pow_mse, region = -4.0 / 9.0, -8.0 / 9.0, "interior"
    if n is None:
        hopt = np.nan
        mserate = np.nan
    else:
        nn = int(n)
        if nn < 1:
            raise ValueError(f"sample size must be at least 1, got {nn}.")
        hopt = float(nn) ** pow_h
        mserate = float(nn) ** pow_mse
    return RichResult(
        payload={
            "variance": float(var_t),
            "hopt": float(hopt),
            "mserate": float(mserate),
            "region": region,
            "method": "modified gamma KDE variance (Theorem 1.5)",
        }
    )


fauzi_thm1_5_consistency_mgkde = mgkvar


def cheatsheet():
    return "fzt15: Var of the modified gamma KDE = Var[2 A_h - A_4h]; MSE rate n^(-8/9) interior (Thm 1.5)"


# CANONICAL TEST
# >>> r = mgkvar(varh=0.02, var4h=0.01, cov=0.005, n=100)
# >>> abs(r['variance'] - 0.07) < 1e-15
# True
