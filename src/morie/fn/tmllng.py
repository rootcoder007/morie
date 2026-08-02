# morie.fn -- function file (rootcoder007/morie)
"""TMLE for longitudinal data with time-varying treatments and confounders."""

from . import _array_core as np

from ._richresult import RichResult
from .tmltvc import tmle_time_varying_confound

__all__ = ["tmle_longitudinal"]


def tmle_longitudinal(y, A, L, trunc=0.01):
    """Always-treat vs never-treat contrast by longitudinal TMLE.

    Runs :func:`morie.fn.tmltvc.tmle_time_varying_confound` under the
    two static regimes (all 1s and all 0s) and differences them. This
    is the longitudinal analogue of the g-formula contrast in
    :mod:`morie.fn.gctvc`, but with a targeting step at every time
    point rather than plain substitution.

    Returns
    -------
    RichResult
        keys: ``estimate`` (contrast), ``ey_always``, ``ey_never``,
        ``n_periods``, ``n``, ``method``.

    References
    ----------
    van der Laan, M. J. & Gruber, S. (2012). Targeted minimum
    loss-based estimation of causal effects of multiple time point
    interventions. *The International Journal of Biostatistics*, 8(1),
    Article 9.
    """
    A = np.asarray(A, dtype=float)
    T = 1 if A.ndim == 1 else A.shape[1]
    hi = tmle_time_varying_confound(y, A, L, regime=np.ones(T), trunc=trunc)
    lo = tmle_time_varying_confound(y, A, L, regime=np.zeros(T), trunc=trunc)
    return RichResult(
        payload={
            "estimate": hi["estimate"] - lo["estimate"],
            "ey_always": hi["estimate"],
            "ey_never": lo["estimate"],
            "n_periods": int(T),
            "n": hi["n"],
            "method": "Longitudinal TMLE: always-treat minus never-treat",
        }
    )


def cheatsheet():
    return "tmllng: tmltvc under abar = 1 and abar = 0, differenced"
