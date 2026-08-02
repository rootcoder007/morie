# morie.fn -- function file (rootcoder007/morie)
"""Cox profile score."""

from . import _array_core as np

from scipy import optimize

from ._kosorok import cox_score
from ._richresult import RichResult

__all__ = ["kosorok_ch3_cox_profile_score"]


def kosorok_ch3_cox_profile_score(beta=None, Z=None, Y=None, X=None, tau=None,
                                  n=None, time=None, event=None):
    r"""Empirical profile score for the Cox model (Kosorok Ch. 3):

    .. math:: \hat\ell_{\beta,n} = \int_0^\tau \Big\{ Z -
              \frac{\mathbb P_n[Z Y(t) e^{\beta'Z}]}
                    {\mathbb P_n[Y(t) e^{\beta'Z}]}\Big\} dM(t),

    the sample version of :mod:`morie.fn.ksr063` with the population
    measure P replaced by the empirical measure. Solving it for zero
    is exactly Cox's partial-likelihood estimator, and the profile
    information equals the efficient information -- so profiling out
    an infinite-dimensional nuisance loses nothing here.

    Returns the score at the supplied beta and, additionally, the
    root, so the two can be compared.

    Parameters
    ----------
    Z, time, event : array-like
        Covariates, follow-up times, 0/1 indicators. ``Y``/``X`` are
        accepted as aliases for ``time`` for interface compatibility.
    beta : array-like, optional
        Where to evaluate the score; zero if omitted.
    tau, n : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``score_at_beta``, ``beta_hat`` (the root),
        ``score_at_root`` (numerically zero), ``information``,
        ``loglik``, ``converged``, ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 3 (profile likelihood; the Cox example).
    """
    if time is None:
        time = Y if Y is not None else X
    if time is None or event is None or Z is None:
        raise ValueError("supply Z, time (or Y/X) and event.")
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    p = Z.shape[1]
    b0 = np.zeros(p) if beta is None else np.atleast_1d(np.asarray(beta, float))
    at_beta = cox_score(b0, Z, time, event)

    res = optimize.root(
        lambda b: cox_score(b, Z, time, event)["score"], np.zeros(p), method="hybr"
    )
    root = res.x
    at_root = cox_score(root, Z, time, event)
    return RichResult(
        payload={"score_at_beta": at_beta["score"], "beta_hat": root,
                 "score_at_root": at_root["score"],
                 "information": at_root["information"],
                 "loglik": at_root["loglik"], "converged": bool(res.success),
                 "n": at_root["n"],
                 "method": "Empirical profile score; its root is the Cox estimator"}
    )


def cheatsheet():
    return "ksr068: empirical version of ksr063; root = partial-likelihood MLE"
