# morie.fn -- function file (rootcoder007/morie)
"""Consistency and boundedness of the estimated efficient score."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_eff_score_consistency", "kosorok_ch3_z_estimator_consistency_score"]


def kosorok_eff_score_consistency(scores_est, scores_true):
    r"""The two efficient-score regularity conditions (Kosorok
    Eq. 3.7, p. 44):

    .. math:: P_{\theta,\eta}\big\|\tilde\ell_{\theta_n,\eta_n}
              - \tilde\ell_{\theta,\eta}\big\|^2 \to 0,
              \qquad
              P_{\theta_n,\eta_n}
              \big\|\tilde\ell_{\theta_n,\eta_n}\big\|^2
              = O_P(1).

    A convergence requirement and a boundedness requirement, and
    they do different jobs. The first says the estimated efficient
    score converges in mean square to the true one -- without it the
    estimating equation is not asymptotically the right one. The
    second says the estimated score does not blow up under its OWN
    law, which rules out the usual failure mode of plug-in
    semiparametric estimators, where a nuisance estimate that is
    consistent but poorly behaved in the tails produces scores with
    exploding second moments.

    Both are returned separately because a fit can satisfy one and
    fail the other, and only the pair licenses Theorem 3.1.

    Parameters
    ----------
    scores_est : array-like, shape (n, p)
        Efficient scores at the estimated nuisance.
    scores_true : array-like, shape (n, p)
        Efficient scores at the true nuisance.

    Returns
    -------
    RichResult
        keys: ``mean_square_difference``, ``second_moment``,
        ``converges``, ``bounded``, ``both_hold``, ``n``, ``p``,
        ``method``.
    References
    ----------
    Kosorok, Ch. 3, Eq. (3.7), p. 44.
    """
    A = np.atleast_2d(np.asarray(scores_est, dtype=float))
    B = np.atleast_2d(np.asarray(scores_true, dtype=float))
    if A.shape != B.shape:
        raise ValueError(f"shapes differ: {A.shape} and {B.shape}.")
    n, p = A.shape
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    msd = float(np.mean(np.sum((A - B) ** 2, axis=1)))
    sm = float(np.mean(np.sum(A ** 2, axis=1)))
    return RichResult(payload={
        "mean_square_difference": msd, "second_moment": sm,
        "converges": bool(msd < 1.0),
        "bounded": bool(np.isfinite(sm)),
        "both_hold": bool(msd < 1.0 and np.isfinite(sm)),
        "roles": "the first makes the estimating equation asymptotically correct; "
                 "the second rules out exploding second moments under the estimated law",
        "n": int(n), "p": int(p),
        "method": "Efficient-score regularity (Eq. 3.7); convergence AND boundedness, separately"})


def cheatsheet():
    return "ksr067: a fit can converge in mean square and still explode under its own law"


#: Catalogue alias for :func:`kosorok_eff_score_consistency`.
kosorok_ch3_z_estimator_consistency_score = kosorok_eff_score_consistency
