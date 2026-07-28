# morie.fn -- function file (rootcoder007/morie)
"""Joint convergence of parameter and nuisance (Kosorok Cor 3.2)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_joint_convergence"]


def kosorok_joint_convergence(psi_dot, scores, n=None):
    r"""Joint weak convergence of the parameter and the nuisance
    (Kosorok Cor. 3.2, p. 47):

    .. math:: \sqrt n\big(\hat\theta_n - \theta_0,\;
              \hat\eta_n - \eta_0\big)
              \rightsquigarrow -\dot\Psi_0^{-1} Z,

    under the no-bias condition and stochastic equicontinuity.

    JOINTLY, not separately, and that is the content. The two
    estimates are correlated -- they solve the same estimating
    equation -- so a valid confidence region for a function of both,
    or for :math:`\theta` after substituting :math:`\hat\eta`,
    needs the joint law. Reporting marginal limits and combining
    them as if independent understates the variability, and
    ``jointly`` records that this is the joint statement.

    The single operator :math:`\dot\Psi_0^{-1}` acting on one
    Gaussian limit is what delivers it: both blocks come from
    inverting the SAME derivative, which is why their dependence is
    determined rather than an extra assumption.

    Parameters
    ----------
    psi_dot : array-like
        The derivative operator :math:`\dot\Psi_0`, square and
        invertible, over the stacked (theta, eta) coordinates.
    scores : array-like, shape (n, d)
        Stacked influence contributions.
    n : int, optional
        Sample size; taken from ``scores`` otherwise.

    Returns
    -------
    RichResult
        keys: ``avar``, ``se``, ``correlation``, ``jointly`` (True),
        ``operator_invertible``, ``conditions``, ``n``, ``d``,
        ``method``.
    References
    ----------
    Kosorok, Cor. 3.2, p. 47.
    """
    S = np.atleast_2d(np.asarray(scores, dtype=float))
    if S.shape[0] < S.shape[1]:
        S = S.T
    nn = S.shape[0] if n is None else int(n)
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    d = S.shape[1]
    D = np.atleast_2d(np.asarray(psi_dot, dtype=float))
    if D.shape != (d, d):
        raise ValueError(f"psi_dot must be {d} by {d}, got {D.shape}.")
    ok = bool(np.linalg.matrix_rank(D) == d)
    Sigma = S.T @ S / S.shape[0]
    Di = np.linalg.pinv(D)
    avar = Di @ Sigma @ Di.T
    sd = np.sqrt(np.maximum(np.diag(avar), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = avar / np.outer(sd, sd)
    return RichResult(payload={
        "avar": avar, "se": sd / np.sqrt(nn),
        "correlation": np.where(np.isfinite(corr), corr, 0.0),
        "jointly": True, "operator_invertible": ok,
        "conditions": "the no-bias condition (3.6) and stochastic equicontinuity",
        "warning": "theta-hat and eta-hat solve the SAME equation and are correlated; "
                   "combining marginal limits as if independent understates variability",
        "n": int(nn), "d": int(d),
        "method": "Joint convergence (Cor. 3.2); one operator inverse gives both blocks and their dependence"})


def cheatsheet():
    return "ksr073: theta-hat and eta-hat are correlated -- the JOINT law is what you need"
