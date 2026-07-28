# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric efficiency theorem (Kosorok Thm 3.1)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_semipar_efficiency"]


def kosorok_semipar_efficiency(scores, nuisance_scores=None):
    r"""Semiparametric efficiency (Kosorok Thm. 3.1, p. 44):

    .. math:: \sqrt n(\theta_n - \theta)
              \rightsquigarrow -\tilde I_{\theta,\eta}^{-1} Z,

    with :math:`Z` the Gaussian limit of
    :math:`\mathbb G_n \tilde\ell_{\theta,\eta}` and
    :math:`\tilde I = P\tilde\ell\tilde\ell'` the EFFICIENT
    information.

    The efficient score is the ordinary score with its projection
    onto the nuisance tangent space REMOVED, and the efficient
    information is its variance. That projection is the whole cost
    of not knowing the nuisance:

    .. math:: \tilde I \preceq I_{\text{full}},

    always, with equality only when the score for :math:`\theta` is
    already orthogonal to the nuisance directions -- the adaptive
    case. The module computes both informations and their ratio, so
    the price is a number rather than a remark.

    Parameters
    ----------
    scores : array-like, shape (n, p)
        Scores for the parameter of interest.
    nuisance_scores : array-like, shape (n, q), optional
        Scores spanning the nuisance tangent space. Without them the
        problem is parametric and the two informations coincide.

    Returns
    -------
    RichResult
        keys: ``efficient_information``, ``full_information``,
        ``efficient_scores``, ``avar``, ``se``,
        ``information_loss``, ``adaptive``, ``n``, ``p``, ``method``.
    References
    ----------
    Kosorok, Thm. 3.1, p. 44 and Ch. 3.
    """
    from ._kosorok import efficient_information

    S = np.atleast_2d(np.asarray(scores, dtype=float))
    if S.shape[0] < S.shape[1]:
        S = S.T
    n, p = S.shape
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    full = S.T @ S / n
    if nuisance_scores is None:
        eff_info, eff = full, S
    else:
        eff_info, eff = efficient_information(S, nuisance_scores)
    avar = np.linalg.pinv(eff_info)
    loss = float(np.trace(full) - np.trace(eff_info))
    return RichResult(payload={
        "efficient_information": eff_info, "full_information": full,
        "efficient_scores": eff, "avar": avar,
        "se": np.sqrt(np.maximum(np.diag(avar), 0.0) / n),
        "information_loss": loss,
        "adaptive": bool(abs(loss) < 1e-9),
        "ordering": "efficient information <= full information, always",
        "n": int(n), "p": int(p),
        "method": "Semiparametric efficiency (Thm. 3.1); the projection IS the cost of the nuisance"})


def cheatsheet():
    return "ksr072: efficient information <= full, and the gap is exactly what the nuisance costs"
