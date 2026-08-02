# morie.fn -- function file (rootcoder007/morie)
"""No-bias condition for semiparametric efficiency."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_no_bias", "kosorok_ch3_z_estimator_no_bias"]


def kosorok_no_bias(eff_scores, theta_seq, theta0, n_seq):
    r"""The no-bias condition (Kosorok Eq. 3.6, p. 44):

    .. math:: P_{\theta_n,\eta}\,\tilde\ell_{\theta_n,\eta_n}
              = o_P\big(n^{-1/2} + \|\theta_n - \theta\|\big).

    The efficient score, evaluated at the ESTIMATED nuisance but
    averaged under the true law, must be negligible. It is the
    condition that lets an infinite-dimensional nuisance be
    estimated at a rate slower than root-n without contaminating
    :math:`\hat\theta` -- which is the entire reason semiparametric
    estimation is possible at all.

    It works because the efficient score is ORTHOGONAL to the
    nuisance tangent space: a first-order error in
    :math:`\hat\eta` moves the score only to second order. Where
    orthogonality fails the condition fails with it, and
    :math:`\hat\theta` inherits the nuisance's slower rate.

    The tolerance is :math:`n^{-1/2} + \|\theta_n - \theta\|`, not
    :math:`n^{-1/2}` alone -- the slack grows when
    :math:`\theta_n` is far from the truth, which is what makes the
    condition usable before consistency has been established.

    Parameters
    ----------
    eff_scores : callable
        ``eff_scores(theta, n)`` returning the efficient scores at
        the estimated nuisance, averaged under the true law.
    theta_seq : sequence
        Candidate parameters, one per sample size.
    theta0 : array-like
        The truth.
    n_seq : sequence of int
        Sample sizes.

    Returns
    -------
    RichResult
        keys: ``n``, ``bias``, ``tolerance``, ``ratio``, ``holds``,
        ``why_it_works``, ``method``.
    References
    ----------
    Kosorok, Ch. 3, Eq. (3.6), p. 44.
    """
    ns = np.atleast_1d(np.asarray(n_seq, dtype=float)).ravel()
    ths = list(theta_seq)
    if len(ths) != ns.size:
        raise ValueError(f"theta_seq has {len(ths)} entries for {ns.size} sizes.")
    if np.any(ns < 1):
        raise ValueError("sample sizes must be at least 1.")
    t0 = np.atleast_1d(np.asarray(theta0, dtype=float)).ravel()
    bias, tol = [], []
    for th, nn in zip(ths, ns):
        b = float(np.abs(np.atleast_1d(eff_scores(th, nn))).max())
        d = float(np.abs(np.atleast_1d(np.asarray(th, dtype=float)) - t0).max())
        bias.append(b)
        tol.append(nn ** -0.5 + d)
    bias = np.array(bias); tol = np.array(tol)
    ratio = bias / tol
    return RichResult(payload={
        "n": ns, "bias": bias, "tolerance": tol, "ratio": ratio,
        "holds": bool(ratio[-1] < ratio[0]),
        "why_it_works": "the efficient score is orthogonal to the nuisance tangent "
                        "space, so a first-order nuisance error moves it only to second order",
        "tolerance_note": "n^{-1/2} + ||theta_n - theta||, not n^{-1/2} alone: the "
                          "slack grows when theta_n is far from the truth",
        "method": "No-bias condition (Eq. 3.6); why a slow nuisance rate does not contaminate theta"})


def cheatsheet():
    return "ksr066: orthogonality is what lets a slower-than-root-n nuisance be harmless"


#: Catalogue alias for :func:`kosorok_no_bias`.
kosorok_ch3_z_estimator_no_bias = kosorok_no_bias
