# morie.fn -- function file (rootcoder007/morie)
"""Theorem 4.4: asymptotic normality of boundary-free MRL estimators."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["fauzi_theorem_4_4", "fauzi_thm4_4_mrl_normality"]


def fauzi_theorem_4_4(mrl_hat, mrl_true, variance):
    r"""Theorem 4.4 (Fauzi): asymptotic normality of the
    boundary-free mean-residual-life estimators,

    .. math:: \frac{\tilde m_{X,i}(t) - m_X(t)}
                    {\sqrt{\mathrm{Var}[\tilde m_{X,i}(t)]}}
              \;\to_D\; N(0,1),
              \qquad i = 1, 2 .

    The proof runs through Lyapunov's condition, and what makes it
    work is that :math:`V(\cdot)` is BOUNDED between 0 and 1, so all
    its moments exist automatically -- the book checks
    :math:`E|V - EV|^{2+\delta} \le 2^{2+\delta} < \infty` and the
    condition follows. A kernel-type estimator whose summands were
    unbounded would need a genuine moment assumption instead.

    Both estimators satisfy it, so a normal confidence interval is
    licensed at any interior point AND at the boundary, which is not
    true of the naive estimator.

    Parameters
    ----------
    mrl_hat : array-like
        The estimates.
    mrl_true : array-like
        The target, for the standardised statistic.
    variance : array-like
        The Theorem 4.3 variances.

    Returns
    -------
    RichResult
        keys: ``z``, ``p_two_sided``, ``holds_for``,
        ``why_lyapunov_works``, ``valid_at_boundary`` (True),
        ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Theorem 4.4. Transcribed from the PDF.
    """
    from scipy import stats

    mh = np.atleast_1d(np.asarray(mrl_hat, dtype=float)).ravel()
    mt = np.atleast_1d(np.asarray(mrl_true, dtype=float)).ravel()
    v = np.atleast_1d(np.asarray(variance, dtype=float)).ravel()
    if not (mh.size == mt.size == v.size):
        raise ValueError("all three arguments must have the same length.")
    if np.any(v < 0):
        raise ValueError("variances must be non-negative.")
    sd = np.sqrt(np.maximum(v, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (mh - mt) / np.maximum(sd, 1e-300), np.nan)
    return RichResult(payload={
        "z": z, "p_two_sided": 2 * stats.norm.sf(np.abs(z)),
        "holds_for": "both m_tilde_{X,1} and m_tilde_{X,2}",
        "why_lyapunov_works": "V is bounded in [0, 1], so every moment exists "
                              "automatically and the Lyapunov condition needs "
                              "no extra assumption",
        "valid_at_boundary": True,
        "method": "Theorem 4.4: standardised boundary-free MRL estimators are asymptotically N(0, 1)"})


def cheatsheet():
    return "fzt44: V bounded in [0,1] is what makes Lyapunov automatic -- no moment assumption needed"


#: Catalogue alias for :func:`fauzi_theorem_4_4`.
fauzi_thm4_4_mrl_normality = fauzi_theorem_4_4
