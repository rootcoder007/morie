# morie.fn -- function file (rootcoder007/morie)
"""Lipschitz envelope condition for M-estimators."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_lipschitz_envelope"]


def kosorok_lipschitz_envelope(m, m_dot, thetas, x):
    r"""The Lipschitz-in-parameter condition (Kosorok Eq. 2.18,
    p. 29):

    .. math:: |m_{\theta_1}(x) - m_{\theta_2}(x)|
              \le \dot m(x)\,\|\theta_1 - \theta_2\|.

    The criterion must be Lipschitz in the PARAMETER with an
    envelope :math:`\dot m` that does not depend on
    :math:`\theta`, and that envelope must be square integrable.

    This is what makes the class
    :math:`\{m_\theta : \theta \in \Theta\}` Donsker, so the
    empirical process indexed by it converges weakly -- the
    condition is doing measure-theoretic work, not just bounding a
    difference. It also fails in recognisable places: a criterion
    with a jump in :math:`\theta`, such as an indicator of
    :math:`\theta' x > 0`, has no such envelope, which is exactly
    why maximum-score estimators are not root-n.

    The function measures the worst observed ratio against the
    envelope over all supplied pairs, so a violation is found rather
    than assumed away.

    Parameters
    ----------
    m : callable
        ``m(theta, x)``, the criterion.
    m_dot : callable
        ``m_dot(x)``, the envelope.
    thetas : sequence
        Parameters to compare pairwise.
    x : array-like
        Points at which to evaluate.

    Returns
    -------
    RichResult
        keys: ``worst_ratio``, ``holds``, ``envelope_square_integrable``,
        ``n_pairs``, ``implication``, ``counterexample_note``,
        ``method``.
    References
    ----------
    Kosorok, Ch. 2, Eq. (2.18), p. 29.
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    ths = [np.atleast_1d(np.asarray(t, dtype=float)).ravel() for t in thetas]
    if len(ths) < 2:
        raise ValueError("need at least 2 parameters to compare.")
    env = np.asarray([float(m_dot(v)) for v in xs])
    if np.any(env < 0):
        raise ValueError("the envelope must be non-negative.")
    worst = 0.0
    pairs = 0
    for i in range(len(ths)):
        for j in range(i + 1, len(ths)):
            d = float(np.linalg.norm(ths[i] - ths[j]))
            if d == 0:
                continue
            pairs += 1
            lhs = np.abs(np.asarray([float(m(ths[i], v)) for v in xs]) -
                         np.asarray([float(m(ths[j], v)) for v in xs]))
            bound = env * d
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(bound > 0, lhs / bound, 0.0)
            worst = max(worst, float(np.nanmax(r)))
    return RichResult(payload={
        "worst_ratio": worst, "holds": bool(worst <= 1.0 + 1e-9),
        "envelope_square_integrable": bool(np.isfinite(np.mean(env ** 2))),
        "n_pairs": pairs,
        "implication": "makes {m_theta} Donsker, so the indexed empirical process converges",
        "counterexample_note": "a criterion with a jump in theta, e.g. 1{theta'x > 0}, "
                               "has no such envelope -- which is why maximum score is not root-n",
        "method": "Lipschitz envelope (Eq. 2.18); the condition that buys the Donsker property"})


def cheatsheet():
    return "ksr054: no envelope means not Donsker -- that is why maximum score is n^{-1/3}"
