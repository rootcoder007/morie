# morie.fn -- function file (rootcoder007/morie)
"""ATT inverse probability of treatment weights."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_iptw_attweights"]


def causal_iptw_attweights(treat, ps):
    r"""ATT inverse-probability-of-treatment weights.

    For the average treatment effect on the treated, treated units keep
    weight 1 and controls are reweighted to look like the treated
    population:

    .. math:: w_i = T_i + (1 - T_i)\,\frac{e(x_i)}{1 - e(x_i)}.

    Also reports the Kish effective sample size of each reweighted
    group, :math:`\mathrm{ESS} = (\sum w)^2 / \sum w^2`, the standard
    diagnostic for how much information the weighting discards.

    Parameters
    ----------
    treat : array-like of {0, 1}, shape (n,)
        Treatment indicator.
    ps : array-like, shape (n,)
        Estimated propensity scores; must lie strictly in (0, 1) for
        controls.

    Returns
    -------
    RichResult
        keys: ``weights`` (n,), ``ess_control``, ``ess_treated``,
        ``n``, ``method``.

    References
    ----------
    Hernan, M. A. & Robins, J. M. (2020). *Causal Inference: What If*.
    Chapman & Hall/CRC. Ch. 15 (propensity scores; ATT weighting).
    """
    treat = np.asarray(treat, dtype=float).ravel()
    ps = np.asarray(ps, dtype=float).ravel()
    if treat.size != ps.size:
        raise ValueError(f"treat and ps must have equal length, got {treat.size} and {ps.size}.")
    if not np.all(np.isin(treat, (0.0, 1.0))):
        raise ValueError("treat must be binary 0/1.")
    ctrl = treat == 0
    if np.any((ps[ctrl] <= 0) | (ps[ctrl] >= 1)):
        raise ValueError("control propensity scores must lie strictly in (0, 1).")

    w = np.where(treat == 1, 1.0, ps / (1.0 - ps))

    def _ess(v):
        s = v.sum()
        return float(s * s / (v**2).sum()) if v.size else 0.0

    return RichResult(
        payload={
            "weights": w,
            "ess_control": _ess(w[ctrl]),
            "ess_treated": _ess(w[~ctrl]),
            "n": int(treat.size),
            "method": "ATT inverse probability of treatment weights",
        }
    )


def cheatsheet():
    return "causipsw: ATT IPT weights (controls e/(1-e), treated 1) + Kish ESS"
