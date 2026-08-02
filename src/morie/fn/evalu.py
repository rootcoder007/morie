# morie.fn -- function file (rootcoder007/morie)
"""E-value for unmeasured confounding (VanderWeele-Ding)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["evalue"]


def evalue(RR, ci_lower=None, ci_upper=None, rare_outcome=True):
    r"""E-value: minimum confounding strength that explains an effect away.

    .. math:: E = RR^* + \sqrt{RR^*(RR^* - 1)}, \qquad
              RR^* = \max(RR, 1/RR)

    The E-value is the minimum strength of association, on the risk
    ratio scale, that an unmeasured confounder would need with BOTH
    the treatment and the outcome to fully explain away the observed
    association (VanderWeele & Ding 2017). RR = 1 gives E = 1 (no
    confounding needed); RR = 2 gives :math:`2 + \sqrt{2} = 3.41`.

    When a confidence interval is supplied, the E-value for the limit
    CLOSER to the null is also reported -- 1 exactly if the interval
    crosses 1, since then no unmeasured confounding at all is needed
    to explain away significance.

    This replaces a placeholder that returned the mean of the RR
    argument.

    Parameters
    ----------
    RR : float
        Observed risk ratio (or odds/hazard ratio for a rare outcome,
        where they approximate the RR).
    ci_lower, ci_upper : float, optional
        Confidence limits on the same scale.
    rare_outcome : bool, default True
        Set False to record in the payload that an OR/HR input for a
        common outcome only approximates the RR (the paper suggests
        the square-root transform in that case; not applied silently).

    Returns
    -------
    RichResult
        keys: ``evalue`` / ``estimate``, ``evalue_ci`` (for the limit
        closer to the null; None without a CI), ``rr``, ``rr_star``,
        ``method``.

    References
    ----------
    VanderWeele, T. J. & Ding, P. (2017). Sensitivity analysis in
    observational research: introducing the E-value. *Annals of
    Internal Medicine*, 167(4), 268-274. doi:10.7326/M16-2607.
    """
    rr = float(RR)
    if rr <= 0:
        raise ValueError(f"RR must be positive, got {rr}.")

    def _e(r):
        r_star = max(r, 1.0 / r)
        return r_star + np.sqrt(r_star * (r_star - 1.0))

    e_point = float(_e(rr))
    e_ci = None
    if ci_lower is not None or ci_upper is not None:
        if ci_lower is None or ci_upper is None:
            raise ValueError("Supply both confidence limits or neither.")
        lo, hi = float(ci_lower), float(ci_upper)
        if not (0 < lo <= hi):
            raise ValueError(f"Need 0 < ci_lower <= ci_upper, got ({lo}, {hi}).")
        if lo <= 1.0 <= hi:
            e_ci = 1.0  # the interval crosses the null already
        else:
            # The limit closer to the null.
            limit = lo if lo > 1.0 else hi
            e_ci = float(_e(limit))
    return RichResult(
        payload={
            "evalue": e_point,
            "estimate": e_point,
            "evalue_ci": e_ci,
            "rr": rr,
            "rr_star": float(max(rr, 1.0 / rr)),
            "rare_outcome": bool(rare_outcome),
            "method": "E-value (VanderWeele & Ding 2017)",
        }
    )


def cheatsheet():
    return "evalu: E-value = RR* + sqrt(RR*(RR*-1)) (VanderWeele & Ding 2017)"
