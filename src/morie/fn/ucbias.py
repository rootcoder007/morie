# morie.fn -- function file (rootcoder007/morie)
"""Bounding factor for an unmeasured confounder (Ding-VanderWeele)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["unmeasured_conf_bias"]


def unmeasured_conf_bias(RR_UD, RR_UY, RR_obs=None):
    r"""Ding-VanderWeele bounding factor for unmeasured confounding.

    .. math:: B = \frac{RR_{UD} \cdot RR_{UY}}{RR_{UD} + RR_{UY} - 1}

    where :math:`RR_{UD}` is the confounder-treatment association and
    :math:`RR_{UY}` the confounder-outcome association, both on the
    risk ratio scale. The true causal risk ratio is bounded below by
    :math:`RR_{obs} / B`: a confounder of the given strengths cannot
    move the estimate by more than the factor B. The E-value is the
    solution of B = RR_obs on the diagonal
    :math:`RR_{UD} = RR_{UY}`, which is how the two modules connect.

    This replaces a placeholder that averaged its first argument.

    Parameters
    ----------
    RR_UD, RR_UY : float
        Confounder-treatment and confounder-outcome risk ratios,
        both >= 1 (take reciprocals first for protective directions).
    RR_obs : float, optional
        Observed risk ratio; when given, the confounding-adjusted
        lower bound ``RR_obs / B`` is included.

    Returns
    -------
    RichResult
        keys: ``bias_factor`` / ``estimate``, ``rr_ud``, ``rr_uy``,
        ``rr_obs``, ``rr_bound`` (None without RR_obs),
        ``explains_away`` (None without RR_obs), ``method``.

    References
    ----------
    Ding, P. & VanderWeele, T. J. (2016). Sensitivity analysis without
    assumptions. *Epidemiology*, 27(3), 368-377.
    VanderWeele, T. J. & Ding, P. (2017). Sensitivity analysis in
    observational research: introducing the E-value. *Annals of
    Internal Medicine*, 167(4), 268-274 (the E-value as the diagonal
    case).
    """
    a = float(RR_UD)
    b = float(RR_UY)
    if a < 1.0 or b < 1.0:
        raise ValueError(
            f"RR_UD and RR_UY must be >= 1 (reciprocate protective ratios first); got ({a}, {b})."
        )
    B = a * b / (a + b - 1.0)
    rr_bound = explains = None
    if RR_obs is not None:
        r = float(RR_obs)
        if r <= 0:
            raise ValueError(f"RR_obs must be positive, got {r}.")
        r_star = max(r, 1.0 / r)
        rr_bound = r_star / B
        # At associations exactly equal to the E-value, B equals RR up to
        # rounding; the boundary counts as explaining away.
        explains = bool(B >= r_star * (1.0 - 1e-9))
    return RichResult(
        payload={
            "bias_factor": float(B),
            "estimate": float(B),
            "rr_ud": a,
            "rr_uy": b,
            "rr_obs": None if RR_obs is None else float(RR_obs),
            "rr_bound": None if rr_bound is None else float(rr_bound),
            "explains_away": explains,
            "method": "Ding-VanderWeele bounding factor B = RR_UD RR_UY / (RR_UD + RR_UY - 1)",
        }
    )


def cheatsheet():
    return "ucbias: Ding-VanderWeele bounding factor for unmeasured confounding"
