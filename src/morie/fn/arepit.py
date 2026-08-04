"""Asymptotic relative efficiency from Pitman efficacies."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["are_pitman", "pitman_efficacy"]


def are_pitman(deriv, var, deriv_star, var_star):
    r"""ARE of one test against another via their Pitman efficacies.

    The efficacy of a test whose statistic has asymptotic mean function
    :math:`\mu(\theta)` and variance :math:`\sigma^2` at the null is

    .. math:: c = \frac{[\mu'(\theta_0)]^2}{\sigma^2}

    and the asymptotic relative efficiency of the first test with respect
    to the second is the ratio of their efficacies,

    .. math:: ARE = \frac{c}{c^*}
              = \left(\frac{\mu'}{\mu^{*\prime}}\right)^2
                \frac{\sigma^{*2}}{\sigma^2}.

    This is the efficacy form, and is a different quantity from
    :func:`morie.fn.areN.areratio`, which is the two-variance
    Hodges-Lehmann ratio. Both were previously called ``Areratio``; the R
    arm of this one is ``Arepitman``.

    Returns ``are``, ``check`` (the same ratio computed as c/c*, which
    must equal ``are`` identically), ``efficacy`` and ``efficacy_star``.
    """
    d = float(deriv)
    v = float(var)
    ds = float(deriv_star)
    vs = float(var_star)
    if not (v > 0.0) or not (vs > 0.0):
        raise ValueError("variances must be strictly positive.")
    if ds == 0.0:
        raise ValueError("the reference derivative must be non-zero.")
    e1 = d * d / v
    e2 = ds * ds / vs
    return RichResult(
        payload={
            "are": (d / ds) ** 2 * vs / v,
            "check": e1 / e2,
            "efficacy": e1,
            "efficacy_star": e2,
            "method": "ARE from Pitman efficacies",
        }
    )


def pitman_efficacy(deriv, var):
    """Efficacy c = [mu'(theta_0)]^2 / sigma^2 of a single test."""
    d = float(deriv)
    v = float(var)
    if not (v > 0.0):
        raise ValueError("var must be strictly positive.")
    return RichResult(payload={"efficacy": d * d / v, "deriv": d, "var": v})


arepit = are_pitman
