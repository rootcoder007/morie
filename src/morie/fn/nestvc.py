"""Variance components of a two-stage nested design."""

import math

from ._richresult import RichResult

__all__ = ["nested_variance_components"]


def nested_variance_components(y, group, unit):
    r"""Variance components and induced correlation of a nested design.

    The model of Schabenberger & Gotway (2005, Example 1.1, p. 2) is

    .. math::

        Y_{ijk} = \mu + \tau_i + e_{ij} + \epsilon_{ijk},

    with fixed groups :math:`\tau_i`, random units :math:`e_{ij}` within
    groups and sub-sampling errors :math:`\epsilon_{ijk}`. Observations on
    the same unit are correlated because they share :math:`e_{ij}`:
    :math:`\mathrm{Cov}[Y_{ijk}, Y_{ijk'}] = \mathrm{Var}[e_{ij}]`, and
    observations on different units are uncorrelated.

    The components are the ANOVA (method-of-moments) estimators from the
    nested analysis of variance, which for balanced data are also the
    REML estimates whenever they are positive:
    :math:`\hat\sigma^2_\epsilon = MS_E`,
    :math:`\hat\sigma^2_e = (MS_{U(G)} - MS_E)/n`.

    Parameters
    ----------
    y : sequence of float
        Responses.
    group : sequence
        Fixed-effect group label of each response (e.g. facility).
    unit : sequence
        Random experimental-unit label (e.g. batch), nested in ``group``.
        Every unit must have the same number ``n`` of sub-samples.

    Returns
    -------
    RichResult
        ``ms_unit``, ``df_unit``, ``ms_error``, ``df_error``,
        ``sigma2_unit`` (ANOVA estimate, possibly negative),
        ``sigma2_error``, ``within_unit_cov`` (``max(sigma2_unit, 0)``),
        ``icc`` (the induced correlation between two sub-samples of one
        unit), ``n_sub``, ``n_units``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, Example 1.1, p. 2.
    Searle, S. R., Casella, G. & McCulloch, C. E. (1992). Variance
    Components. Wiley.
    """
    y = [float(v) for v in y]
    group = list(group)
    unit = list(unit)
    if not (len(y) == len(group) == len(unit)) or not y:
        raise ValueError("`y`, `group` and `unit` must be non-empty and the same length")
    cells = {}
    for v, g, u in zip(y, group, unit):
        cells.setdefault((g, u), []).append(v)
    owner = {}
    for g, u in cells:
        if owner.setdefault(u, g) != g:
            raise ValueError(f"unit {u!r} appears in more than one group; units must be nested")
    sizes = {len(v) for v in cells.values()}
    if len(sizes) != 1:
        raise ValueError("every unit must have the same number of sub-samples")
    n = sizes.pop()
    if n < 2:
        raise ValueError("each unit needs at least two sub-samples")
    groups = {}
    for (g, _u), v in cells.items():
        groups.setdefault(g, []).append(sum(v) / n)
    ss_unit = 0.0
    df_unit = 0
    for means in groups.values():
        gm = sum(means) / len(means)
        ss_unit += n * sum((m - gm) ** 2 for m in means)
        df_unit += len(means) - 1
    ss_err = sum(sum((x - sum(v) / n) ** 2 for x in v) for v in cells.values())
    df_err = len(y) - len(cells)
    if df_unit < 1:
        raise ValueError("at least one group needs two or more units")
    ms_unit = ss_unit / df_unit
    ms_err = ss_err / df_err
    s2u = (ms_unit - ms_err) / n
    cov = max(s2u, 0.0)
    icc = cov / (cov + ms_err) if cov + ms_err > 0 else math.nan
    return RichResult(
        title="Nested design variance components (Example 1.1)",
        summary_lines=[("sigma2_unit", s2u), ("sigma2_error", ms_err), ("icc", icc)],
        payload={
            "ms_unit": ms_unit,
            "df_unit": df_unit,
            "ms_error": ms_err,
            "df_error": df_err,
            "sigma2_unit": s2u,
            "sigma2_error": ms_err,
            "within_unit_cov": cov,
            "icc": icc,
            "n_sub": n,
            "n_units": len(cells),
        },
    )


def cheatsheet():
    return "nestvc: nested-design variance components; Cov within a unit = Var[e_ij]"
