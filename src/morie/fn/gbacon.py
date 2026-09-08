# morie.fn -- function file (rootcoder007/morie)
"""Goodman-Bacon decomposition of the two-way fixed-effects estimator."""

from . import _array_core as np

from ._did import as_panel, first_treatment, twfe_beta
from ._richresult import RichResult

__all__ = ["goodman_bacon_decomp"]


def _cell_mean(Y, rows, cols):
    if rows.size == 0 or cols.size == 0:
        return np.nan
    return float(Y[np.ix_(rows, cols)].mean())


def _did_2x2(Y, tr, ct, pre, post):
    return (_cell_mean(Y, tr, post) - _cell_mean(Y, tr, pre)) - (
        _cell_mean(Y, ct, post) - _cell_mean(Y, ct, pre)
    )


def goodman_bacon_decomp(y, D, unit, time):
    r"""Decompose the TWFE coefficient into its 2x2 building blocks.

    Goodman-Bacon (2021) Theorem 1: with staggered adoption, the
    two-way fixed-effects coefficient is exactly

    .. math:: \hat\beta^{DD} = \sum_{k \neq U} s_{kU}\,
              \hat\beta^{2\times2}_{kU}
              + \sum_{k \neq U}\sum_{\ell > k}
              \Big[ s^{k}_{k\ell}\,\hat\beta^{2\times2,k}_{k\ell}
              + s^{\ell}_{k\ell}\,\hat\beta^{2\times2,\ell}_{k\ell}
              \Big],

    a weighted average of every 2x2 DiD in the panel. Three kinds of
    comparison appear: each timing cohort against the never-treated,
    an earlier cohort against a later one before the later adopts
    (a legitimate comparison), and -- the problem -- a LATER cohort
    against an EARLIER one after the earlier has already been
    treated. That last type uses treated units as controls, so it
    differences out the earlier cohort's own evolving effect. When
    effects grow over time it enters with the wrong sign, which is
    how a TWFE coefficient can be negative while every unit's effect
    is positive.

    The weights depend only on group sizes and treatment TIMING, not
    on the outcome: a cohort treated near the middle of the panel
    gets the most weight, because that is where treatment has the
    most within-variance. This is why TWFE quietly overweights
    mid-panel adopters.

    The decomposition is an identity, so it is verified rather than
    asserted: ``identity_residual`` is
    :math:`\hat\beta^{DD} - \sum_j s_j \hat\beta_j` and
    ``weight_sum`` is :math:`\sum_j s_j`, which must be 1.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome in long format.
    D : array-like, shape (n,)
        Absorbing binary treatment.
    unit, time : array-like, shape (n,)
        Identifiers; the panel must be balanced.

    Returns
    -------
    RichResult
        ``estimate`` (the TWFE coefficient), ``components`` (a list of
        dicts with ``type``, ``treated``, ``control``, ``weight``,
        ``beta``), ``weight_sum``, ``identity_residual``,
        ``forbidden_weight`` (weight on already-treated-control
        comparisons), ``weight_by_type``, ``n_components``.

    Notes
    -----
    The always-treated group is dropped, as in Goodman-Bacon: a unit
    treated in every period contributes no within variation and is
    absorbed by its own fixed effect.

    References
    ----------
    Goodman-Bacon (2021), *Journal of Econometrics* 225:254-277,
    Theorem 1 and Section 3.

    Examples
    --------
    >>> import numpy as np
    >>> unit = np.repeat(np.arange(6), 6)
    >>> time = np.tile(np.arange(6), 6)
    >>> g = np.repeat([2.0, 2.0, 4.0, 4.0, np.inf, np.inf], 6)
    >>> D = (time >= g).astype(float)
    >>> y = unit * 0.1 + time * 0.2 + 1.0 * D
    >>> out = goodman_bacon_decomp(y, D, unit, time)
    >>> round(out["weight_sum"], 10), round(out["identity_residual"], 10)
    (1.0, 0.0)
    """
    Y, units, periods = as_panel(y, unit, time)
    g, Dm, _, _ = first_treatment(D, unit, time, units, periods)
    beta, _, Dt, denom = twfe_beta(Y, Dm)

    n_units, T = Y.shape
    always = np.isfinite(g) & (g == 0)
    if always.any():
        keep = ~always
        if keep.sum() < 2:
            raise ValueError(
                "after dropping always-treated units fewer than 2 remain; "
                "a unit treated in every period carries no within variation."
            )
        Y, g, Dm = Y[keep], g[keep], Dm[keep]
        units = units[keep]
        n_units = Y.shape[0]
        beta, _, Dt, denom = twfe_beta(Y, Dm)

    never = np.nonzero(~np.isfinite(g))[0]
    cohorts = np.unique(g[np.isfinite(g)])
    n_all = float(n_units)
    Dbar = {float(c): float((T - c) / T) for c in cohorts}
    rows = {float(c): np.nonzero(g == c)[0] for c in cohorts}
    share = {float(c): rows[float(c)].size / n_all for c in cohorts}
    share_U = never.size / n_all

    comps = []
    var_D = denom / (n_units * T)

    for c in cohorts:
        c = float(c)
        k = int(c)
        if never.size:
            nk, nu = share[c], share_U
            nbar = nk / (nk + nu)
            w = (nk + nu) ** 2 * nbar * (1 - nbar) * Dbar[c] * (1 - Dbar[c])
            b = _did_2x2(Y, rows[c], never, np.arange(k), np.arange(k, T))
            comps.append(
                {
                    "type": "treated vs never-treated",
                    "treated": c,
                    "control": np.inf,
                    "weight": w / var_D,
                    "beta": b,
                    "forbidden": False,
                }
            )

    for i, c in enumerate(cohorts):
        for l in cohorts[i + 1:]:
            c, l = float(c), float(l)
            k, m = int(c), int(l)
            nk, nl = share[c], share[l]
            nbar = nk / (nk + nl)
            Dk, Dl = Dbar[c], Dbar[l]
            # earlier cohort treated, later cohort still untreated
            w_k = (
                ((nk + nl) * (1 - Dl)) ** 2
                * nbar
                * (1 - nbar)
                * ((Dk - Dl) / (1 - Dl))
                * ((1 - Dk) / (1 - Dl))
            )
            b_k = _did_2x2(Y, rows[c], rows[l], np.arange(k), np.arange(k, m))
            comps.append(
                {
                    "type": "early vs late (before late adopts)",
                    "treated": c,
                    "control": l,
                    "weight": w_k / var_D,
                    "beta": b_k,
                    "forbidden": False,
                }
            )
            # later cohort treated, EARLIER cohort used as control while
            # already treated -- the comparison the literature forbids
            w_l = (
                ((nk + nl) * Dk) ** 2
                * nbar
                * (1 - nbar)
                * (Dl / Dk)
                * ((Dk - Dl) / Dk)
            )
            b_l = _did_2x2(Y, rows[l], rows[c], np.arange(k, m),
                           np.arange(m, T))
            comps.append(
                {
                    "type": "late vs early (early already treated)",
                    "treated": l,
                    "control": c,
                    "weight": w_l / var_D,
                    "beta": b_l,
                    "forbidden": True,
                }
            )

    w = np.array([c["weight"] for c in comps])
    b = np.array([c["beta"] for c in comps])
    wsum = float(w.sum())
    recomposed = float(np.sum(w * b))
    by_type = {}
    for comp in comps:
        by_type[comp["type"]] = by_type.get(comp["type"], 0.0) + comp["weight"]

    return RichResult(
        payload={
            "estimate": beta,
            "components": comps,
            "weights": w,
            "betas": b,
            "weight_sum": wsum,
            "recomposed": recomposed,
            "identity_residual": beta - recomposed,
            "weight_by_type": by_type,
            "forbidden_weight": float(
                sum(c["weight"] for c in comps if c["forbidden"])
            ),
            "n_components": len(comps),
            "n_units": int(n_units),
            "n_periods": int(T),
            "dropped_always_treated": int(always.sum()),
            "identity_note": (
                "weight_sum is 1 and identity_residual is 0 by Theorem 1; "
                "they are computed rather than assumed so a violated "
                "assumption surfaces as a number"
            ),
            "reading": (
                "forbidden_weight is the share of the TWFE coefficient coming "
                "from comparisons that use already-treated units as controls; "
                "a large share with heterogeneous dynamic effects is how TWFE "
                "gets the sign wrong"
            ),
            "method": "Goodman-Bacon (2021) decomposition of the TWFE estimator",
        }
    )


def cheatsheet():
    return (
        "gbacon: decompose a TWFE DiD into its 2x2 comparisons and weights; "
        "forbidden_weight is the already-treated-control share"
    )
