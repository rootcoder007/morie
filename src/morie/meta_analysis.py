# SPDX-License-Identifier: AGPL-3.0-or-later
"""Meta-analysis composites: pool a set of estimates, and move between scales.

Parity with rmorie's `morie_meta_pool`, `morie_meta_effect_sizes` and
`morie_meta_convert` (R/ca_crim_native.R).

The arithmetic already lives in `morie.fn._ca_crim` as separate
primitives -- inverse-variance weights, Q, I-squared, the
DerSimonian-Laird tau-squared, and the effect-size conversions. What was
missing is the composite surface that does a whole pooling in one call,
which is how the R side is used, so this module composes them rather
than restating the formulas.

Why both a fixed and a random-effects answer come back together: the
fixed-effect weights assume every study estimates the same quantity, and
I-squared is the evidence about whether that holds. Reporting the pooled
mean without the heterogeneity beside it is the standard way to overstate
a meta-analysis.
"""

from __future__ import annotations

import math

from morie.fn import _ca_crim as _cc

__all__ = ["meta_pool", "meta_effect_sizes", "meta_convert"]


def _num(x, what):
    try:
        v = [float(i) for i in x]
    except TypeError:
        v = [float(x)]
    if not v:
        raise ValueError("`%s` must not be empty" % what)
    for i in v:
        if i != i:
            raise ValueError("`%s` must not contain missing values" % what)
    return v


def meta_pool(ys, ses, z_cv: float = 1.96, groups=None) -> dict:
    """Pool estimates by inverse variance, with the heterogeneity beside it.

    Returns the fixed-effect mean and its interval, Cochran's Q, I-squared,
    the DerSimonian-Laird tau-squared and the random-effects weights that
    follow from it. With `groups`, Q is split into within and between
    components, which is how a subgroup claim is actually tested.
    """
    y = _num(ys, "ys")
    s = _num(ses, "ses")
    if len(y) != len(s):
        raise ValueError("`ys` and `ses` must be the same length")
    if any(i <= 0 for i in s):
        raise ValueError("`ses` must be strictly positive")

    w = [_cc.fixed_effect_weight(i) for i in s]
    # The primitives return records, not bare numbers: mean_effect_size
    # already carries the standard error and z, and q_statistic carries
    # its own df. Take those rather than recomputing them here, so there
    # is one definition of each quantity.
    fe = _cc.mean_effect_size(y, w)
    m = fe["mean"]
    se_m = fe["se"]
    qs = _cc.q_statistic(y, w)
    q = qs["q"]
    df = qs["df"]
    tau2 = _cc.tau2_dersimonian_laird(y, w)
    out = {
        "weights": w,
        "mean": m,
        "se": se_m,
        "z": fe["z"],
        "ci": (m - z_cv * se_m, m + z_cv * se_m),
        "q": q,
        "df": df,
        "i2": _cc.i_squared(q, df),
        "tau2": tau2,
        "weights_random": [_cc.random_effects_weight(i, tau2) for i in s],
    }
    if groups is not None:
        g = list(groups)
        if len(g) != len(y):
            raise ValueError("`groups` must be the same length as `ys`")
        ys_by, ws_by = [], []
        for lab in dict.fromkeys(g):                 # first-seen order
            idx = [i for i, v in enumerate(g) if v == lab]
            ys_by.append([y[i] for i in idx])
            ws_by.append([w[i] for i in idx])
        qwb = _cc.q_within_between(ys_by, ws_by)
        out["q_within"] = qwb["q_within"]
        out["q_between"] = qwb["q_between"]
        out["df_within"] = qwb["df_within"]
        out["df_between"] = qwb["df_between"]
    return out


def meta_effect_sizes(m1=None, m2=None, s1=None, s2=None, n1=None, n2=None,
                      t_value=None, a=None, b=None, c=None, d=None,
                      r=None) -> dict:
    """Effect sizes from whatever a paper actually reported.

    Primary studies report means and SDs, or a t statistic, or a 2x2
    table, or a correlation. Each of those determines an effect size and
    its standard error, and only the ones the arguments support are
    returned -- an absent input yields an absent key rather than a
    fabricated zero.
    """
    out: dict = {}
    if n1 is not None and n2 is not None:
        n1f, n2f = float(n1), float(n2)
        if s1 is not None and m1 is not None and m2 is not None \
                and s2 is not None:
            out["s_pooled"] = _cc.pooled_sd(float(s1), float(s2), n1f, n2f)
            out["d"] = _cc.cohens_d_sample(float(m1), float(m2), float(s1),
                                           float(s2), n1f, n2f)
        out["j"] = _cc.hedges_j(n1f, n2f)
        if "d" in out:
            out["g"] = _cc.hedges_g(out["d"], n1f, n2f)
            out["se_g"] = _cc.se_g(out["g"], n1f, n2f)
        if t_value is not None:
            out["d_from_t"] = _cc.d_from_t(float(t_value), n1f, n2f)
    if a is not None and None not in (b, c, d):
        af, bf, cf, df_ = float(a), float(b), float(c), float(d)
        p1 = af / (af + bf)
        p2 = cf / (cf + df_)
        out["rr"] = _cc.risk_ratio(af, bf, cf, df_)
        out["or"] = _cc.odds_ratio_2x2(af, bf, cf, df_)
        # se_log_rr takes the two risks and the two group sizes, not the
        # four cells
        out["se_ln_rr"] = _cc.se_log_rr(p1, p2, af + bf, cf + df_)
        out["se_ln_or"] = _cc.se_log_or(af, bf, cf, df_)
    if r is not None:
        rf = float(r)
        out["fisher_z"] = _cc.fisher_z(rf)
        if n1 is not None:
            out["se_fisher_z"] = _cc.se_fisher_z(float(n1))
    return out


def meta_convert(ln_or=None, se_ln_or=None, p1=None, p2=None, n1=None,
                 n2=None, d=None, se_d=None, rr=None, or_value=None,
                 r=None, se_r=None) -> dict:
    """Move an effect between the scales a synthesis has to mix.

    A review rarely gets one scale. Log odds ratios, standardised mean
    differences, probit differences and correlations all convert, and the
    conversion constants are the ones the literature uses: the logistic
    SD (pi/sqrt(3)) for the logit route and Cox's 1.65.
    """
    sd_logistic = math.sqrt(math.pi ** 2 / 3.0)
    out: dict = {"sd_logistic": sd_logistic}
    if ln_or is not None:
        out["d_logit"] = _cc.d_from_log_or(float(ln_or), method="logit")
        out["d_cox"] = _cc.d_from_log_or(float(ln_or), method="cox")
    if se_ln_or is not None:
        out["se_d_logit"] = math.sqrt(float(se_ln_or) ** 2 / sd_logistic ** 2)
        out["se_d_cox"] = math.sqrt(float(se_ln_or) ** 2 / 1.65 ** 2)
    if p1 is not None and p2 is not None:
        out["d_probit"] = _cc.d_probit(float(p1), float(p2))
        if n1 is not None and n2 is not None:
            out["se_d_probit"] = _cc.se_d_probit(float(p1), float(p2),
                                                 float(n1), float(n2))
    if d is not None:
        dv = float(d)
        out["ln_or_logit"] = _cc.log_or_from_d(dv, method="logit")
        out["ln_or_cox"] = _cc.log_or_from_d(dv, method="cox")
        out["r_from_d"] = _cc.r_from_d(dv, float(n1), float(n2)) \
            if (n1 is not None and n2 is not None) else _cc.r_from_d(dv)
        if se_d is not None:
            out["se_ln_or_logit"] = _cc.se_log_or_from_se_d(float(se_d),
                                                            method="logit")
    if rr is not None and p2 is not None:
        out["or_from_rr"] = _cc.or_from_rr(float(rr), float(p2))
    if or_value is not None and p2 is not None:
        out["rr_from_or"] = _cc.rr_from_or(float(or_value), float(p2))
    if r is not None:
        out["fisher_z"] = _cc.fisher_z(float(r))
        out["d_from_r"] = _cc.d_from_r_pointbiserial(float(r))
        if se_r is not None:
            out["se_d_from_se_r"] = _cc.se_d_from_se_r(float(r),
                                                       float(se_r))
    return out
