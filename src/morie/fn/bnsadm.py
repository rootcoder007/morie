# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What no estimator, and no treatment rule, can beat.

Two bounds that constrain every admissible procedure for the average
treatment effect, computed from the same fitted nuisances:

* the semiparametric efficiency bound for the ATE -- Hahn (1998),
  *Econometrica* 66(2):315-331 -- which no regular asymptotically
  linear estimator can undercut;
* the local asymptotic minimax regret bound for treatment choice --
  Hirano and Porter (2009), *Econometrica* 77(5):1683-1701 -- which no
  statistical treatment rule can undercut, and which the plug-in rule
  on an efficient estimator attains.
"""

import math

from . import _array_core as np

from ._did import add_intercept, logit_fit, logit_predict
from ._richresult import RichResult

__all__ = ["bound_admissible_estimators", "minimax_regret_constant"]

_METHOD = "Semiparametric efficiency and minimax regret bounds for the ATE"


def _phi(t):
    return math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)


def _Phi(t):
    return 0.5 * math.erfc(-t / math.sqrt(2.0))


def minimax_regret_constant(tol=1e-14):
    r"""The constant in the Hirano-Porter treatment-choice bound.

    Under local asymptotics the welfare regret of a rule that treats
    when :math:`\hat\tau > 0`, against a truth :math:`\tau = h/\sqrt n`
    with :math:`\hat\tau \sim N(h/n^{1/2}, V/n)`, is
    :math:`|h|\,\Phi(-|h|/\sqrt V)/\sqrt n`. The worst case over the
    local parameter :math:`h` therefore scales as
    :math:`c\sqrt{V/n}` with

    .. math::
        c = \max_{t \ge 0}\ t\,\Phi(-t),

    whose stationary condition is :math:`\Phi(-t) = t\,\phi(t)`.

    The constant is solved here rather than quoted, because the value
    usually quoted to two figures (0.17) is not accurate enough to
    check an attainment claim against simulation.

    Returns
    -------
    dict with ``t_star``, ``constant``, ``stationarity_residual``.
    """
    lo, hi = 0.0, 5.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        # decreasing in t: positive below t*, negative above
        if _Phi(-mid) - mid * _phi(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    t = 0.5 * (lo + hi)
    return {
        "t_star": t,
        "constant": t * _Phi(-t),
        "stationarity_residual": abs(_Phi(-t) - t * _phi(t)),
    }


def _arm_fit(Xd, y, mask, family):
    """Fit the outcome regression within one treatment arm."""
    if mask.sum() < Xd.shape[1] + 1:
        raise ValueError(
            f"only {int(mask.sum())} observations in one treatment arm for "
            f"{Xd.shape[1]} design columns; the outcome model is not "
            "identified."
        )
    if family == "binomial":
        beta, _ = logit_fit(Xd[mask], y[mask])
        mu = logit_predict(Xd, beta)
        # Bernoulli variance is a function of the mean, not a free
        # parameter, so there is nothing further to estimate
        return mu, mu * (1.0 - mu)
    beta, *_ = np.linalg.lstsq(Xd[mask], y[mask], rcond=None)
    mu = Xd @ beta
    r = y[mask] - Xd[mask] @ beta
    dof = max(int(mask.sum()) - Xd.shape[1], 1)
    s2 = float(r @ r) / dof
    return mu, np.full(y.shape, s2)


def bound_admissible_estimators(y, D, X, family="gaussian", trim=0.01,
                                alpha=0.05):
    r"""Efficiency and minimax bounds, with two estimators measured
    against them.

    Hahn's bound for the ATE under unconfoundedness is

    .. math::
        V_{\mathrm{eff}} = \mathbb{E}\left[
        \frac{\sigma_1^2(X)}{e(X)} + \frac{\sigma_0^2(X)}{1 - e(X)}
        + (\tau(X) - \tau)^2 \right].

    Three things follow from the shape of it, and the payload reports
    each so they can be checked rather than taken on faith.

    The first term blows up as the propensity approaches zero or one.
    That is not a numerical artefact to be clipped away: it is the
    bound saying that regions of covariate space with no comparison
    units carry no information about the contrast there, and no
    estimator, however clever, can manufacture it.

    The third term is the variance of the *conditional* effect. It does
    not shrink with better nuisance estimation and does not vanish when
    the propensity is known. Effect heterogeneity has an irreducible
    price for anyone reporting an average.

    Knowing the true propensity score does not lower the bound. Hahn's
    result is that the bound is the same whether ``e`` is known or
    estimated -- and, more sharply, that using an *estimated* score can
    yield a smaller variance than plugging in the true one. This is the
    single most counter-intuitive fact in the area, and the reason
    inverse-probability weighting with a known design is not
    automatically the right thing to do.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like, shape (n,)
        Binary treatment indicator.
    X : array-like, shape (n, p)
        Covariates. An intercept is added.
    family : {"gaussian", "binomial"}
        Outcome family for the arm regressions.
    trim : float
        Propensity scores are confined to ``[trim, 1 - trim]`` before
        the bound is evaluated. ``trim_binding`` records whether this
        actually bit, because a bound computed after heavy trimming is
        a bound for a different estimand.
    alpha : float
        Level for the reported interval on the ATE.

    Returns
    -------
    RichResult
        ``efficiency_bound``, ``se_bound``, ``estimate`` (the AIPW
        ATE), ``var_aipw``, ``var_ipw``, ``aipw_efficiency_ratio``,
        ``ipw_efficiency_ratio``, ``minimax_regret_bound``,
        ``overlap_term``, ``heterogeneity_term``, ``tau_x``.

    References
    ----------
    Hahn J (1998) *Econometrica* 66(2):315-331, doi:10.2307/2998560.
    Hirano K, Porter JR (2009) *Econometrica* 77(5):1683-1701,
    doi:10.3982/ECTA6630.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(4000, 2))
    >>> e = 1 / (1 + np.exp(-0.6 * X[:, 0]))
    >>> D = (rng.random(4000) < e).astype(float)
    >>> y = 2.0 + X @ [1.0, -0.5] + D * 1.0 + rng.normal(size=4000)
    >>> out = bound_admissible_estimators(y, D, X)
    >>> bool(abs(out["estimate"] - 1.0) < 0.1)
    True
    >>> bool(out["ipw_efficiency_ratio"] >= 1.0)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    d = np.asarray(D, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    if Xa.shape[0] != yv.size and Xa.shape[1] == yv.size:
        Xa = Xa.T
    n = yv.size
    if d.size != n or Xa.shape[0] != n:
        raise ValueError(
            f"y, D and X must agree in length; got {n}, {d.size} and "
            f"{Xa.shape[0]}."
        )
    if n < 10:
        raise ValueError(f"need at least 10 observations; got {n}.")
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise ValueError("D must be binary 0/1.")
    if family not in ("gaussian", "binomial"):
        raise ValueError('family must be "gaussian" or "binomial".')
    if not 0 <= trim < 0.5:
        raise ValueError(f"trim must lie in [0, 0.5); got {trim}.")
    if family == "binomial" and not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError('family="binomial" requires a binary outcome.')

    Xd = add_intercept(Xa)
    gbeta, separated = logit_fit(Xd, d)
    e_raw = logit_predict(Xd, gbeta)
    e = np.clip(e_raw, trim, 1.0 - trim)
    trim_binding = int(np.sum((e_raw < trim) | (e_raw > 1.0 - trim)))

    m1 = d == 1.0
    m0 = d == 0.0
    mu1, s2_1 = _arm_fit(Xd, yv, m1, family)
    mu0, s2_0 = _arm_fit(Xd, yv, m0, family)
    tau_x = mu1 - mu0

    # AIPW / doubly robust point estimate and its influence function
    psi_aipw = (tau_x
                + d * (yv - mu1) / e
                - (1.0 - d) * (yv - mu0) / (1.0 - e))
    tau = float(np.mean(psi_aipw))
    var_aipw = float(np.mean((psi_aipw - tau) ** 2))

    # Hajek inverse-probability weighting: consistent, but it throws
    # away the outcome regression and pays for it in variance
    w1 = d / e
    w0 = (1.0 - d) / (1.0 - e)
    tau_ipw = float(np.sum(w1 * yv) / np.sum(w1)
                    - np.sum(w0 * yv) / np.sum(w0))
    psi_ipw = w1 * (yv - np.sum(w1 * yv) / np.sum(w1)) \
        - w0 * (yv - np.sum(w0 * yv) / np.sum(w0))
    var_ipw = float(np.mean((psi_ipw - np.mean(psi_ipw)) ** 2))

    overlap = float(np.mean(s2_1 / e + s2_0 / (1.0 - e)))
    heterogeneity = float(np.mean((tau_x - np.mean(tau_x)) ** 2))
    v_eff = overlap + heterogeneity
    se_bound = math.sqrt(v_eff / n)

    mc = minimax_regret_constant()
    regret = mc["constant"] * math.sqrt(v_eff / n)

    zc = _z(1 - alpha / 2)
    se = math.sqrt(var_aipw / n)
    out = RichResult(
        title="Efficiency and minimax bounds for the ATE",
        summary_lines=[
            ("AIPW ATE", tau),
            ("Efficiency bound V_eff", v_eff),
            ("Bound on the SE", se_bound),
            ("AIPW / bound", var_aipw / v_eff if v_eff > 0 else float("nan")),
            ("IPW / bound", var_ipw / v_eff if v_eff > 0 else float("nan")),
            ("Minimax regret bound", regret),
        ],
        tables=[{
            "title": "Where the bound comes from",
            "headers": ["Component", "Value", "Share"],
            "rows": [
                ["Overlap  E[s1^2/e + s0^2/(1-e)]", overlap,
                 overlap / v_eff if v_eff > 0 else float("nan")],
                ["Heterogeneity  Var(tau(X))", heterogeneity,
                 heterogeneity / v_eff if v_eff > 0 else float("nan")],
            ],
        }],
        payload={
            "estimate": tau,
            "ate_aipw": tau,
            "ate_ipw": tau_ipw,
            "se": se,
            "ci_lower": tau - zc * se,
            "ci_upper": tau + zc * se,
            "efficiency_bound": v_eff,
            "se_bound": se_bound,
            "var_aipw": var_aipw,
            "var_ipw": var_ipw,
            "aipw_efficiency_ratio": (var_aipw / v_eff if v_eff > 0
                                      else float("nan")),
            "ipw_efficiency_ratio": (var_ipw / v_eff if v_eff > 0
                                     else float("nan")),
            "overlap_term": overlap,
            "heterogeneity_term": heterogeneity,
            "minimax_regret_bound": regret,
            "minimax_constant": mc["constant"],
            "minimax_t_star": mc["t_star"],
            "propensity": e,
            "propensity_untrimmed": e_raw,
            "tau_x": tau_x,
            "mu1": mu1,
            "mu0": mu0,
            "trim": float(trim),
            "trim_binding": trim_binding,
            "min_propensity": float(np.min(e_raw)),
            "max_propensity": float(np.max(e_raw)),
            "family": family,
            "n": n,
            "n_treated": int(m1.sum()),
            "method": _METHOD,
        },
        interpretation=(
            "V_eff is a floor on the asymptotic variance of every regular "
            "estimator, and c*sqrt(V_eff/n) is a floor on the worst-case "
            "welfare regret of every treatment rule. Neither falls by "
            "improving the estimator; they fall only by improving overlap "
            "or by collecting more data."
        ),
    )
    if separated:
        out.warnings.append(
            "The propensity model separated the data perfectly. The fitted "
            "scores are not trustworthy and neither is any bound computed "
            "from them."
        )
    if trim_binding:
        out.warnings.append(
            f"{trim_binding} of {n} propensity scores lay outside "
            f"[{trim}, {1 - trim}] and were trimmed. The bound reported is "
            "for the trimmed subpopulation, which is a different estimand "
            "from the ATE."
        )
    if var_aipw < v_eff * (1.0 - 1e-8):
        out.warnings.append(
            f"The AIPW influence-function variance ({var_aipw:.6g}) came out "
            f"below the efficiency bound ({v_eff:.6g}). That cannot happen "
            "asymptotically, so one of the nuisance models is misspecified "
            "or the bound's variance components are understated."
        )
    return out


def _z(q):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _Phi(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cheatsheet():
    return (
        "bnsadm: Hahn semiparametric efficiency bound for the ATE and the "
        "Hirano-Porter minimax regret bound for treatment choice, with AIPW "
        "and IPW measured against both"
    )
