# SPDX-License-Identifier: AGPL-3.0-or-later
"""Taphonomic preservation as a causal-inference problem.

A thin domain layer that recasts "is this body's preservation natural or
anomalous?" as a treatment-effect estimate over MORIE's existing causal
estimators. It adds no new statistics: it documents the taphonomy variable set,
dispatches to :func:`morie.estimate_irm` / :func:`morie.estimate_cate`, and
attaches an E-value (:func:`morie.sensitivity.e_value_d`) that quantifies how
strong an *unmeasured* cause would have to be to explain away the natural
preservation effect -- i.e. it turns a claim of "incorruptibility" into a
bounded statistical one.

**This module ships no data.** Forensic-taphonomy comparanda (documented
lime/desiccation burials with preservation outcomes) and any non-invasive
readings of a specific case (CT/micro-CT density, pXRF elemental signatures,
hyperspectral surface composition) must be supplied by the caller from real
sources. :func:`taphonomy_schema` returns a typed, zero-row frame describing the
expected columns; it never fabricates rows.

**What it can and cannot do.** With comparanda it estimates the average
preservation effect attributable to burial *processing* (e.g. quicklime
desiccation) and reports the E-value residual. It cannot falsify a miracle: a
small natural-preservation probability yields a *delta*, not a disproof
(Chernozhukov et al. 2018 give the effect; VanderWeele & Ding 2017 bound the
unmeasured confounding needed to nullify it).

R parity: ``r-morie-oss`` ``R/taphonomy.R``
(``morie_taphonomy_schema`` / ``morie_taphonomy_preservation_delta``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .causal import estimate_cate, estimate_irm
from .sensitivity import e_value_d

# Canonical taphonomy variable set, grouped by causal role. One source of truth
# for both the schema and the estimator.
_TREATMENT = {"lime_treatment": "Int64"}  # 1 = interred with quicklime, 0 = not
_COVARIATES = {
    "temp_c": "float64",             # mean interment temperature (deg C)
    "humidity_pct": "float64",       # mean relative humidity (%)
    "arid": "Int64",                 # 1 = cool/arid microclimate
    "casket_sealed": "Int64",        # 1 = sealed casket / low air exchange
    "reinterment_count": "Int64",    # times exhumed / moved
    "exposure_days": "float64",      # days of pre-burial exposure
    "decades_elapsed": "float64",    # time since death (decades)
}
_MEASUREMENTS = {
    "ct_density_hu": "float64",       # CT/micro-CT mean tissue density (HU)
    "ct_void_fraction": "float64",    # internal void fraction (0-1)
    "pxrf_ca_ppm": "float64",         # pXRF residual surface calcium (ppm)
    "hyperspectral_resin": "Int64",   # 1 = applied resin/wax detected
}
_OUTCOME = {"preservation_score": "float64"}  # 0 (decayed) .. 1 (intact)

_ROLES = (
    [("treatment", c) for c in _TREATMENT]
    + [("covariate", c) for c in _COVARIATES]
    + [("measurement", c) for c in _MEASUREMENTS]
    + [("outcome", c) for c in _OUTCOME]
)


def taphonomy_schema() -> pd.DataFrame:
    """Return an empty, typed DataFrame documenting the expected columns.

    No rows are fabricated -- the caller fills it with real comparanda. The
    ``role`` of each column (``treatment`` / ``covariate`` / ``measurement`` /
    ``outcome``) is stored in ``df.attrs['role']``.
    """
    dtypes = {**_TREATMENT, **_COVARIATES, **_MEASUREMENTS, **_OUTCOME}
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})
    df.attrs["role"] = {c: role for role, c in _ROLES}
    return df


def taphonomy_preservation_delta(
    data: pd.DataFrame,
    *,
    treatment: str = "lime_treatment",
    outcome: str = "preservation_score",
    covariates: list[str] | None = None,
    estimator: str = "irm",
    se_method: str = "none",
    n_boot: int = 199,
    boot_seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Estimate the natural preservation "delta" and its E-value.

    Recasts a taphonomy question as a treatment-effect estimate: how much of the
    observed preservation is attributable to burial *processing* (the
    ``treatment``, e.g. quicklime desiccation), holding environment and handling
    fixed, and how strong an unmeasured cause (a "miracle") would need to be to
    explain the rest. A thin dispatch over :func:`morie.estimate_irm` (default)
    or :func:`morie.estimate_cate` plus :func:`morie.sensitivity.e_value_d`.
    Adds no new statistics.

    Parameters
    ----------
    data : pandas.DataFrame
        Real comparanda (see :func:`taphonomy_schema`). Must be non-empty.
    treatment : str
        Binary treatment column (default ``"lime_treatment"``).
    outcome : str
        Continuous preservation outcome (default ``"preservation_score"``).
    covariates : list[str], optional
        Confounder/measurement columns. Defaults to every schema covariate +
        measurement present in ``data``.
    estimator : str
        ``"irm"`` (DoubleML, default) or ``"cate"``.
    se_method : str
        Inference for the ``"cate"`` path only: ``"none"`` (default) reports the
        point estimate + ``cate_sd`` dispersion with no SE/CI/p-value;
        ``"bootstrap"`` resamples rows and refits the CATE procedure ``n_boot``
        times for a *valid* SE, percentile CI, and p-value. (``sd(tau)/sqrt(n)``
        is deliberately not offered -- the per-unit effects are correlated
        fitted predictions, so it understates the SE; use ``cate_sd`` for
        heterogeneity or bootstrap for inference.) Ignored for ``"irm"``.
    n_boot : int
        Bootstrap resamples when ``se_method="bootstrap"`` (default 199).
    boot_seed : int
        RNG seed for the bootstrap (default 42).
    **kwargs
        Passed to the chosen estimator.

    Returns
    -------
    dict
        All estimates are ``float`` (double); ``n`` is an ``int`` count.
        Keys: ``value`` (preservation delta = ATE / mean CATE), ``se``,
        ``p_value`` (Wald; ``None`` on the CATE path -- no valid SE),
        ``ci_lower``, ``ci_upper``, ``n``, ``e_value``, ``e_value_ci``,
        ``estimator``, ``cate_per_unit`` and ``cate_sd`` (``None`` unless
        ``estimator="cate"``), ``warnings``, ``interpretation``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame")
    if len(data) == 0:
        raise ValueError(
            "`data` is empty. Fill taphonomy_schema() with real comparanda; "
            "this module never fabricates burial rows."
        )
    for col in (treatment, outcome):
        if col not in data.columns:
            raise KeyError(f"column '{col}' not found in `data`")

    if covariates is None:
        wanted = list(_COVARIATES) + list(_MEASUREMENTS)
        covariates = [c for c in wanted if c in data.columns]
    covariates = [c for c in covariates if c not in (treatment, outcome)]
    if not covariates:
        raise ValueError(
            "no covariates available; supply environment/handling/measurement "
            "columns (see taphonomy_schema())"
        )

    warnings: list[str] = []
    if data[treatment].dropna().nunique() < 2:
        warnings.append(
            "treatment has no contrast (all treated or all control) -- "
            "the effect is not identified."
        )

    # Normalise each estimator's output to a common summary. estimate_irm
    # returns a dict (cross-fit ATE + orthogonal SE); estimate_cate returns a
    # per-unit Series, summarised here to its mean (point estimate) + dispersion.
    cate_per_unit: pd.Series | None = None
    cate_sd: float | None = None
    if estimator == "irm":
        est = estimate_irm(
            data, treatment=treatment, outcome=outcome,
            covariates=covariates, **kwargs,
        )
        ate = float(est["ate"])
        se = est.get("se")
        se = float(se) if se is not None else None
        ci_lower, ci_upper = est.get("ci_lower"), est.get("ci_upper")
        n = est.get("n", len(data))
        method = est.get("method", "IRM (DoubleML)")
    elif estimator == "cate":
        tau = estimate_cate(
            data, treatment=treatment, outcome=outcome,
            covariates=covariates, **kwargs,
        ).dropna()
        if len(tau) == 0:
            raise ValueError("CATE estimation returned no finite effects")
        cate_per_unit = tau
        n = int(len(tau))
        ate = float(tau.mean())
        cate_sd = float(tau.std(ddof=1)) if n > 1 else None  # effect heterogeneity
        method = "CATE (meta-learner, mean of per-unit effects)"
        if se_method == "bootstrap":
            # Valid inference: resample rows and refit the whole CATE procedure,
            # so the SE reflects sampling + estimation uncertainty of the mean
            # effect (not the correlated per-unit dispersion).
            import numpy as np

            rng = np.random.default_rng(boot_seed)
            nrow = len(data)
            boot = []
            for _ in range(int(n_boot)):
                idx = rng.integers(0, nrow, size=nrow)
                tb = estimate_cate(
                    data.iloc[idx], treatment=treatment, outcome=outcome,
                    covariates=covariates, **kwargs,
                ).dropna()
                if len(tb):
                    boot.append(float(tb.mean()))
            boot_arr = np.asarray(boot, dtype=float)
            se = float(boot_arr.std(ddof=1))
            ci_lower, ci_upper = (float(x) for x in np.quantile(boot_arr, [0.025, 0.975]))
            method = f"CATE (meta-learner, mean; {len(boot)}-boot SE/CI)"
        elif se_method == "none":
            # Point summary only. sd(tau)/sqrt(n) is NOT reported as an SE: the
            # per-unit effects are correlated fitted predictions, so it would
            # understate the true uncertainty. Use se_method="bootstrap" for a
            # valid SE, or cate_sd for heterogeneity.
            se = None
            ci_lower = ci_upper = None
            _sd = cate_sd if cate_sd is not None else float("nan")
            warnings.append(
                "CATE point summary: no SE reported (per-unit effects are "
                "correlated fitted predictions; "
                f"cate_sd={_sd:.3f} is heterogeneity, range "
                f"[{float(tau.min()):.3f}, {float(tau.max()):.3f}]). Pass "
                "se_method='bootstrap' for a valid SE + CI + p-value, or use "
                "estimator='irm'."
            )
        else:
            raise ValueError(
                f"unknown se_method {se_method!r}; choose 'none' or 'bootstrap'"
            )
    else:
        raise ValueError(
            f"unknown estimator {estimator!r}; choose 'irm' or 'cate'"
        )

    # Wald p-value from a valid SE only (double). IRM's SE is a cross-fit
    # orthogonal SE; the CATE path leaves se = None, so p stays None.
    p_value: float | None = None
    if se is not None and se > 0:
        p_value = float(2.0 * norm.sf(abs(ate / se)))

    out_sd = float(data[outcome].std(skipna=True))
    e_point = e_ci = None
    if out_sd and out_sd > 0:
        d = ate / out_sd
        se_d = (se / out_sd) if (se is not None and se > 0) else None
        ev = e_value_d(d, se=se_d)
        e_point, e_ci = ev.e_value_point, ev.e_value_ci
    else:
        warnings.append("outcome has zero/undefined SD; E-value skipped.")

    lo = float("nan") if ci_lower is None else ci_lower
    hi = float("nan") if ci_upper is None else ci_upper
    interpretation = (
        f"Preservation delta (ATE of {treatment} on {outcome}) = {ate:.3f} "
        f"[{lo:.3f}, {hi:.3f}], n={n}, via {method}. "
    )
    if e_point is not None:
        interpretation += (
            f"An unmeasured cause would need an E-value of {e_point:.2f} "
            "(association with both treatment and outcome, on the risk-ratio "
            "scale) to fully explain it away. This bounds -- it does not falsify "
            "-- any 'incorruptibility' claim: a natural mechanism (e.g. quicklime "
            "desiccation) of this strength is sufficient, but sufficiency is not "
            "proof of exclusivity."
        )
    else:
        interpretation += "E-value unavailable."

    return {
        "value": ate,
        "se": se,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
        "e_value": e_point,
        "e_value_ci": e_ci,
        "estimator": method,
        "cate_per_unit": cate_per_unit,
        "cate_sd": cate_sd,
        "warnings": warnings,
        "interpretation": interpretation,
    }


# ===========================================================================
# Stochastic decay modelling -- absorbing Markov chain (DTMC)
# ===========================================================================

_DECAY_STATES = ("fresh", "bloat", "active", "advanced")
_ABSORBING = ("skeletal", "mummified")


def taphonomy_decay_chain(
    preservation: float = 0.0,
    decay_rate: float = 0.5,
    mummify_rate: float = 0.5,
    states: list[str] | None = None,
) -> dict[str, Any]:
    """Build a taphonomic decay Markov chain (absorbing DTMC).

    From each transient decomposition stage a body either *progresses* one step
    toward the terminal ``"skeletal"`` state (ordinary decay) or *diverts* to
    the terminal ``"mummified"`` state (preserved). ``preservation`` in [0, 1]
    (quicklime desiccation, aridity, sealing) shifts mass from progression
    toward diversion. Compare against the ``preservation=0`` chain to quantify
    how much the burial practice changes the fate distribution.

    Returns a dict with ``P`` (row-stochastic float transition matrix),
    ``states``, ``transient``, ``absorbing`` (``["skeletal", "mummified"]``),
    and ``preservation``. R parity: ``morie_taphonomy_decay_chain``.
    """
    if not 0.0 <= preservation <= 1.0:
        raise ValueError("`preservation` must be in [0, 1]")
    if not (0.0 < decay_rate <= 1.0) or not (0.0 <= mummify_rate <= 1.0):
        raise ValueError("`decay_rate` in (0,1] and `mummify_rate` in [0,1]")
    transient = list(states) if states is not None else list(_DECAY_STATES)
    if len(transient) < 1 or len(set(transient)) != len(transient):
        raise ValueError("`states` must be >= 1 unique transient stage(s)")
    absorbing = list(_ABSORBING)
    all_states = transient + absorbing
    idx = {s: i for i, s in enumerate(all_states)}
    k = len(transient)
    P = np.zeros((len(all_states), len(all_states)))
    for i in range(k):
        prog = decay_rate * (1.0 - preservation)
        mum = mummify_rate * preservation
        tot = prog + mum
        if tot > 1.0:
            prog /= tot
            mum /= tot
            tot = 1.0
        stay = 1.0 - tot
        nxt = transient[i + 1] if i < k - 1 else "skeletal"
        P[i, idx[nxt]] += prog
        P[i, idx["mummified"]] += mum
        P[i, i] += stay
    P[idx["skeletal"], idx["skeletal"]] = 1.0
    P[idx["mummified"], idx["mummified"]] = 1.0
    return {
        "P": P,
        "states": all_states,
        "transient": transient,
        "absorbing": absorbing,
        "preservation": float(preservation),
    }


def taphonomy_decay_absorption(
    chain: dict[str, Any], start: str | None = None
) -> dict[str, Any]:
    """Absorption analysis via the fundamental matrix ``N = (I - Q)^-1``.

    For a body entering at ``start``, returns P(each terminal fate) and the
    expected steps to absorption (Grinstead & Snell, Ch. 11). Keys:
    ``absorption`` (dict over absorbing states, sums to 1), ``expected_steps``,
    ``fundamental`` (N), ``B``. R parity: ``morie_taphonomy_decay_absorption``.
    """
    tr, ab, all_states = chain["transient"], chain["absorbing"], chain["states"]
    if start is None:
        start = tr[0]
    if start not in tr:
        raise ValueError(f"`start` must be a transient state ({', '.join(tr)})")
    idx = {s: i for i, s in enumerate(all_states)}
    tr_i = [idx[s] for s in tr]
    ab_i = [idx[s] for s in ab]
    P = chain["P"]
    Q = P[np.ix_(tr_i, tr_i)]
    R = P[np.ix_(tr_i, ab_i)]
    N = np.linalg.inv(np.eye(len(tr)) - Q)
    B = N @ R
    si = tr.index(start)
    return {
        "absorption": {ab[j]: float(B[si, j]) for j in range(len(ab))},
        "expected_steps": float(N.sum(axis=1)[si]),
        "fundamental": N,
        "B": B,
    }


def taphonomy_decay_simulate(
    chain: dict[str, Any],
    start: str | None = None,
    n_steps: int = 100,
    seed: int = 42,
) -> list[str]:
    """Simulate one realised decay trajectory to an absorbing fate.

    Deterministic given ``seed``. R parity: ``morie_taphonomy_decay_simulate``.
    """
    tr, states, P = chain["transient"], chain["states"], chain["P"]
    if start is None:
        start = tr[0]
    if start not in tr:
        raise ValueError("`start` must be a transient state")
    rng = np.random.default_rng(seed)
    idx = {s: i for i, s in enumerate(states)}
    s = start
    path = [s]
    for _ in range(int(n_steps)):
        s = states[rng.choice(len(states), p=P[idx[s]])]
        path.append(s)
        if s in chain["absorbing"]:
            break
    return path


def taphonomy_decay_delta(
    preservation: float, start: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Natural-vs-treated fate delta: change in P(mummified) from preservation.

    The Markov-chain analogue of the preservation delta -- the rise in the
    probability of ending ``"mummified"`` when a preservation factor is applied,
    relative to the natural (``preservation=0``) baseline. Keys:
    ``p_mummified_natural``, ``p_mummified_treated``, ``delta``,
    ``interpretation``. R parity: ``morie_taphonomy_decay_delta``.
    """
    if not 0.0 < preservation <= 1.0:
        raise ValueError("`preservation` must be in (0, 1] for a contrast")
    nat = taphonomy_decay_chain(preservation=0.0, **kwargs)
    trt = taphonomy_decay_chain(preservation=preservation, **kwargs)
    if start is None:
        start = nat["transient"][0]
    p_nat = taphonomy_decay_absorption(nat, start)["absorption"]["mummified"]
    p_trt = taphonomy_decay_absorption(trt, start)["absorption"]["mummified"]
    delta = p_trt - p_nat
    return {
        "p_mummified_natural": p_nat,
        "p_mummified_treated": p_trt,
        "delta": delta,
        "interpretation": (
            f"P(mummified) rises from {p_nat:.3f} (natural, no preservation) to "
            f"{p_trt:.3f} under preservation={preservation:.2f} -- a fate delta "
            f"of {delta:+.3f}. The preserved outcome is driven by the burial "
            "practice, not baseline decay dynamics."
        ),
    }


# ===========================================================================
# Forensic likelihood-ratio framework
# ===========================================================================


def _lr_verbal(lr: float) -> str:
    """ENFSI (2015) verbal-equivalent scale for a likelihood ratio."""
    if not np.isfinite(lr):
        return "extremely strong support (LR effectively infinite)"
    x = lr if lr >= 1 else 1.0 / lr
    side = "H1" if lr >= 1 else "H2"
    if x <= 1:
        return "no support either way"
    if x <= 10:
        band = "weak support"
    elif x <= 100:
        band = "moderate support"
    elif x <= 1000:
        band = "moderately strong support"
    elif x <= 10000:
        band = "strong support"
    elif x <= 1e6:
        band = "very strong support"
    else:
        band = "extremely strong support"
    return f"{band} for {side}"


def taphonomy_evidence_loglik(evidence, mean, sd) -> float:
    """Gaussian log-likelihood of measured evidence under a model.

    Sum of independent normal log-densities for measured evidence (e.g. pXRF
    calcium, CT density) under a model with expected ``mean`` and ``sd``
    (broadcast over ``evidence``). R parity: ``morie_taphonomy_evidence_loglik``.
    """
    evidence = np.asarray(evidence, dtype=float)
    if evidence.size == 0:
        raise ValueError("`evidence` is empty")
    if not np.all(np.isfinite(evidence)):
        raise ValueError("`evidence` has non-finite values")
    if np.any(np.asarray(sd, dtype=float) <= 0):
        raise ValueError("`sd` must be > 0")
    return float(norm.logpdf(evidence, loc=mean, scale=sd).sum())


def taphonomy_likelihood_ratio(
    loglik_h1: float, loglik_h2: float
) -> dict[str, Any]:
    """Forensic likelihood ratio LR = P(E|H1) / P(E|H2).

    Given the evidence log-likelihood under a natural/target hypothesis and an
    alternative, returns the LR, its base-10 log, and the ENFSI (2015) verbal
    equivalent. Computed in log space for stability. The LR reports how much the
    evidence favours H1 over H2 -- it is not posterior odds and does not prove
    either hypothesis. R parity: ``morie_taphonomy_likelihood_ratio``.
    """
    log_lr = float(loglik_h1) - float(loglik_h2)
    lr = float(np.exp(log_lr))
    log10_lr = log_lr / np.log(10)
    verbal = _lr_verbal(lr)
    if np.isfinite(lr) and lr >= 1:
        factor = f"{lr:.4g} times"
    elif np.isfinite(lr):
        factor = f"{1 / lr:.4g} times less"
    else:
        factor = "infinitely"
    return {
        "lr": lr,
        "log10_lr": float(log10_lr),
        "log_lr": log_lr,
        "verbal": verbal,
        "interpretation": (
            f"LR = {lr:.4g} (log10 = {log10_lr:.3f}): the evidence is {verbal}. "
            f"The observed state is {factor} more probable under H1 (natural "
            "preservation model) than under H2. This quantifies support; it is "
            "not proof and not a posterior probability."
        ),
    }


def taphonomy_preservation_lr(
    evidence, natural: dict[str, Any], alternative: dict[str, Any]
) -> dict[str, Any]:
    """Preservation likelihood ratio from measured evidence.

    Evaluate measured non-invasive evidence under a natural-preservation model
    (H1) and an alternative model (H2), each a dict ``{"mean": .., "sd": ..}``,
    and return the forensic LR (with ``loglik_h1``/``loglik_h2`` attached).
    R parity: ``morie_taphonomy_preservation_lr``.
    """
    for m in (natural, alternative):
        if not {"mean", "sd"} <= set(m):
            raise ValueError("`natural`/`alternative` must be {'mean':.., 'sd':..}")
    ll1 = taphonomy_evidence_loglik(evidence, natural["mean"], natural["sd"])
    ll2 = taphonomy_evidence_loglik(evidence, alternative["mean"], alternative["sd"])
    out = taphonomy_likelihood_ratio(ll1, ll2)
    out["loglik_h1"] = ll1
    out["loglik_h2"] = ll2
    return out


# ===========================================================================
# Bayesian preservation model (conjugate + empirical-Bayes hierarchy)
# ===========================================================================


def taphonomy_bhm(
    data: pd.DataFrame,
    *,
    outcome: str = "preservation_score",
    covariates: list[str] | None = None,
    group: str | None = None,
    priors: dict[str, dict[str, float]] | None = None,
    prior_sd_default: float = 10.0,
) -> dict[str, Any]:
    """Bayesian hierarchical preservation model (conjugate + EB pooling).

    Conjugate Gaussian-linear Bayesian model with informative Normal priors on
    the coefficients (domain knowledge, e.g. quicklime's desiccant effect,
    enters as a prior). Closed-form posterior; noise variance from OLS residuals
    (empirical Bayes). With ``group``, adds empirical-Bayes partial-pooled random
    intercepts (normal-normal shrinkage) for a two-level hierarchical model.

    For full HMC/NUTS hierarchical inference, use ``bambi``/``pymc`` (Python) or
    ``rstanarm`` (R); this is the dependency-free conjugate core. R parity:
    ``morie_taphonomy_bhm``.

    Returns a dict: ``coefficients`` (DataFrame: term, post_mean, post_sd,
    ci_lower, ci_upper, prob_positive), ``sigma``, ``group_effects`` (None
    unless ``group``), ``fitted``, ``n``, ``interpretation``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame")
    if len(data) == 0:
        raise ValueError("`data` is empty")
    if outcome not in data.columns:
        raise KeyError(f"column '{outcome}' not found")
    if covariates is None:
        wanted = list(_COVARIATES) + list(_MEASUREMENTS)
        covariates = [c for c in wanted if c in data.columns]
    covariates = [c for c in covariates if c not in (outcome, group)]
    if not covariates:
        raise ValueError("no covariates available")

    cols = [outcome, *covariates] + ([group] if group else [])
    frame = data[cols].dropna().copy()
    y = frame[outcome].to_numpy(dtype=float)
    n = int(len(y))

    def _numeric(col):
        s = frame[col]
        return s.to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(s) \
            else pd.factorize(s)[0].astype(float)

    terms = ["(Intercept)", *covariates]
    X = np.column_stack([np.ones(n)] + [_numeric(c) for c in covariates])
    p = X.shape[1]

    m0 = np.zeros(p)
    s0 = np.full(p, float(prior_sd_default))
    for i, t in enumerate(terms):
        if priors and t in priors:
            m0[i] = float(priors[t]["mean"])
            s0[i] = float(priors[t]["sd"])
    Lambda0 = np.diag(1.0 / s0**2)

    # Empirical-Bayes noise variance from the OLS fit.
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = max(1, n - p)
    sigma2 = float(((y - X @ beta_ols) ** 2).sum() / dof)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = float(np.var(y, ddof=1)) if n > 1 else 1.0

    Sigma = np.linalg.inv(X.T @ X / sigma2 + Lambda0)
    mu = Sigma @ (X.T @ y / sigma2 + Lambda0 @ m0)
    post_sd = np.sqrt(np.diag(Sigma))
    z = 1.959964
    coefficients = pd.DataFrame(
        {
            "term": terms,
            "post_mean": mu,
            "post_sd": post_sd,
            "ci_lower": mu - z * post_sd,
            "ci_upper": mu + z * post_sd,
            "prob_positive": norm.cdf(mu / post_sd),
        }
    )
    fitted = X @ mu

    group_effects = None
    if group is not None:
        g = frame[group].to_numpy()
        resid = y - fitted
        labels = pd.unique(g)
        gm = np.array([resid[g == lab].mean() for lab in labels])
        nj = np.array([int((g == lab).sum()) for lab in labels])
        tau2 = max(0.0, float(np.var(gm, ddof=1)) - float(np.mean(sigma2 / nj))) \
            if len(gm) > 1 else 0.0
        lam = tau2 / (tau2 + sigma2 / nj)
        group_effects = pd.DataFrame(
            {
                "group": labels,
                "raw_mean": gm,
                "shrinkage": lam,
                "pooled_intercept": lam * gm,
                "n": nj,
            }
        )

    lime_terms = [t for t in ("lime_treatment", covariates[0]) if t in terms]
    lrow = coefficients.loc[coefficients["term"] == lime_terms[0]].iloc[0]
    interp = (
        f"Bayesian preservation model (n={n}, conjugate Gaussian, EB noise "
        f"sd={sigma2 ** 0.5:.3f}). Posterior effect of '{lrow['term']}' = "
        f"{lrow['post_mean']:.3f} [{lrow['ci_lower']:.3f}, {lrow['ci_upper']:.3f}], "
        f"P(effect>0)={lrow['prob_positive']:.3f}. Priors update to posteriors: "
        "an informative lime prior encodes the desiccant belief, the data "
        "revises it."
    )
    if group is not None:
        interp += f" {len(group_effects)} group intercepts partially pooled."

    return {
        "coefficients": coefficients,
        "sigma": float(sigma2 ** 0.5),
        "group_effects": group_effects,
        "fitted": fitted,
        "n": n,
        "interpretation": interp,
    }
