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
from morie.fn import _frame_core as pd
from morie.fn._stats_core import norm

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


_MORIE_BHM_STAN = """
data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  vector[N] y;
  vector[K] prior_mean;
  vector<lower=0>[K] prior_sd;
  int<lower=0> J;
  array[N] int<lower=0> g;
}
parameters {
  vector[K] beta;
  real<lower=0> sigma;
  vector[J] z;
  real<lower=0> tau;
}
model {
  beta ~ normal(prior_mean, prior_sd);
  sigma ~ exponential(1);
  tau ~ exponential(1);
  z ~ normal(0, 1);
  vector[N] mu = X * beta;
  if (J > 0)
    for (n in 1:N) mu[n] += z[g[n]] * tau;
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[J] group_intercept;
  for (j in 1:J) group_intercept[j] = z[j] * tau;
}
"""


def taphonomy_bhm(
    data: pd.DataFrame,
    *,
    outcome: str = "preservation_score",
    covariates: list[str] | None = None,
    group: str | None = None,
    priors: dict[str, dict[str, float]] | None = None,
    prior_sd_default: float = 10.0,
    backend: str = "conjugate",
    chains: int = 4,
    iter: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bayesian hierarchical preservation model (conjugate + EB pooling).

    Conjugate Gaussian-linear Bayesian model with informative Normal priors on
    the coefficients (domain knowledge, e.g. quicklime's desiccant effect,
    enters as a prior). Closed-form posterior; noise variance from OLS residuals
    (empirical Bayes). With ``group``, adds empirical-Bayes partial-pooled random
    intercepts (normal-normal shrinkage) for a two-level hierarchical model.

    ``backend="conjugate"`` (default) uses the dependency-free closed form;
    ``backend="cmdstanpy"`` fits the same model by full-Bayes HMC/NUTS via
    cmdstanpy + a built CmdStan (returns the ``stanfit`` too). ``chains``/
    ``iter``/``seed`` control the sampler. R parity: ``morie_taphonomy_bhm``
    (its ``backend="cmdstanr"``).

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

    if backend == "cmdstanpy":
        return _bhm_cmdstanpy(X, y, terms, m0, s0, frame, group, chains, iter, seed)
    if backend != "conjugate":
        raise ValueError(f"unknown backend {backend!r}; 'conjugate' or 'cmdstanpy'")

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
        "backend": "conjugate (closed form)",
        "interpretation": interp,
    }


def _bhm_cmdstanpy(X, y, terms, m0, s0, frame, group, chains, iter, seed):
    """HMC/NUTS fit of the BHM via cmdstanpy (same Stan model as R cmdstanr)."""
    try:
        import cmdstanpy
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "backend='cmdstanpy' needs the 'cmdstanpy' package and a built "
            "CmdStan. pip install cmdstanpy; "
            "python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'."
        ) from e
    import tempfile
    from pathlib import Path

    n = X.shape[0]
    if group is not None:
        codes, levels = pd.factorize(frame[group])
        J = int(len(levels))
        g = (codes + 1).astype(int)  # Stan is 1-indexed
    else:
        J, g, levels = 0, np.zeros(n, dtype=int), []
    standata = {
        "N": int(n), "K": int(X.shape[1]), "X": X.tolist(), "y": y.tolist(),
        "prior_mean": m0.tolist(), "prior_sd": s0.tolist(),
        "J": J, "g": g.tolist(),
    }
    stan_path = Path(tempfile.gettempdir()) / "morie_taphonomy_bhm.stan"
    stan_path.write_text(_MORIE_BHM_STAN)
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))
    fit = model.sample(
        data=standata, chains=chains, iter_warmup=iter, iter_sampling=iter,
        seed=seed, show_progress=False, show_console=False,
    )
    beta = fit.stan_variable("beta")          # (draws, K)
    sig = fit.stan_variable("sigma")
    coefficients = pd.DataFrame(
        {
            "term": terms,
            "post_mean": beta.mean(axis=0),
            "post_sd": beta.std(axis=0, ddof=1),
            "ci_lower": np.quantile(beta, 0.025, axis=0),
            "ci_upper": np.quantile(beta, 0.975, axis=0),
            "prob_positive": (beta > 0).mean(axis=0),
        }
    )
    group_effects = None
    if J > 0:
        gi = fit.stan_variable("group_intercept")  # (draws, J)
        group_effects = pd.DataFrame(
            {
                "group": list(levels),
                "pooled_intercept": gi.mean(axis=0),
                "post_sd": gi.std(axis=0, ddof=1),
                "n": [int((g == j + 1).sum()) for j in range(J)],
            }
        )
    fitted = X @ coefficients["post_mean"].to_numpy()
    if J > 0:
        fitted = fitted + group_effects["pooled_intercept"].to_numpy()[g - 1]
    lime = "lime_treatment" if "lime_treatment" in terms else terms[1]
    lrow = coefficients.loc[coefficients["term"] == lime].iloc[0]
    return {
        "coefficients": coefficients,
        "sigma": float(sig.mean()),
        "group_effects": group_effects,
        "fitted": fitted,
        "n": int(n),
        "backend": "cmdstanpy (NUTS)",
        "stanfit": fit,
        "interpretation": (
            f"Bayesian preservation model (n={n}, HMC/NUTS via cmdstanpy, "
            f"{chains} chains x {iter} draws). Posterior effect of '{lrow['term']}' "
            f"= {lrow['post_mean']:.3f} [{lrow['ci_lower']:.3f}, "
            f"{lrow['ci_upper']:.3f}], P(effect>0)={lrow['prob_positive']:.3f}. "
            "Full-Bayes posterior (no conjugacy approximation)."
        ),
    }


# ===========================================================================
# Synthetic pXRF (compositional) generation + log-ratio transforms
# ===========================================================================

_PXRF_ELEMENTS = ("Ca", "P", "Fe", "Sr", "Pb", "Zn")
_PXRF_ALPHA = {
    "control": (30.0, 15.0, 5.0, 1.0, 0.5, 0.5),   # natural soil/bone matrix
    "treatment": (85.0, 5.0, 2.0, 0.5, 0.2, 0.2),  # quicklime: calcium spike
}


def taphonomy_simulate_pxrf(
    n: int,
    condition: str = "control",
    elements: list[str] | None = None,
    alpha=None,
    seed: int | None = None,
    as_ppm: bool = False,
    total_ppm: float = 1e6,
) -> pd.DataFrame:
    """Simulate synthetic pXRF compositional data (Dirichlet).

    **Synthetic data for testing/calibration ONLY -- never a substitute for real
    comparanda, and never written to bundled data.** pXRF elemental
    concentrations are closed compositions on the simplex, so this samples from
    a Dirichlet (not Gaussian). Control = natural matrix; treatment = quicklime
    (calcium-skewed). R parity: ``morie_taphonomy_simulate_pxrf``.

    Returns a DataFrame with one column per element plus ``condition`` and
    ``lime_treatment`` (1 = treatment); ``df.attrs`` carries ``elements``,
    ``alpha``, ``synthetic=True``.
    """
    elements = list(elements) if elements is not None else list(_PXRF_ELEMENTS)
    if alpha is None:
        if condition not in _PXRF_ALPHA:
            raise ValueError("condition must be 'control' or 'treatment'")
        alpha = list(_PXRF_ALPHA[condition])
        if len(alpha) != len(elements):
            raise ValueError("supply `alpha` when using custom `elements`")
    alpha = list(alpha)
    if len(alpha) != len(elements):
        raise ValueError("len(alpha) must equal len(elements)")
    if any(a <= 0 for a in alpha):
        raise ValueError("`alpha` must be > 0")
    rng = np.random.default_rng(seed)
    comp = rng.dirichlet(alpha, size=int(n))
    if as_ppm:
        comp = comp * total_ppm
    df = pd.DataFrame(comp, columns=elements)
    df["condition"] = condition
    df["lime_treatment"] = int(condition == "treatment")
    df.attrs.update(elements=elements, alpha=alpha, synthetic=True)
    return df


def _close(x, pseudocount: float) -> np.ndarray:
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if np.any(X < 0):
        raise ValueError("compositions must be non-negative")
    X = X + pseudocount
    return X / X.sum(axis=1, keepdims=True)


def taphonomy_clr(x, pseudocount: float = 1e-6) -> np.ndarray:
    """Centred log-ratio (CLR) transform: clr(x) = log x - mean(log x), row-wise.

    Removes simplex closure. Rank-deficient (columns sum to 0); for regression
    inputs prefer :func:`taphonomy_ilr`. R parity: ``morie_taphonomy_clr``.
    """
    X = _close(x, pseudocount)
    L = np.log(X)
    return L - L.mean(axis=1, keepdims=True)


def taphonomy_ilr(x, pseudocount: float = 1e-6) -> np.ndarray:
    """Isometric log-ratio (ILR): D-part composition -> D-1 orthonormal coords.

    Full-rank (unlike CLR), so it feeds DML/BHM without a singular design.
    Egozcue et al. (2003) pivot-coordinate basis, closed form -- identical to
    the R sibling. R parity: ``morie_taphonomy_ilr``.
    """
    X = _close(x, pseudocount)
    D = X.shape[1]
    if D < 2:
        raise ValueError("need >= 2 parts for ILR")
    L = np.log(X)
    cols = [
        np.sqrt(i / (i + 1)) * (L[:, :i].mean(axis=1) - L[:, i])
        for i in range(1, D)
    ]
    return np.column_stack(cols)


# ===========================================================================
# Open-data ingestion: real comparanda for calibration
# ===========================================================================

# USGS National Geochemical Database (soil) bulk CSV (verified 2026-07-03;
# 54 MB zip -> 482 MB CSV; subject to federal-portal reorg).
_USGS_NGDBSOIL_URL = "https://mrdata.usgs.gov/ngdb/soil/ngdbsoil-csv.zip"


def _read_usgs_soil_zip(zip_path, nrows=None) -> pd.DataFrame:
    """Read the CSV member of a downloaded ngdbsoil zip (no full extract)."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"no CSV member in {zip_path}")
        with zf.open(members[0]) as fh:
            return pd.read_csv(fh, nrows=nrows, low_memory=False)


def taphonomy_fetch_usgs_soil(
    dest=None, nrows: int | None = 1000, url: str = _USGS_NGDBSOIL_URL,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch USGS National Geochemical Database soil geochemistry.

    Downloads the USGS NGDB (soil) bulk CSV -- **real, open** elemental
    concentrations (the compositional analogue of pXRF spectra) to calibrate the
    ``clr``/``ilr``/DML pipeline on genuine open data before real scans exist.
    Dependency-free (base zipfile + pandas; the WFS endpoint is GML-only and not
    used). ~54 MB download (482 MB uncompressed), cached in ``dest``.
    R parity: ``morie_taphonomy_fetch_usgs_soil``.
    """
    import tempfile
    import urllib.request
    from pathlib import Path

    dest = Path(dest) if dest is not None else Path(tempfile.gettempdir())
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / url.rsplit("/", 1)[-1]
    if refresh or not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 (fixed https URL)
    df = _read_usgs_soil_zip(zip_path, nrows)
    df.attrs["source"] = url
    return df


def taphonomy_pmi_schema() -> pd.DataFrame:
    """STO-2022 taphonomic-observation schema (PMI nuisance variables).

    Typed zero-row template of the taphonomic-observation + environmental
    variables for post-mortem-interval work (Standard for Taphonomic
    Observations in Support of the PMI, 2022 -- the geoFOR variable family).
    **geoFOR itself is an app with no open data API**, so this structures your
    own case observations into the DML/Bayesian nuisance set X. No fabricated
    rows. R parity: ``morie_taphonomy_pmi_schema``.
    """
    dtypes = {
        "decomp_stage": "Int64",
        "body_scoring_tbs": "float64",
        "accumulated_deg_days": "float64",
        "temp_c": "float64",
        "humidity_pct": "float64",
        "precipitation_mm": "float64",
        "burial_depth_cm": "float64",
        "soil_ph": "float64",
        "scavenger_activity": "Int64",
        "insect_activity": "Int64",
        "pmi_days": "float64",
    }
    roles = (
        ["observation"] * 2 + ["environment"] * 8 + ["outcome"]
    )
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})
    df.attrs["role"] = dict(zip(df.columns, roles))
    return df


# ===========================================================================
# MorphoSource client -- 3D bioarchaeology media (user-supplied API key)
# ===========================================================================
#
# Endpoints + auth verified 2026-07-03 against github.com/Imageomics/
# pyMorphoSource: base https://www.morphosource.org/api (override
# MORPHOSOURCE_API_URL); search = GET /media|/physical-objects with q +
# search_field=all_fields + f.<facet> + per_page/page; download = POST
# /download/<id> with Authorization: <key> + use-statement body ->
# response.media.download_url. Key from the caller's own env; never hard-coded.


def _morphosource_api() -> str:
    import os

    return os.environ.get("MORPHOSOURCE_API_URL", "https://www.morphosource.org/api")


def _morphosource_key(api_key: str | None = None, required: bool = True):
    import os

    key = api_key or os.environ.get("MORPHOSOURCE_API_KEY", "")
    if not key:
        if required:
            raise ValueError(
                "MorphoSource API key required. Pass api_key= or export "
                "MORPHOSOURCE_API_KEY (token from your account at "
                "https://www.morphosource.org). Never hard-code the key."
            )
        return None
    return key


def _morphosource_search_params(
    query=None, media_type=None, taxonomy_gbif=None, visibility=None,
    media_tag=None, per_page=10, page=1,
) -> dict:
    params = {}
    if query:
        params["q"] = query
        params["search_field"] = "all_fields"
    facets = {
        "media_type": media_type, "taxonomy_gbif": taxonomy_gbif,
        "publication_status": visibility, "tag": media_tag,
    }
    for k, v in facets.items():
        if v:
            params[f"f.{k}"] = v
    params["per_page"] = int(per_page)
    params["page"] = int(page)
    return params


def taphonomy_morphosource_search(
    query=None, kind="media", media_type=None, taxonomy_gbif=None,
    visibility=None, media_tag=None, per_page=10, page=1, api_key=None,
) -> dict:
    """Search MorphoSource for 3D media or physical objects.

    Public search needs no key; a key (restricted records) comes from api_key or
    MORPHOSOURCE_API_KEY and travels only in the Authorization header. ``kind``
    is ``"media"`` or ``"physical-objects"``. Returns a dict with ``items``,
    ``n``, ``total_pages``, and ``df`` (id + title). R parity:
    ``morie_taphonomy_morphosource_search``.
    """
    import json
    import urllib.parse
    import urllib.request

    if kind not in ("media", "physical-objects"):
        raise ValueError("kind must be 'media' or 'physical-objects'")
    params = _morphosource_search_params(
        query, media_type, taxonomy_gbif, visibility, media_tag, per_page, page
    )
    url = f"{_morphosource_api()}/{kind}?" + urllib.parse.urlencode(params)
    key = _morphosource_key(api_key, required=False)
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", key)
    with urllib.request.urlopen(req) as r:  # noqa: S310 (fixed https base)
        parsed = json.load(r)["response"]
    items_name = "media" if kind == "media" else "physical_objects"
    items = parsed.get(items_name) or []
    df = pd.DataFrame(
        {"id": [str(i.get("id", "")) for i in items],
         "title": [str(i.get("title", "")) for i in items]}
    )
    return {
        "items": items,
        "n": len(items),
        "total_pages": (parsed.get("pages") or {}).get("total_pages"),
        "df": df,
    }


def taphonomy_morphosource_fetch(
    media_id, use_statement, use_categories=None, use_category_other=None,
    dest=None, api_key=None,
) -> str:
    """Download a MorphoSource media bundle (requires a data-use statement).

    MorphoSource enforces a use agreement on every download, so ``use_statement``
    is required (``agreements_accepted=True`` is sent); restricted media also
    need per-item permission on the site. Key from api_key/MORPHOSOURCE_API_KEY,
    Authorization header only. Returns the path to the downloaded zip. R parity:
    ``morie_taphonomy_morphosource_fetch``.
    """
    import json
    import tempfile
    import urllib.error
    import urllib.request
    from pathlib import Path

    if not use_statement or not isinstance(use_statement, str):
        raise ValueError(
            "MorphoSource requires a non-empty use_statement (data-use agreement)."
        )
    key = _morphosource_key(api_key, required=True)
    body = {"use_statement": use_statement, "agreements_accepted": True}
    if use_categories:
        body["use_categories"] = use_categories
    if use_category_other:
        body["use_category_other"] = use_category_other
    url = f"{_morphosource_api()}/download/{media_id}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:  # noqa: S310
            dl_url = json.load(r)["response"]["media"]["download_url"]
    except urllib.error.HTTPError as e:  # noqa: F821
        if e.code == 403:
            raise PermissionError(
                f"Restricted media {media_id}: request download permission at "
                "https://www.morphosource.org"
            ) from e
        raise
    dest = Path(dest) if dest is not None else Path(tempfile.gettempdir())
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"morphosource_{media_id}.zip"
    dreq = urllib.request.Request(dl_url, headers={"Authorization": key})
    with urllib.request.urlopen(dreq) as r, open(path, "wb") as fh:  # noqa: S310
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    return str(path)
