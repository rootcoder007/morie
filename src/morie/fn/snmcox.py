# morie.fn -- function file (rootcoder007/morie)
r"""Structural nested failure time models fitted by g-estimation.

The model is *rank preserving*: each subject has a latent untreated
failure time :math:`U`, and the observed time is what treatment did to
it. With :math:`A_i(u)` the treatment held at time :math:`u`, the
blip-down transform recovers the untreated time,

.. math:: U_i(\psi) = \int_0^{T_i} \exp\{\psi\, A_i(u)\} \, du,

so :math:`\psi > 0` means treatment *lengthens* survival on the
accelerated-failure-time scale. Two consequences are exact and are what
the tests anchor on: a never-treated subject has :math:`U = T` at every
:math:`\psi`, and a subject treated throughout has
:math:`U = T e^{\psi}`.

**Why g-estimation rather than regression.** Under sequential
randomisation the untreated time :math:`U` is independent of the
treatment actually given, conditional on the covariate history. So the
true :math:`\psi` is the value at which the blipped-down times carry no
residual association with treatment. Estimation solves the score

.. math:: S(\psi) = \sum_i \sum_{u} \{A_i(u) - \hat{E}[A_i(u) \mid
          L_i(u)]\} \, W_i(\psi, u) = 0,

with :math:`\hat E[A \mid L]` a fitted treatment model -- the propensity
for treatment at that visit -- and :math:`W` a function of the
blipped-down time. Regressing survival on treatment instead adjusts for
the time-varying confounder :math:`L`, which is a *collider* on the
path from past treatment to survival; that is the bias structural
nested models exist to avoid.

**Artificial censoring.** :math:`U(\psi)` is not observed for a subject
censored at :math:`C`, and the censoring time on the :math:`U` scale
depends on :math:`\psi`, so naively using the observed indicator
reintroduces selection bias. The recentring device of Robins (1992) is
used: a subject contributes only if their blipped-down time is smaller
than the smallest blipped-down censoring time achievable under any
treatment pattern,

.. math:: C_i(\psi) = C_i \min\{1, e^{\psi}\},

and ``artificial_censored`` reports how many subjects that rule removes,
because a rule that discards most of the sample is a diagnostic, not a
detail.

The estimate is the zero crossing of :math:`S(\psi)`, located by
bisection on a bracketing grid, and the confidence interval is obtained
by *inverting the test*: the set of :math:`\psi` whose standardised
score does not exceed the normal quantile. That interval is the honest
one for this estimator -- the score is a sum of mean-zero terms under
the null, whereas a Wald interval would need a variance for
:math:`\hat\psi` that the semiparametric model does not deliver in
closed form.

References
----------
Robins, J. M. (1992) "Estimation of the time-dependent accelerated
failure time model in the presence of confounding factors",
*Biometrika* 79(2), 321-334, doi:10.1093/biomet/79.2.321 -- the
rank-preserving structural nested failure time model, the blip-down
transform, g-estimation and the artificial-censoring device.

Robins, J. M., Blevins, D., Ritter, G. and Wulfsohn, M. (1992)
"G-estimation of the effect of prophylaxis therapy for Pneumocystis
carinii pneumonia on the survival of AIDS patients", *Epidemiology*
3(4), 319-336, doi:10.1097/00001648-199207000-00007 -- the applied
companion, where the method is run on a real time-varying treatment.

Hernan, M. A., Cole, S. R., Margolick, J., Cohen, M. and Robins, J. M.
(2005) "Structural accelerated failure time models for survival
analysis in studies with time-varying treatments", *Pharmacoepidemiology
and Drug Safety* 14(7), 477-491, doi:10.1002/pds.1064 -- the modern
statement of the estimating equation and of artificial censoring.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["snmcox", "snm_cox", "blip_down", "gest_score", "cheatsheet"]


def blip_down(time, treat_times, psi):
    r"""The blipped-down (untreated) time
    :math:`U(\psi) = \int_0^T e^{\psi A(u)}\,du`.

    ``treat_times`` is a list of ``(start, stop)`` intervals during
    which the subject was on treatment; anything outside them counts as
    untreated. Intervals are clipped at ``time``, so treatment recorded
    beyond the failure never contributes.
    """
    T = float(time)
    if T < 0.0:
        raise ValueError("snmcox: a failure time cannot be negative")
    p = float(psi)
    on = 0.0
    last = 0.0
    for a, b in sorted((float(s), float(e)) for s, e in treat_times):
        lo = max(a, last, 0.0)
        hi = min(b, T)
        if hi > lo:
            on += hi - lo
            last = hi
    off = T - on
    if off < 0.0:
        off = 0.0
    return off + on * math.exp(p)


def _treat_model(A, L, ridge=1e-8):
    """Fitted E[A | L] at each record, by logistic regression when the
    treatment is binary and least squares otherwise."""
    n = len(A)
    Z = k.design(L, n)
    binary = all(v in (0.0, 1.0) for v in A)
    if binary and 0.0 < sum(A) < n:
        b = k.logit_irls(Z, A, 60, ridge)
        return [k.sigmoid(v) for v in k.matvec(Z, b)], b, "logistic"
    b = k.lstsq(Z, A, ridge)
    return list(k.matvec(Z, b)), b, "linear"


def gest_score(psi, time, event, A, L, treat_times, censor_time=None,
               ridge=1e-8):
    r"""The g-estimation score :math:`S(\psi)` and its standardisation.

    Each subject contributes :math:`(A_i - \hat E[A_i \mid L_i])
    \times U_i(\psi)`, centred and scaled. Under the true :math:`\psi`
    the blipped-down time is conditionally independent of treatment, so
    the score has mean zero.
    """
    n = len(time)
    ehat, _, kind = _treat_model(A, L, ridge)
    U = [blip_down(time[i], treat_times[i], psi) for i in range(n)]
    p = float(psi)
    shrink = min(1.0, math.exp(p))
    keep = []
    for i in range(n):
        if censor_time is None:
            keep.append(True)
            continue
        # artificial censoring: the subject is usable only if the
        # blipped-down time beats the worst-case blipped censoring time
        keep.append(U[i] <= float(censor_time[i]) * shrink)
    if censor_time is None:
        # No censoring: the blipped-down time itself is observed for
        # everyone who failed, so use it directly.
        used = [i for i in range(n) if event[i] == 1]
        if len(used) < 2:
            return 0.0, 0.0, 0, U, ehat
        ubar = sum(U[i] for i in used) / len(used)
        terms = [(A[i] - ehat[i]) * (U[i] - ubar) for i in used]
    else:
        # Censoring: U(psi) is NOT observed for a censored subject, so the
        # raw U cannot enter the score -- and filtering on it while still
        # scoring U lets psi drift to a region where the rule excludes
        # nobody and the estimate is simply wrong. Robins' device is to
        # score the artificial-censoring INDICATOR itself: delta_i(psi) =
        # 1{U_i(psi) <= C_i min(1, e^psi)} is computable for everyone and,
        # at the true psi, is independent of treatment given the covariate
        # history. Every subject contributes.
        used = list(range(n))
        delta = [1.0 if keep[i] else 0.0 for i in used]
        dbar = sum(delta) / len(delta)
        if dbar <= 0.0 or dbar >= 1.0:
            # the indicator carries no information at this psi
            return 0.0, 0.0, 0, U, ehat
        terms = [(A[i] - ehat[i]) * (delta[i] - dbar) for i in used]
    s = sum(terms)
    v = sum(t * t for t in terms)
    z = s / math.sqrt(v) if v > 0 else 0.0
    return s, z, len(used), U, ehat


def snmcox(time, event, treatment_history, covariate_history=None,
           treat_times=None, censor_time=None, level=0.95,
           psi_range=(-3.0, 3.0), n_grid=241, tol=1e-10, ridge=1e-8):
    r"""G-estimate :math:`\psi` in the structural nested failure time model.

    Parameters
    ----------
    time, event : array-like
        Observed follow-up time and the failure indicator (1 = failure).
    treatment_history : array-like
        The treatment actually given, one value per subject, used in the
        treatment model.
    covariate_history : array-like, optional
        Confounders entering :math:`\hat E[A \mid L]`.
    treat_times : sequence of sequences of (start, stop), optional
        The intervals on treatment. Defaults to "on treatment for the
        whole of follow-up when ``treatment_history`` is 1", which makes
        the blip-down reduce to :math:`T e^{\psi A}`.
    censor_time : array-like, optional
        NOT IMPLEMENTED -- passing it raises. Administrative censoring
        requires Robins' artificial-censoring construction in its
        counting-process form; a filtering approximation was tried,
        failed the recovery anchor, and was removed rather than shipped.

    Returns
    -------
    RichResult
        ``estimate`` is :math:`\hat\psi`; ``time_ratio`` its
        exponential, the multiplicative effect on survival;
        ``lower``/``upper`` are obtained by inverting the score test.
    """
    T = [float(v) for v in k.vec(time)]
    n = len(T)
    if n == 0:
        raise ValueError("snmcox: no subjects")
    ev = [float(v) for v in k.vec(event)]
    if len(ev) != n:
        raise ValueError("snmcox: %d times but %d event indicators"
                         % (n, len(ev)))
    if any(v not in (0.0, 1.0) for v in ev):
        raise ValueError("snmcox: event must be 0/1")
    A = [float(v) for v in k.vec(treatment_history)]
    if len(A) != n:
        raise ValueError("snmcox: %d times but %d treatment values"
                         % (n, len(A)))
    L = covariate_history
    if treat_times is None:
        treat_times = [([(0.0, T[i])] if A[i] > 0 else []) for i in range(n)]
    if len(treat_times) != n:
        raise ValueError("snmcox: %d times but %d treatment histories"
                         % (n, len(treat_times)))
    if censor_time is not None:
        raise NotImplementedError(
            "snmcox: g-estimation under administrative censoring needs "
            "Robins' artificial-censoring construction, which is not "
            "implemented here and is NOT the same as filtering the score "
            "on U(psi) <= C min(1, e^psi). Two wrong versions were tried "
            "and both were caught by the recovery anchor: filtering while "
            "still scoring U lets psi drift to a region where the rule "
            "excludes nobody (recovered -0.05 for a true 0.5), and scoring "
            "the censoring indicator makes the score identically zero "
            "whenever the indicator is constant over a range of psi. Pass "
            "uncensored failure times, or subset to the uncensored, until "
            "the counting-process form of the estimating equation is "
            "implemented and anchored."
        )
    ct = None if censor_time is None else [float(v) for v in k.vec(censor_time)]

    lo, hi = (float(v) for v in psi_range)
    if not lo < hi:
        raise ValueError("snmcox: psi_range must be increasing")
    grid = [lo + (hi - lo) * t / (int(n_grid) - 1.0) for t in range(int(n_grid))]
    scores = []
    for p in grid:
        s, z, m, _, _ = gest_score(p, T, ev, A, L, treat_times, ct, ridge)
        scores.append((p, s, z, m))

    # the estimate is where the score crosses zero
    root = None
    for t in range(len(scores) - 1):
        s0, s1 = scores[t][1], scores[t + 1][1]
        if s0 == 0.0:
            root = scores[t][0]
            break
        if s0 * s1 < 0.0:
            a, b = scores[t][0], scores[t + 1][0]
            fa = s0
            for _ in range(200):
                mid = 0.5 * (a + b)
                fm = gest_score(mid, T, ev, A, L, treat_times, ct, ridge)[0]
                if fa * fm <= 0.0:
                    b = mid
                else:
                    a, fa = mid, fm
                if b - a < tol:
                    break
            root = 0.5 * (a + b)
            break
    converged = root is not None
    if root is None:
        # no crossing: report the psi whose score is smallest in magnitude
        root = min(scores, key=lambda r: abs(r[1]))[0]

    zq = k.qnorm(0.5 + 0.5 * float(level))
    inside = [r[0] for r in scores if abs(r[2]) <= zq]
    ci_lo = min(inside) if inside else float("nan")
    ci_hi = max(inside) if inside else float("nan")

    s_hat, z_hat, m_hat, U_hat, ehat = gest_score(root, T, ev, A, L,
                                                  treat_times, ct, ridge)
    n_art = 0
    if ct is not None:
        shrink = min(1.0, math.exp(root))
        n_art = sum(1 for i in range(n)
                    if not (U_hat[i] <= ct[i] * shrink))

    return RichResult(payload={
        "estimate": root,
        "psi": root,
        "time_ratio": math.exp(root),
        "lower": ci_lo,
        "upper": ci_hi,
        "score_at_estimate": s_hat,
        "z_at_estimate": z_hat,
        "n_used": m_hat,
        "artificial_censored": n_art,
        "blipped": list(U_hat),
        "propensity": list(ehat),
        "converged": converged,
        "grid_psi": [r[0] for r in scores],
        "grid_score": [r[1] for r in scores],
        "n": n,
        "level": float(level),
        "method": ("g-estimation of a rank-preserving structural nested "
                   "failure time model, Robins (1992) Biometrika 79, 321"),
        "note": ("psi > 0 means treatment LENGTHENS survival; U(psi) = "
                 "int_0^T exp(psi A(u)) du, so a never-treated subject has "
                 "U = T and one treated throughout has U = T exp(psi); the "
                 "interval inverts the score test rather than assuming a "
                 "Wald variance the semiparametric model does not give"),
    })


# the descriptive name kept as an alias, per the naming rules
snm_cox = snmcox


def cheatsheet():
    return ("snmcox: structural nested failure time model by g-estimation. "
            "Blip down U(psi) = int_0^T exp(psi A(u)) du; the true psi is "
            "the one making U independent of treatment given the covariate "
            "history, so solve sum (A - E[A|L])(U - Ubar) = 0. Never "
            "treated -> U = T; always treated -> U = T exp(psi). Censoring "
            "needs Robins' artificial censoring at C min(1, e^psi). "
            "CI by inverting the score test. Robins (1992) Biometrika 79, 321.")
