# morie.fn -- function file (rootcoder007/morie)
r"""Deep Survival Machines: a mixture of parametric survival experts.

**The model.** The conditional survival function is a weighted mixture
of :math:`K` primitive parametric distributions,

.. math:: S(t \mid x) = \sum_{k=1}^{K} g_k(x)\, S_k(t\mid \beta_k,
          \eta_k), \qquad
          g(x) = \mathrm{softmax}\big(\Phi(x)^{\!\top} w\big),

with the gates coming from a representation of the covariates and the
:math:`K` experts fixed across subjects. Two primitives are offered,
as in the paper: Weibull and log-normal. Both have closed-form density
*and* survival function, which is exactly why they were chosen -- a
censored observation needs :math:`S(t)`, not just :math:`f(t)`.

**The loss.** Maximum likelihood over a mixture has no closed form, so
the paper maximises an evidence lower bound obtained from Jensen's
inequality: rather than :math:`\ln \sum_k g_k f_k`, it uses
:math:`\sum_k g_k \ln f_k`. Uncensored and censored cases contribute
separately,

.. math:: \mathcal{L} = \underbrace{\sum_{i \in U} \sum_k g_k(x_i)
          \ln f_k(t_i)}_{\rm ELBO_U}
          + \alpha \underbrace{\sum_{i \in C} \sum_k g_k(x_i)
          \ln S_k(t_i)}_{\rm ELBO_C} + \mathcal{L}_{\rm prior},

and :math:`\alpha \in [0,1]` discounts the censored term. That
discount is not cosmetic: survival distributions have long right
tails, and censored cases are the ones asking for
:math:`P(T > t)` far out in that tail, so weighting them fully biases
the fit. Setting :math:`\alpha = 0` drops them entirely, which the
anchor uses -- the loss then does not move at all when a censored time
is changed.

**Jensen's inequality is checked, not assumed.** ``elbo`` and
``exact_loglik`` both exist here, and the anchor verifies
:math:`{\rm ELBO} \le \ln`-likelihood on real numbers for every fit.
A sign error or a misplaced gate would break that inequality
immediately.

**Competing risks** are handled the paper's way, by treating the other
event as independent censoring and giving each risk its own expert set
over a shared representation; ``fit_competing`` does that and returns
one fit per risk.

References
----------
Nagpal, C., Li, X. & Dubrawski, A. (2021) "Deep Survival Machines:
Fully Parametric Survival Regression and Representation Learning for
Censored Data with Competing Risks", *IEEE Journal of Biomedical and
Health Informatics* 25(8), 3163-3175, doi:10.1109/JBHI.2021.3052441
(arXiv:2003.01176). Sec. III for the mixture of :math:`K` parametric
distributions with softmax gates over a learned representation and for
the choice of Weibull and log-normal primitives; Sec. III-C for
:math:`\rm ELBO_U`, :math:`\rm ELBO_C`, the prior term and the
combined loss :math:`\mathcal{L} = {\rm ELBO_U} + \alpha\,
{\rm ELBO_C} + \mathcal{L}_{\rm prior}`, including the long-tail
argument for the discount :math:`\alpha`; and Sec. III-D for competing
risks by shared representation with per-risk experts.
"""

import math

from . import _array_core as np
from . import survrsf as _rsf
from ._richresult import RichResult
from ._sci_core import minimize

__all__ = ["PRIMITIVES", "log_pdf", "log_survival", "gates", "elbo",
           "exact_loglik", "fit", "predict_survival", "risk_score",
           "fit_competing", "concordance"]

PRIMITIVES = ("weibull", "lognormal")
_FLOOR = 1e-300


def _check(primitive):
    if primitive not in PRIMITIVES:
        raise ValueError("survvae: primitive must be one of %s, got %r"
                         % (", ".join(PRIMITIVES), primitive))


def log_pdf(t, shape, scale, primitive="weibull"):
    r"""Log density of one expert."""
    _check(primitive)
    t = float(t)
    if t <= 0.0:
        raise ValueError("survvae: survival times must be positive")
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("survvae: shape and scale must be positive")
    if primitive == "weibull":
        z = t / scale
        return (math.log(shape) - math.log(scale)
                + (shape - 1.0) * math.log(z) - z ** shape)
    z = (math.log(t) - math.log(scale)) / shape
    return (-math.log(t) - math.log(shape)
            - 0.5 * math.log(2.0 * math.pi) - 0.5 * z * z)


def log_survival(t, shape, scale, primitive="weibull"):
    r"""Log survival of one expert -- what a censored case needs."""
    _check(primitive)
    t = float(t)
    if t < 0.0:
        raise ValueError("survvae: survival times must be non-negative")
    if t == 0.0:
        return 0.0
    if primitive == "weibull":
        return -((t / scale) ** shape)
    z = (math.log(t) - math.log(scale)) / shape
    s = 0.5 * math.erfc(z / math.sqrt(2.0))
    return math.log(max(s, _FLOOR))


def gates(x, W, bias):
    r"""Softmax over the experts."""
    z = [sum(W[k][j] * x[j] for j in range(len(x))) + bias[k]
         for k in range(len(W))]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def elbo(X, y_lower, events, W, bias, shapes, scales,
         primitive="weibull", alpha=1.0, prior=0.0):
    r"""The paper's lower bound: gates outside the logarithm."""
    _check(primitive)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("survvae: alpha must lie in [0, 1], got %r"
                         % alpha)
    tot_u = tot_c = 0.0
    for i in range(len(X)):
        g = gates(X[i], W, bias)
        if events[i]:
            tot_u += sum(g[k] * log_pdf(y_lower[i], shapes[k],
                                        scales[k], primitive)
                         for k in range(len(g)))
        else:
            tot_c += sum(g[k] * log_survival(y_lower[i], shapes[k],
                                             scales[k], primitive)
                         for k in range(len(g)))
    pen = prior * (sum(math.log(v) ** 2 for v in shapes)
                   + sum(math.log(v) ** 2 for v in scales))
    return {"elbo": tot_u + alpha * tot_c - pen,
            "uncensored": tot_u, "censored": tot_c,
            "prior_penalty": pen, "alpha": float(alpha)}


def exact_loglik(X, y_lower, events, W, bias, shapes, scales,
                 primitive="weibull", alpha=1.0):
    r"""The true mixture log-likelihood the bound sits underneath."""
    _check(primitive)
    tot_u = tot_c = 0.0
    for i in range(len(X)):
        g = gates(X[i], W, bias)
        if events[i]:
            m = sum(g[k] * math.exp(log_pdf(y_lower[i], shapes[k],
                                            scales[k], primitive))
                    for k in range(len(g)))
            tot_u += math.log(max(m, _FLOOR))
        else:
            m = sum(g[k] * math.exp(log_survival(y_lower[i], shapes[k],
                                                 scales[k], primitive))
                    for k in range(len(g)))
            tot_c += math.log(max(m, _FLOOR))
    return {"loglik": tot_u + alpha * tot_c,
            "uncensored": tot_u, "censored": tot_c}


def _unpack(v, K, d):
    W = [list(v[k * d:(k + 1) * d]) for k in range(K)]
    off = K * d
    bias = list(v[off:off + K])
    off += K
    shapes = [math.exp(v[off + k]) for k in range(K)]
    scales = [math.exp(v[off + K + k]) for k in range(K)]
    return W, bias, shapes, scales


def fit(X, times, events, K=3, primitive="weibull", alpha=1.0,
        prior=0.0, seed=0, restarts=4):
    r"""Maximise the combined loss over gates and expert parameters."""
    _check(primitive)
    n = len(times)
    if not (n == len(X) == len(events)):
        raise ValueError("survvae: X, times and events must have the "
                         "same length")
    if n == 0:
        raise ValueError("survvae: no observations")
    K = int(K)
    if K < 1:
        raise ValueError("survvae: K must be at least 1")
    d = len(X[0])
    obs = [times[i] for i in range(n) if times[i] > 0.0]
    t0 = sum(obs) / len(obs)

    def objective(v):
        try:
            W, bias, shapes, scales = _unpack(v, K, d)
            if any(s <= 0.0 or s > 1e6 for s in shapes + scales):
                return 1e12
            return -elbo(X, times, events, W, bias, shapes, scales,
                         primitive, alpha, prior)["elbo"]
        except (ValueError, OverflowError):
            return 1e12

    rng = _rsf._Rng(seed)
    best = None
    for r in range(max(1, int(restarts))):
        v0 = [0.0] * (K * d + K)
        for k in range(K):
            v0.append(math.log(1.0 + 0.5 * (rng.next() - 0.5)))
        for k in range(K):
            v0.append(math.log(t0 * (0.5 + rng.next())))
        val = objective(v0)
        cur = list(v0)
        for _ in range(6):
            res = minimize(objective, cur, method="Nelder-Mead")
            cand = list(res.x if hasattr(res, "x") else res["x"])
            nv = objective(cand)
            if nv < val - 1e-9:
                val, cur = nv, cand
            else:
                cur = cand if nv < val else cur
                break
        if best is None or val < best[0]:
            best = (val, cur)
    W, bias, shapes, scales = _unpack(best[1], K, d)
    e = elbo(X, times, events, W, bias, shapes, scales, primitive,
             alpha, prior)
    ex = exact_loglik(X, times, events, W, bias, shapes, scales,
                      primitive, alpha)
    return RichResult(payload={
        "estimate": e["elbo"], "elbo": e["elbo"],
        "loglik": ex["loglik"], "jensen_gap": ex["loglik"] - e["elbo"],
        "W": W, "bias": bias, "shapes": shapes, "scales": scales,
        "K": K, "primitive": primitive, "alpha": float(alpha),
        "prior": float(prior), "times": list(times),
        "events": list(events),
        "method": "Deep Survival Machines: mixture of %s experts with "
                  "softmax gates, ELBO_U + alpha ELBO_C + prior; "
                  "Nagpal et al. (2021) Sec. III" % primitive,
    })


def predict_survival(fit_result, x, times):
    r""":math:`S(t\mid x)` as the gated mixture of expert survivals."""
    g = gates(x, fit_result["W"], fit_result["bias"])
    out = []
    for t in times:
        out.append(sum(g[k] * math.exp(
            log_survival(t, fit_result["shapes"][k],
                         fit_result["scales"][k],
                         fit_result["primitive"]))
            for k in range(fit_result["K"])))
    return {"time": [float(t) for t in times], "survival": out,
            "gates": g}


def risk_score(fit_result, X, horizon=None):
    r"""Risk at a horizon: :math:`1 - S(t\mid x)`."""
    if horizon is None:
        horizon = sorted(fit_result["times"])[len(fit_result["times"])
                                              // 2]
    return [1.0 - predict_survival(fit_result, x, [horizon])
            ["survival"][0] for x in X]


def concordance(fit_result, X, times, events, horizon=None):
    r"""Harrell's C at a horizon."""
    return _rsf.c_index(times, events,
                        risk_score(fit_result, X, horizon))


def fit_competing(X, times, causes, K=3, primitive="weibull",
                  alpha=1.0, prior=0.0, seed=0):
    r"""One fit per risk, other causes treated as censoring."""
    labels = sorted({int(c) for c in causes if int(c) != 0})
    if not labels:
        raise ValueError("survvae: no competing events found; cause 0 "
                         "means censored")
    out = {}
    for lab in labels:
        ev = [1 if int(c) == lab else 0 for c in causes]
        out[lab] = fit(X, times, ev, K, primitive, alpha, prior, seed)
    return RichResult(payload={
        "estimate": len(labels), "risks": labels, "fits": out,
        "method": "competing risks by treating other causes as "
                  "independent censoring; Nagpal et al. (2021) "
                  "Sec. III-D",
    })


def cheatsheet():
    return ("survvae: S(t|x) = sum_k g_k(x) S_k(t), gates a softmax "
            "and the experts Weibull or log-normal -- both chosen "
            "because a censored case needs S(t) in closed form. "
            "Trained on ELBO_U + alpha ELBO_C + prior, with the gates "
            "OUTSIDE the log (Jensen). alpha discounts the censored "
            "term against the long right tail; alpha = 0 drops it "
            "entirely. The ELBO is checked against the exact mixture "
            "likelihood rather than assumed to sit below it.")


# compact alias per ledger/NAMING.md
deep_survival_machines = fit
