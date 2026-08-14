# morie.fn -- function file (rootcoder007/morie)
r"""DeepSurv: a neural network trained on the Cox partial likelihood.

**The model.** Cox's proportional hazards writes
:math:`\lambda(t\mid x) = \lambda_0(t)\,e^{h(x)}`. Cox regression takes
:math:`h(x) = \beta'x`; DeepSurv replaces that linear term with the
single-node output of a multilayer perceptron and keeps everything
else, so the training objective is the average negative log partial
likelihood over the *uncensored* cases,

.. math:: \ell(\theta) = -\frac{1}{N_{E=1}} \sum_{i: E_i = 1}
          \Big( \hat h_\theta(x_i)
          - \log \sum_{j \in \Re(T_i)} e^{\hat h_\theta(x_j)} \Big)
          + \lambda \lVert\theta\rVert_2^2 ,

with :math:`\Re(t) = \{i : T_i \ge t\}` the risk set.

**Why the linear case is the anchor.** Strip the hidden layers and
DeepSurv *is* Cox regression, so the network must land on the
coefficients that :mod:`morie.fn.coxph` finds by Newton-Raphson. That
comparison fails on a wrong risk set, a wrong sign in the gradient, or
a mis-scaled learning rate, and it does not depend on any number I
chose myself.

**What the partial likelihood does not give you.** It never estimates
:math:`\lambda_0`, so the network output is a log-risk on an arbitrary
scale: only *differences* between subjects mean anything, and adding a
constant to every prediction leaves the loss unchanged. ``fit``
reports that invariance rather than hiding it, and ``baseline_hazard``
supplies Breslow's estimator when an absolute survival curve is
actually wanted.

**Ties.** Breslow's handling is used -- every tied event contributes
the same risk-set denominator. Efron's correction is not implemented;
with heavy tying the two differ, so the choice is reported in the
result.

References
----------
Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T. &
Kluger, Y. (2018) "DeepSurv: personalized treatment recommender system
using a Cox proportional hazards deep neural network", *BMC Medical
Research Methodology* 18, 24, doi:10.1186/s12874-018-0482-1. Sec.
"DeepSurv" for the architecture -- fully connected hidden layers, a
single linear output node estimating the log-risk -- and for the
objective, the average negative log partial likelihood of Eq. 3 with
:math:`\ell_2` regularisation, reproduced above.

Cox, D. R. (1972) "Regression Models and Life-Tables", *Journal of the
Royal Statistical Society. Series B* 34(2), 187-220,
doi:10.1111/j.2517-6161.1972.tb00899.x, for the partial likelihood
itself and for the fact that it leaves the baseline hazard
unspecified.

Breslow, N. (1974) "Covariance Analysis of Censored Survival Data",
*Biometrics* 30(1), 89-99, doi:10.2307/2529620, for the tie handling
and the baseline cumulative hazard estimator used here.
"""

import math

from . import _array_core as np
from . import survrsf as _rsf
from ._richresult import RichResult

__all__ = ["partial_loglik", "forward", "fit", "risk_score",
           "baseline_hazard", "survival_function", "concordance"]

ACTIVATIONS = ("tanh", "relu", "identity")


def _act(v, kind):
    if kind == "tanh":
        return math.tanh(v)
    if kind == "relu":
        return v if v > 0.0 else 0.0
    return v


def _dact(v, kind):
    if kind == "tanh":
        t = math.tanh(v)
        return 1.0 - t * t
    if kind == "relu":
        return 1.0 if v > 0.0 else 0.0
    return 1.0


def _init(d, hidden, seed):
    rng = _rsf._Rng(seed)
    sizes = [d] + [int(h) for h in hidden] + [1]
    W, b = [], []
    for k in range(len(sizes) - 1):
        scale = math.sqrt(2.0 / sizes[k])
        W.append([[(rng.next() - 0.5) * 2.0 * scale
                   for _ in range(sizes[k])]
                  for _ in range(sizes[k + 1])])
        b.append([0.0] * sizes[k + 1])
    return W, b


def forward(W, b, x, activation="tanh"):
    r"""Run one case through the network, keeping the activations."""
    a = [float(v) for v in x]
    pre, acts = [], [a]
    for k in range(len(W)):
        z = [sum(W[k][i][j] * a[j] for j in range(len(a))) + b[k][i]
             for i in range(len(W[k]))]
        pre.append(z)
        a = z if k == len(W) - 1 else [_act(v, activation) for v in z]
        acts.append(a)
    return {"output": a[0], "pre": pre, "acts": acts}


def partial_loglik(times, events, risk):
    r"""Cox's log partial likelihood with Breslow's tie handling."""
    n = len(times)
    if not (n == len(events) == len(risk)):
        raise ValueError("survnnr: times, events and risks must have "
                         "the same length")
    n_events = sum(1 for e in events if e)
    if n_events == 0:
        raise ValueError("survnnr: the partial likelihood needs at "
                         "least one event")
    order = sorted(range(n), key=lambda i: times[i])
    total = 0.0
    for i in range(n):
        if not events[i]:
            continue
        at_risk = [j for j in order if times[j] >= times[i]]
        m = max(risk[j] for j in at_risk)
        lse = m + math.log(sum(math.exp(risk[j] - m)
                               for j in at_risk))
        total += risk[i] - lse
    return {"loglik": total, "average": total / n_events,
            "n_events": n_events, "ties": "Breslow"}


def _grad_wrt_risk(times, events, risk):
    """d(-average partial loglik)/d risk_k, exactly."""
    n = len(times)
    n_events = float(sum(1 for e in events if e))
    g = [0.0] * n
    for i in range(n):
        if not events[i]:
            continue
        at_risk = [j for j in range(n) if times[j] >= times[i]]
        m = max(risk[j] for j in at_risk)
        w = [math.exp(risk[j] - m) for j in at_risk]
        s = sum(w)
        g[i] -= 1.0 / n_events
        for k, j in enumerate(at_risk):
            g[j] += (w[k] / s) / n_events
    return g


def fit(X, times, events, hidden=(), activation="tanh", l2=0.0,
        lr=0.1, n_epochs=400, seed=0, tol=1e-10):
    r"""Train the network on the average negative log partial
    likelihood.

    ``hidden=()`` is Cox regression by gradient descent, which is what
    the anchor holds against :mod:`morie.fn.coxph`.
    """
    if activation not in ACTIVATIONS:
        raise ValueError("survnnr: activation must be one of %s, got "
                         "%r" % (", ".join(ACTIVATIONS), activation))
    n = len(times)
    if n != len(X) or n != len(events):
        raise ValueError("survnnr: X, times and events must have the "
                         "same length")
    if n == 0:
        raise ValueError("survnnr: no observations")
    d = len(X[0])
    W, b = _init(d, hidden, seed)
    history = []
    prev = None
    for _ in range(int(n_epochs)):
        fwd = [forward(W, b, x, activation) for x in X]
        risk = [f["output"] for f in fwd]
        loss = -partial_loglik(times, events, risk)["average"]
        loss += l2 * sum(v * v for M in W for row in M for v in row)
        history.append(loss)
        if prev is not None and abs(prev - loss) < tol:
            break
        prev = loss
        gr = _grad_wrt_risk(times, events, risk)
        gW = [[[0.0] * len(W[k][0]) for _ in W[k]]
              for k in range(len(W))]
        gb = [[0.0] * len(W[k]) for k in range(len(W))]
        for i in range(n):
            delta = [gr[i]]
            for k in range(len(W) - 1, -1, -1):
                acts = fwd[i]["acts"][k]
                for r in range(len(W[k])):
                    gb[k][r] += delta[r]
                    for c in range(len(acts)):
                        gW[k][r][c] += delta[r] * acts[c]
                if k:
                    pre = fwd[i]["pre"][k - 1]
                    delta = [sum(W[k][r][c] * delta[r]
                                 for r in range(len(W[k])))
                             * _dact(pre[c], activation)
                             for c in range(len(pre))]
        for k in range(len(W)):
            for r in range(len(W[k])):
                b[k][r] -= lr * gb[k][r]
                for c in range(len(W[k][r])):
                    W[k][r][c] -= lr * (gW[k][r][c]
                                        + 2.0 * l2 * W[k][r][c])
    risk = [forward(W, b, x, activation)["output"] for x in X]
    mean_risk = sum(risk) / len(risk)
    return RichResult(payload={
        "estimate": history[-1] if history else float("nan"),
        "W": W, "b": b, "activation": activation, "hidden": tuple(hidden),
        "l2": float(l2), "loss_history": history,
        "risk": risk, "centred_risk": [v - mean_risk for v in risk],
        "coefficients": [W[0][0][j] for j in range(d)]
        if not hidden else None,
        "times": list(times), "events": list(events),
        "epochs": len(history), "ties": "Breslow",
        "scale_note": "the partial likelihood is invariant to adding "
                      "a constant to every risk; only differences are "
                      "identified",
        "method": "DeepSurv: MLP trained on the average negative log "
                  "partial likelihood; Katzman et al. (2018) Eq. 4",
    })


def risk_score(fit_result, X):
    r"""Log-risk for new cases."""
    return [forward(fit_result["W"], fit_result["b"], x,
                    fit_result["activation"])["output"] for x in X]


def baseline_hazard(fit_result):
    r"""Breslow's cumulative baseline hazard at the event times."""
    t = fit_result["times"]
    e = fit_result["events"]
    r = [math.exp(v) for v in fit_result["risk"]]
    order = sorted(range(len(t)), key=lambda i: t[i])
    out_t, out_h = [], []
    cum = 0.0
    seen = set()
    for i in order:
        if not e[i] or t[i] in seen:
            continue
        seen.add(t[i])
        d = sum(1 for j in range(len(t)) if t[j] == t[i] and e[j])
        denom = sum(r[j] for j in range(len(t)) if t[j] >= t[i])
        cum += d / denom
        out_t.append(float(t[i]))
        out_h.append(cum)
    return {"time": out_t, "cumulative_hazard": out_h}


def survival_function(fit_result, x, times=None):
    r""":math:`S(t\mid x) = \exp(-H_0(t) e^{h(x)})`."""
    base = baseline_hazard(fit_result)
    r = math.exp(risk_score(fit_result, [x])[0])
    ts = base["time"] if times is None else [float(v) for v in times]
    out = []
    for t in ts:
        h = 0.0
        for k, bt in enumerate(base["time"]):
            if bt <= t:
                h = base["cumulative_hazard"][k]
            else:
                break
        out.append(math.exp(-h * r))
    return {"time": ts, "survival": out}


def concordance(fit_result, X, times, events):
    r"""Harrell's C; a larger log-risk means a shorter life."""
    return _rsf.c_index(times, events, risk_score(fit_result, X))


def cheatsheet():
    return ("survnnr: DeepSurv = Cox's partial likelihood with the "
            "linear predictor replaced by an MLP output. Loss is the "
            "AVERAGE negative log partial likelihood over events, "
            "Breslow ties. With no hidden layer it is Cox regression "
            "and must reproduce coxph. The baseline hazard is never "
            "estimated by the likelihood, so risks are identified "
            "only up to an additive constant; Breslow's estimator "
            "supplies it when an absolute survival curve is wanted.")


# compact alias per ledger/NAMING.md
deep_surv = fit
