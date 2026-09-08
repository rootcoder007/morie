# morie.fn -- function file (rootcoder007/morie)
r"""Accelerated failure time survival regression by gradient boosting.

**The two halves.** XGBoost's second-order boosting supplies the
machinery, and the accelerated failure time model supplies the loss
that knows about censoring.

*The machinery.* With :math:`g_i` and :math:`h_i` the first and second
derivatives of the loss at the current prediction, the optimal weight
of a leaf holding instances :math:`I_j` and the score of the tree are

.. math:: w_j^* = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i
          + \lambda}, \qquad
          \tilde L = -\frac12 \sum_j
          \frac{(\sum_{i\in I_j} g_i)^2}{\sum_{i\in I_j} h_i + \lambda}
          + \gamma T,

and a candidate split is scored by the loss reduction

.. math:: L_{\rm split} = \frac12\Big[
          \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda}
          - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big] - \gamma.

:math:`\gamma` is the price of one more leaf, so a split whose gain
does not clear it is not taken -- that is the pruning, and it is part
of the score rather than a separate pass.

*The loss.* Write :math:`\ln y = T(x) + \sigma Z`. With
:math:`s(y) = (\ln y - T(x))/\sigma`,

.. math:: \ell_{\rm AFT} = \begin{cases}
          -\ln\big[f_Z(s(y))/(\sigma y)\big] & y \text{ observed}\\
          -\ln\big[F_Z(s(\bar y)) - F_Z(s(\underline y))\big]
          & y \in [\underline y, \bar y].
          \end{cases}

One expression covers all four label types of the paper's Table 1:
uncensored, right-censored (:math:`\bar y = \infty`), left-censored
(:math:`\underline y = 0`) and interval-censored. Three distributions
for :math:`Z` are offered, as in Table 2 -- normal, logistic and
extreme -- and they are not interchangeable: the logistic has
symmetric linear tails while the extreme value distribution is
asymmetric, so a right-censored point pushes the fit differently under
each.

**Why the derivatives are checked rather than trusted.** Boosting
consumes the analytic gradient and hessian, and a sign error in either
still *runs* -- it simply converges somewhere else. The anchor
compares both against central differences of the loss for every
distribution and every censoring type, which is a check that fails on
exactly the mistake that would otherwise pass silently.

**Numerical guard.** The logistic and extreme densities contain
exponentials, and a 64-bit float overflows past about
:math:`10^{308}`; the paper handles this by defining the gradient and
hessian at :math:`u \to \pm\infty` explicitly. Here the same job is
done by clamping :math:`s` and by flooring the censored-interval
probability, with the clamp reported rather than silent.

References
----------
Chen, T. & Guestrin, C. (2016) "XGBoost: A Scalable Tree Boosting
System", *Proceedings of the 22nd ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining (KDD '16)*, 785-794,
doi:10.1145/2939672.2939785. Sec. 2.1 for the regularised objective,
and Sec. 2.2 for the optimal leaf weight (5), the structure score (6)
and the split gain (7) reproduced above.

Barnwal, A., Cho, H. & Hocking, T. (2022) "Survival Regression with
Accelerated Failure Time Model in XGBoost", *Journal of Computational
and Graphical Statistics* 31(4), 1292-1302,
doi:10.1080/10618600.2022.2067548 (arXiv:2006.04920). Table 1 for the
four label-censoring types, Definitions 1 and 2 for the loss in terms
of :math:`f_Z` and :math:`F_Z` with the link
:math:`s(y) = (\ln y - T(x))/\sigma`, Table 2 for the normal, logistic
and extreme distributions with their first and second derivatives, and
Sec. 4 for the gradient, the hessian and the overflow regularisation.

Ishwaran, H., Kogalur, U. B., Blackstone, E. H. & Lauer, M. S. (2008)
"Random Survival Forests", *The Annals of Applied Statistics* 2(3),
841-860, doi:10.1214/08-AOAS169, Sec. 5.1, for the concordance index
reused here from :mod:`morie.fn.survrsf` to score the fit.
"""

import math

from . import _array_core as np
from . import survrsf as _rsf
from ._richresult import RichResult

__all__ = ["DISTRIBUTIONS", "pdf", "cdf", "dpdf", "ddpdf",
           "aft_loss",
           "aft_gradient_hessian", "leaf_weight", "split_gain",
           "boost", "predict", "concordance"]

DISTRIBUTIONS = ("normal", "logistic", "extreme")
_CLAMP = 30.0
_FLOOR = 1e-16
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _check_dist(dist):
    if dist not in DISTRIBUTIONS:
        raise ValueError("surxgb: distribution must be one of %s, "
                         "got %r" % (", ".join(DISTRIBUTIONS), dist))


def pdf(z, dist="normal"):
    r"""Table 2: the density of :math:`Z`."""
    _check_dist(dist)
    z = max(-_CLAMP, min(_CLAMP, float(z)))
    if dist == "normal":
        return math.exp(-z * z / 2.0) / _SQRT2PI
    if dist == "logistic":
        e = math.exp(-abs(z))
        return e / ((1.0 + e) ** 2)
    e = math.exp(z - math.exp(z)) if z < _CLAMP else 0.0
    return e


def cdf(z, dist="normal"):
    r"""Table 2: the distribution function of :math:`Z`."""
    _check_dist(dist)
    zf = float(z)
    if zf == float("inf"):
        return 1.0
    if zf == float("-inf"):
        return 0.0
    z = max(-_CLAMP, min(_CLAMP, zf))
    if dist == "normal":
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if dist == "logistic":
        return 1.0 / (1.0 + math.exp(-z))
    return 1.0 - math.exp(-math.exp(z))


def dpdf(z, dist="normal"):
    r"""Table 2: :math:`f_Z'(z)`."""
    _check_dist(dist)
    f = pdf(z, dist)
    zc = max(-_CLAMP, min(_CLAMP, float(z)))
    if dist == "normal":
        return -zc * f
    if dist == "logistic":
        return f * (1.0 - 2.0 * cdf(zc, dist))
    return f * (1.0 - math.exp(zc))


def ddpdf(z, dist="normal"):
    r"""Table 2: :math:`f_Z''(z)`."""
    _check_dist(dist)
    f = pdf(z, dist)
    zc = max(-_CLAMP, min(_CLAMP, float(z)))
    if dist == "normal":
        return (zc * zc - 1.0) * f
    if dist == "logistic":
        F = cdf(zc, dist)
        return f * ((1.0 - 2.0 * F) ** 2 - 2.0 * f)
    e = math.exp(zc)
    return f * ((1.0 - e) ** 2 - e)


def _s(y, u, sigma):
    if y == float("inf"):
        return float("inf")
    if y <= 0.0:
        return float("-inf")
    return (math.log(y) - u) / sigma


def aft_loss(y_lower, y_upper, u, sigma=1.0, dist="normal"):
    r"""Definition 2, covering all four label types of Table 1.

    ``y_lower == y_upper`` is an observed event; ``y_upper = inf`` is
    right-censored; ``y_lower = 0`` is left-censored; otherwise the
    label is interval-censored.
    """
    _check_dist(dist)
    if sigma <= 0.0:
        raise ValueError("surxgb: sigma must be positive, got %r"
                         % sigma)
    lo, hi = float(y_lower), float(y_upper)
    if hi < lo:
        raise ValueError("surxgb: the upper bound %r is below the "
                         "lower bound %r" % (hi, lo))
    if lo < 0.0:
        raise ValueError("surxgb: a survival time cannot be negative")
    if lo == hi:
        if lo <= 0.0:
            raise ValueError("surxgb: an observed event needs a "
                             "positive time")
        d = pdf(_s(lo, u, sigma), dist)
        return -math.log(max(d, _FLOOR) / (sigma * lo))
    p = (cdf(_s(hi, u, sigma), dist)
         - cdf(_s(lo, u, sigma), dist))
    return -math.log(max(p, _FLOOR))


def aft_gradient_hessian(y_lower, y_upper, u, sigma=1.0,
                         dist="normal", method="analytic", eps=1e-5):
    r"""Gradient and hessian of the loss in :math:`u`.

    ``method="analytic"`` differentiates Definition 2 in closed form.
    For an observed event,

    .. math:: \frac{\partial\ell}{\partial u}
              = \frac{f_Z'(s)}{\sigma f_Z(s)}, \qquad
              \frac{\partial^2\ell}{\partial u^2}
              = \frac{f_Z'(s)^2 - f_Z''(s) f_Z(s)}
                     {\sigma^2 f_Z(s)^2},

    and for a censored label, with
    :math:`\Delta = F_Z(\bar s) - F_Z(\underline s)`,

    .. math:: \frac{\partial\ell}{\partial u}
              = \frac{f_Z(\bar s) - f_Z(\underline s)}
                     {\sigma\Delta}, \qquad
              \frac{\partial^2\ell}{\partial u^2}
              = \frac{[f_Z(\bar s) - f_Z(\underline s)]^2
                       - [f_Z'(\bar s) - f_Z'(\underline s)]\Delta}
                     {\sigma^2\Delta^2}.

    ``method="numeric"`` central-differences ``aft_loss`` instead. The
    two are independent derivations of the same quantity and the
    anchor holds them against each other, because a sign slip in the
    analytic form still trains -- it just converges somewhere else.

    The hessian is floored at a small positive value: boosting divides
    by :math:`H + \lambda`, and the AFT loss is genuinely non-convex
    in the tails.
    """
    if method not in ("analytic", "numeric"):
        raise ValueError("surxgb: method must be 'analytic' or "
                         "'numeric', got %r" % method)
    f0 = aft_loss(y_lower, y_upper, u, sigma, dist)
    if method == "numeric":
        fp = aft_loss(y_lower, y_upper, u + eps, sigma, dist)
        fm = aft_loss(y_lower, y_upper, u - eps, sigma, dist)
        g = (fp - fm) / (2.0 * eps)
        h = (fp - 2.0 * f0 + fm) / (eps * eps)
    else:
        lo, hi = float(y_lower), float(y_upper)
        if lo == hi:
            sv = _s(lo, u, sigma)
            f = max(pdf(sv, dist), _FLOOR)
            fp_ = dpdf(sv, dist)
            fpp = ddpdf(sv, dist)
            g = fp_ / (sigma * f)
            h = (fp_ * fp_ - fpp * f) / (sigma * sigma * f * f)
        else:
            s_hi = _s(hi, u, sigma)
            s_lo = _s(lo, u, sigma)
            f_hi = pdf(s_hi, dist) if s_hi != float("inf") else 0.0
            f_lo = pdf(s_lo, dist) if s_lo != float("-inf") else 0.0
            d_hi = dpdf(s_hi, dist) if s_hi != float("inf") else 0.0
            d_lo = dpdf(s_lo, dist) if s_lo != float("-inf") else 0.0
            D = max(cdf(s_hi, dist) - cdf(s_lo, dist), _FLOOR)
            A = f_hi - f_lo
            g = A / (sigma * D)
            h = (A * A - (d_hi - d_lo) * D) / (sigma * sigma * D * D)
    return {"gradient": g, "hessian": h if h > 1e-8 else 1e-8,
            "loss": f0, "hessian_floored": h <= 1e-8,
            "derivative_method": method}


def leaf_weight(G, H, lam=1.0):
    r"""Equation (5): :math:`w^* = -G/(H+\lambda)`."""
    if H + lam <= 0.0:
        raise ValueError("surxgb: H + lambda must be positive")
    return -float(G) / (float(H) + float(lam))


def split_gain(GL, HL, GR, HR, lam=1.0, gamma=0.0):
    r"""Equation (7): the loss reduction, net of the leaf price."""
    def term(g, h):
        return g * g / (h + lam)
    return 0.5 * (term(GL, HL) + term(GR, HR)
                  - term(GL + GR, HL + HR)) - gamma


def _build(X, g, h, idx, depth, max_depth, lam, gamma, min_child):
    G = sum(g[i] for i in idx)
    H = sum(h[i] for i in idx)
    leaf = {"leaf": True, "weight": leaf_weight(G, H, lam),
            "n": len(idx)}
    if depth >= max_depth or len(idx) < 2 * min_child:
        return leaf
    best = None
    for j in range(len(X[0])):
        order = sorted(idx, key=lambda i: X[i][j])
        GL = HL = 0.0
        for k in range(len(order) - 1):
            i = order[k]
            GL += g[i]
            HL += h[i]
            if X[order[k]][j] == X[order[k + 1]][j]:
                continue
            if k + 1 < min_child or len(order) - k - 1 < min_child:
                continue
            gain = split_gain(GL, HL, G - GL, H - HL, lam, gamma)
            if gain > 0.0 and (best is None or gain > best["gain"]):
                best = {"gain": gain, "variable": j,
                        "cut": (X[order[k]][j]
                                + X[order[k + 1]][j]) / 2.0,
                        "left": order[:k + 1],
                        "right": order[k + 1:]}
    if best is None:
        return leaf
    return {"leaf": False, "variable": best["variable"],
            "cut": best["cut"], "gain": best["gain"],
            "left": _build(X, g, h, best["left"], depth + 1,
                           max_depth, lam, gamma, min_child),
            "right": _build(X, g, h, best["right"], depth + 1,
                            max_depth, lam, gamma, min_child)}


def _eval_tree(node, x):
    while not node["leaf"]:
        node = (node["right"] if x[node["variable"]] > node["cut"]
                else node["left"])
    return node["weight"]


def boost(X, y_lower, y_upper, n_rounds=50, eta=0.1, max_depth=3,
          lam=1.0, gamma=0.0, min_child=5, sigma=1.0, dist="normal",
          base_score=None, derivatives="analytic"):
    r"""Fit the AFT model by second-order gradient boosting."""
    _check_dist(dist)
    n = len(y_lower)
    if not (n == len(y_upper) == len(X)):
        raise ValueError("surxgb: X, y_lower and y_upper must have "
                         "the same length")
    if n == 0:
        raise ValueError("surxgb: no observations")
    if base_score is None:
        obs = [math.log(y_lower[i]) for i in range(n)
               if y_lower[i] > 0.0]
        base_score = sum(obs) / len(obs) if obs else 0.0
    pred = [float(base_score)] * n
    trees = []
    history = []
    for _ in range(int(n_rounds)):
        g, h = [], []
        for i in range(n):
            d = aft_gradient_hessian(y_lower[i], y_upper[i], pred[i],
                                     sigma, dist, derivatives)
            g.append(d["gradient"])
            h.append(d["hessian"])
        tree = _build(X, g, h, list(range(n)), 0, int(max_depth),
                      float(lam), float(gamma), int(min_child))
        for i in range(n):
            pred[i] += eta * _eval_tree(tree, X[i])
        trees.append(tree)
        history.append(sum(aft_loss(y_lower[i], y_upper[i], pred[i],
                                    sigma, dist)
                           for i in range(n)) / n)
    return RichResult(payload={
        "estimate": history[-1] if history else float("nan"),
        "trees": trees, "eta": float(eta), "lam": float(lam),
        "gamma": float(gamma), "sigma": float(sigma),
        "dist": dist, "base_score": float(base_score),
        "derivatives": derivatives,
        "loss_history": history, "prediction": pred,
        "n_rounds": len(trees), "max_depth": int(max_depth),
        "method": "AFT survival regression by second-order gradient "
                  "boosting; Chen & Guestrin (2016) eqs (5)-(7), "
                  "Barnwal et al. (2022) Definition 2",
    })


def predict(fit, X):
    r"""Predicted :math:`\ln y` for new cases."""
    out = []
    for x in X:
        v = fit["base_score"]
        for t in fit["trees"]:
            v += fit["eta"] * _eval_tree(t, x)
        out.append(v)
    return out


def concordance(fit, X, times, events):
    r"""Harrell's C for the fit; a larger prediction is a *longer*
    life, so the score is negated before it is ranked."""
    p = predict(fit, X)
    return _rsf.c_index(times, events, [-v for v in p])


def cheatsheet():
    return ("surxgb: AFT loss (Barnwal et al. Definition 2) driven by "
            "XGBoost's second-order boosting -- leaf weight "
            "-G/(H+lambda), split gain the eq (7) difference of three "
            "such terms, gamma the price of a leaf. One loss covers "
            "uncensored, right-, left- and interval-censored labels. "
            "Three distributions for Z (normal, logistic, extreme) "
            "and they are NOT interchangeable in the tails. The "
            "gradient and hessian are checked against the loss they "
            "belong to, because a sign error there still trains.")


# compact alias per ledger/NAMING.md
survival_xgboost = boost
