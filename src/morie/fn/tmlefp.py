r"""Optimal overlap: which subpopulation is estimable, and how to weight it.

Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009) "Dealing
with limited overlap in estimation of average treatment effects",
*Biometrika* 96(1), 187-199.

When propensity scores approach 0 or 1 the average treatment effect over the
whole population is barely estimable: its semiparametric variance bound blows
up. The paper's answer is to change the estimand rather than the estimator --
report the effect for the subpopulation that *can* be estimated precisely, and
say which subpopulation that is.

**Optimal subpopulation (Theorem 5.2).** With
:math:`k(x) = \sigma_1^2(x)/e(x) + \sigma_0^2(x)/(1 - e(x))`, the variance-
minimising set is all of :math:`\mathcal{X}` when
:math:`\sup_x k(x) \le 2\,\mathbb{E}[k(X)]`, and otherwise

.. math:: A^* = \Big\{x : k(x) \le \tfrac{1}{\alpha(1-\alpha)}\Big\},
          \qquad
          \tfrac{1}{\alpha(1-\alpha)} =
          2\,\mathbb{E}\Big[k(X) \,\Big|\, k(X) <
          \tfrac{1}{\alpha(1-\alpha)}\Big].

Under homoskedasticity :math:`k(x) \propto 1/(e(x)(1-e(x)))` and the rule
collapses to the one the abstract advertises (Corollary 5.1):

.. math:: A^*_H = \{x : \alpha \le e(x) \le 1 - \alpha\}.

**For the treated (Theorem 5.3, homoskedastic only, as in the paper)** the set
is one-sided, :math:`A^*_t = \{x : e(x) \le \alpha_t\}`, with
:math:`\alpha_t = 1` when
:math:`\sup_x 1/(1-e(x)) \le 2\,\mathbb{E}[1/(1-e(X)) \mid W = 1]` and
otherwise solving
:math:`1/(1-\alpha_t) = 2\,\mathbb{E}[1/(1-e(X)) \mid W=1, e(X) \le \alpha_t]`.

**Optimal weights (Theorem 5.4).** Rather than an indicator, weight by
:math:`\omega^*(x) = (\sigma_1^2(x)/e(x) + \sigma_0^2(x)/(1-e(x)))^{-1}`,
which under homoskedasticity is :math:`\omega^*_H(x) = e(x)(1 - e(x))`
(Corollary 5.2) -- the OWATE. Both estimands are returned, because they answer
different questions: the OSATE is an average effect over a named
subpopulation, the OWATE is a precisely estimated weighted average whose
weights are not an indicator and whose population is therefore harder to
describe.

The feasible rule of section 6 inverts the threshold:
:math:`1/(\alpha(1-\alpha)) = \gamma` gives
:math:`\alpha = \tfrac12(1 - \sqrt{1 - 4/\gamma})`, defined only for
:math:`\gamma \ge 4`, which is the algebraic statement of "no trimming is
optimal unless the variance is at least four times its best possible value".

What is estimated here, given propensity scores and outcomes, is the
subpopulation effect itself, by the normalised inverse-probability weighting
that Theorem 6.1's variance refers to; the paper's own asymptotic variance is
reported alongside. Estimating the propensity score is the caller's job --
the paper's results condition on it -- and doubly robust or targeted
estimation is out of scope here.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["tmlefp", "optimal_overlap", "tmle_effective_pi", "optimal_alpha", "optimal_alpha_att",
           "owate_weights", "alpha_from_gamma"]


def alpha_from_gamma(gamma):
    r""":math:`\alpha = \tfrac12(1 - \sqrt{1 - 4/\gamma})` for
    :math:`1/(\alpha(1-\alpha)) = \gamma`.

    Undefined below :math:`\gamma = 4`, where the quadratic has no real
    root: :math:`1/(\alpha(1-\alpha)) \ge 4` for every
    :math:`\alpha \in (0, 1/2]`.
    """
    gamma = float(gamma)
    if gamma < 4.0:
        raise ValueError("tmlefp: gamma must be at least 4; below that the "
                         "threshold 1/(alpha(1-alpha)) = gamma has no root "
                         "in (0, 1/2]")
    # (1 - sqrt(1 - 4/g))/2 loses every digit for large g, where the two
    # terms agree to 1e-16; multiply through by the conjugate instead.
    root = math.sqrt(1.0 - 4.0 / gamma)
    return 2.0 / (gamma * (1.0 + root))


def optimal_alpha(pscore, sigma2_treated=None, sigma2_control=None,
                  tol=1e-12, max_iter=200):
    r"""Theorem 5.2: the variance-minimising trimming threshold.

    Solves :math:`\gamma = 2\,\mathbb{E}[k(X) \mid k(X) < \gamma]` for
    :math:`\gamma = 1/(\alpha(1-\alpha))` by the fixed-point iteration the
    equation itself defines, starting from the no-trimming point. With
    homoskedastic variances (the default) :math:`k(x) = 1/(e(1-e))`.

    Returns ``{"alpha", "gamma", "keep", "trim", "no_trimming",
    "k"}`` -- ``keep`` is the boolean membership of :math:`A^*`.
    """
    e = [float(v) for v in pscore]
    n = len(e)
    if n == 0:
        raise ValueError("tmlefp: no propensity scores")
    if any(not 0.0 < v < 1.0 for v in e):
        raise ValueError("tmlefp: propensity scores must lie strictly in "
                         "(0, 1)")
    if sigma2_treated is None and sigma2_control is None:
        k = [1.0 / (v * (1.0 - v)) for v in e]
    else:
        s1 = ([1.0] * n if sigma2_treated is None
              else [float(v) for v in sigma2_treated])
        s0 = ([1.0] * n if sigma2_control is None
              else [float(v) for v in sigma2_control])
        if len(s1) != n or len(s0) != n:
            raise ValueError("tmlefp: one conditional variance per unit")
        if any(v <= 0 for v in s1 + s0):
            raise ValueError("tmlefp: conditional variances must be "
                             "positive")
        k = [s1[i] / e[i] + s0[i] / (1.0 - e[i]) for i in range(n)]

    mean_k = sum(k) / n
    if max(k) <= 2.0 * mean_k:
        return {"alpha": 0.0, "gamma": float("inf"),
                "keep": [True] * n, "trim": 0, "no_trimming": True, "k": k}

    gamma = 2.0 * mean_k
    for _ in range(int(max_iter)):
        sel = [v for v in k if v < gamma]
        if not sel:
            raise ValueError("tmlefp: the fixed point excluded every unit; "
                             "check the propensity scores")
        new = 2.0 * sum(sel) / len(sel)
        if abs(new - gamma) < tol * max(1.0, abs(gamma)):
            gamma = new
            break
        gamma = new
    keep = [v <= gamma for v in k]
    homosk = sigma2_treated is None and sigma2_control is None
    alpha = alpha_from_gamma(gamma) if homosk else float("nan")
    return {"alpha": alpha, "gamma": gamma, "keep": keep,
            "trim": n - sum(1 for v in keep if v), "no_trimming": False,
            "k": k}


def optimal_alpha_att(pscore, treated, tol=1e-12, max_iter=200):
    r"""Theorem 5.3: the one-sided threshold for the effect on the treated.

    :math:`A^*_t = \{x : e(x) \le \alpha_t\}`, with :math:`\alpha_t = 1`
    (no trimming) when
    :math:`\sup_x 1/(1-e(x)) \le 2\mathbb{E}[1/(1-e(X)) \mid W=1]`, and
    otherwise solving
    :math:`1/(1-\alpha_t) = 2\mathbb{E}[1/(1-e(X)) \mid W=1,\,
    e(X) \le \alpha_t]`. Homoskedasticity only, as in the paper.
    """
    e = [float(v) for v in pscore]
    w = [int(v) for v in treated]
    n = len(e)
    if n != len(w):
        raise ValueError("tmlefp: one treatment indicator per unit")
    if any(not 0.0 < v < 1.0 for v in e):
        raise ValueError("tmlefp: propensity scores must lie strictly in "
                         "(0, 1)")
    idx = [i for i in range(n) if w[i] == 1]
    if not idx:
        raise ValueError("tmlefp: no treated units")
    g = [1.0 / (1.0 - e[i]) for i in idx]
    if max(1.0 / (1.0 - v) for v in e) <= 2.0 * sum(g) / len(g):
        return {"alpha_t": 1.0, "keep": [True] * n, "trim": 0,
                "no_trimming": True}
    thr = 2.0 * sum(g) / len(g)
    for _ in range(int(max_iter)):
        sel = [1.0 / (1.0 - e[i]) for i in idx
               if 1.0 / (1.0 - e[i]) < thr]
        if not sel:
            raise ValueError("tmlefp: the fixed point excluded every "
                             "treated unit")
        new = 2.0 * sum(sel) / len(sel)
        if abs(new - thr) < tol * max(1.0, abs(thr)):
            thr = new
            break
        thr = new
    alpha_t = 1.0 - 1.0 / thr
    return {"alpha_t": alpha_t, "keep": [v <= alpha_t for v in e],
            "trim": sum(1 for v in e if v > alpha_t),
            "no_trimming": False}


def owate_weights(pscore, sigma2_treated=None, sigma2_control=None):
    r"""Theorem 5.4 / Corollary 5.2:
    :math:`\omega^*(x) = (\sigma_1^2/e + \sigma_0^2/(1-e))^{-1}`, which is
    :math:`e(x)(1-e(x))` under homoskedasticity."""
    e = [float(v) for v in pscore]
    if any(not 0.0 < v < 1.0 for v in e):
        raise ValueError("tmlefp: propensity scores must lie strictly in "
                         "(0, 1)")
    if sigma2_treated is None and sigma2_control is None:
        return [v * (1.0 - v) for v in e]
    n = len(e)
    s1 = [1.0] * n if sigma2_treated is None else [float(v) for v in
                                                   sigma2_treated]
    s0 = [1.0] * n if sigma2_control is None else [float(v) for v in
                                                   sigma2_control]
    return [1.0 / (s1[i] / e[i] + s0[i] / (1.0 - e[i])) for i in range(n)]


def _ipw(y, w, e, keep=None, weights=None):
    """Normalised IPW effect over a subpopulation or with given weights."""
    n = len(y)
    sel = range(n) if keep is None else [i for i in range(n) if keep[i]]
    if not sel:
        raise ValueError("tmlefp: the selected subpopulation is empty")
    om = [1.0] * n if weights is None else weights
    num1 = sum(om[i] * w[i] * y[i] / e[i] for i in sel)
    den1 = sum(om[i] * w[i] / e[i] for i in sel)
    num0 = sum(om[i] * (1 - w[i]) * y[i] / (1.0 - e[i]) for i in sel)
    den0 = sum(om[i] * (1 - w[i]) / (1.0 - e[i]) for i in sel)
    if den1 <= 0 or den0 <= 0:
        raise ValueError("tmlefp: the subpopulation has no treated or no "
                         "control units")
    return num1 / den1 - num0 / den0, len(list(sel))


def tmlefp(y, treatment, pscore, sigma2_treated=None, sigma2_control=None,
           estimand="ate"):
    r"""Optimal-overlap estimands and their trimming rules.

    Parameters
    ----------
    y : array-like
        Outcomes.
    treatment : array-like
        0/1 treatment indicators.
    pscore : array-like
        Propensity scores, strictly inside (0, 1). Estimating them is the
        caller's job; the paper's results condition on the true score and
        its section 6 shows the estimated version is asymptotically
        equivalent.
    sigma2_treated, sigma2_control : array-like, optional
        Conditional outcome variances. Omitted, homoskedasticity is assumed
        and the rules take their corollary forms.
    estimand : {"ate", "att"}
        Which optimal subpopulation to use: Theorem 5.2's two-sided set or
        Theorem 5.3's one-sided one.

    Returns
    -------
    RichResult
        ``estimate`` is the effect over the optimal subpopulation (OSATE);
        ``ate_full`` the same estimator on the untrimmed sample;
        ``owate`` the optimally weighted effect of Theorem 5.4; ``alpha``,
        ``gamma``, ``keep``, ``n_kept``, ``n_trimmed`` describe the rule;
        ``variance_bound`` and ``variance_bound_full`` are
        :math:`\mathbb{E}[\sigma_1^2/e + \sigma_0^2/(1-e) \mid A]/\Pr(A)`
        from Theorem 6.1, the quantity the whole exercise minimises.

    Examples
    --------
    ::

        r = tmlefp(y, w, e)
        r["alpha"], r["n_trimmed"], r["estimate"]

    References
    ----------
    Crump, Hotz, Imbens & Mitnik (2009) *Biometrika* 96(1), 187-199:
    Theorems 5.2-5.4, Corollaries 5.1-5.2, Theorem 6.1.
    """
    y = [float(v) for v in y]
    w = [int(v) for v in treatment]
    e = [float(v) for v in pscore]
    n = len(y)
    if not (n == len(w) == len(e)):
        raise ValueError("tmlefp: y, treatment and pscore must have the "
                         "same length")
    if any(v not in (0, 1) for v in w):
        raise ValueError("tmlefp: treatment must be 0 or 1")
    if estimand not in ("ate", "att"):
        raise ValueError("tmlefp: estimand must be 'ate' or 'att'")

    if estimand == "ate":
        rule = optimal_alpha(e, sigma2_treated, sigma2_control)
        keep = rule["keep"]
        alpha = rule["alpha"]
        gamma = rule["gamma"]
    else:
        rule = optimal_alpha_att(e, w)
        keep = rule["keep"]
        alpha = rule["alpha_t"]
        gamma = float("nan")

    k = ([1.0 / (v * (1.0 - v)) for v in e] if sigma2_treated is None and
         sigma2_control is None else rule.get("k"))
    if k is None:
        k = [1.0 / (v * (1.0 - v)) for v in e]

    def bound(sel):
        m = [k[i] for i in range(n) if sel[i]]
        if not m:
            return float("inf")
        q = len(m) / float(n)
        return (sum(m) / len(m)) / q

    est, n_kept = _ipw(y, w, e, keep)
    full, _ = _ipw(y, w, e, None)
    om = owate_weights(e, sigma2_treated, sigma2_control)
    owate, _ = _ipw(y, w, e, None, om)
    return RichResult(payload={
        "estimate": est,
        "osate": est,
        "ate_full": full,
        "owate": owate,
        "owate_weights": om,
        "alpha": alpha,
        "gamma": gamma,
        "keep": keep,
        "n": n,
        "n_kept": n_kept,
        "n_trimmed": n - n_kept,
        "no_trimming": rule["no_trimming"],
        "variance_bound": bound(keep),
        "variance_bound_full": bound([True] * n),
        "estimand": estimand,
        "note": "the estimand CHANGES with the rule: this is the effect for "
                "the subpopulation kept, not for the whole population "
                "(Crump et al. 2009, section 5)",
        "method": "optimal-overlap subpopulation and weights (Crump, Hotz, "
                  "Imbens & Mitnik 2009)",
    })


def cheatsheet():
    return ("tmlefp: optimal overlap (Crump, Hotz, Imbens & Mitnik 2009). "
            "With propensity scores near 0 or 1 the ATE is barely "
            "estimable, so CHANGE THE ESTIMAND: keep the subpopulation "
            "k(x) = sigma1^2/e + sigma0^2/(1-e) <= 1/(alpha(1-alpha)), "
            "where 1/(alpha(1-alpha)) = 2 E[k | k < that] (Thm 5.2); "
            "homoskedastic, that is alpha <= e <= 1-alpha (Cor 5.1). For "
            "the treated it is one-sided, e <= alpha_t (Thm 5.3). Or drop "
            "the indicator entirely and weight by "
            "omega* = (sigma1^2/e + sigma0^2/(1-e))^{-1} = e(1-e) "
            "(Thm 5.4). No trimming is optimal unless sup k > 2 E[k], "
            "equivalently gamma >= 4.")


# compact alias per ledger/NAMING.md
optimal_overlap = tmlefp

# name carried over from the generated stub this replaced
tmle_effective_pi = tmlefp
