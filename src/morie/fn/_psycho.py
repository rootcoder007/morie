# morie.fn -- internal helpers (rootcoder007/morie)
"""Shared machinery for the reliability, IRT and meta-analysis shelf.

Three families, one recurring shape: a variance decomposition whose
COMPONENTS are the quantity of interest, and whose reported value
depends on assumptions that the arithmetic cannot see.

* Intraclass correlation. Shrout, P. E. & Fleiss, J. L. (1979)
  "Intraclass correlations: uses in assessing rater reliability",
  *Psychological Bulletin* 86(2), 420-428,
  doi:10.1037/0033-2909.86.2.420, give SIX
  coefficients, not one, and they are different numbers on the same
  data. The choice is a design statement -- are raters random or
  fixed, is the target a single rating or an average of k -- and
  reporting "the ICC" without the case is the field's standard
  error. The single-measure and average-measure forms differ by the
  Spearman-Brown factor: ICC(*,k) is always the larger.

* IRT ability estimation. The three-parameter logistic
  P(theta) = c + (1-c)/(1 + exp(-a(theta - b))) makes the
  likelihood's shape the whole story: an all-correct or all-wrong
  response pattern has NO finite maximum, so ML diverges and only a
  prior (MAP/EAP) or Warm's weighted likelihood returns a number.
  Silently clipping to +/- 4 instead is the standard bad habit.

* Random-effects meta-analysis. Everything turns on tau^2, the
  between-study variance, and the estimators disagree: DerSimonian-
  Laird is closed-form and downward-biased, Paule-Mandel and REML
  are iterative and less so. Since the weights are 1/(v_i + tau^2),
  a different tau^2 is a different pooled estimate -- so the
  estimator is part of the result, not an implementation detail.
"""

from . import _array_core as np

__all__ = ["anova_two_way", "logistic_3pl", "logistic_3pl_deriv",
           "gauss_hermite", "dersimonian_laird", "fixed_effect_pool",
           "spearman_brown"]


def anova_two_way(y, subject, rater):
    r"""Two-way ANOVA mean squares for a subjects-by-raters table,
    returned as the pieces every Shrout-Fleiss ICC is built from:
    :math:`MS_R` (between subjects), :math:`MS_C` (between raters),
    :math:`MS_E` (residual) and :math:`MS_W` (within subjects, the
    one-way residual pooling rater and error).

    Requires a complete crossed design. An unbalanced table makes the
    mean squares non-orthogonal and every downstream ICC ill-defined,
    so it is an error here rather than a number computed from
    whatever cells happen to be present.
    """
    y = np.asarray(y, dtype=float).ravel()
    s = np.asarray(subject).ravel()
    r = np.asarray(rater).ravel()
    if not (y.size == s.size == r.size):
        raise ValueError("y, subject and rater must have the same length.")
    subs = np.unique(s)
    rats = np.unique(r)
    n, k = subs.size, rats.size
    if n < 2 or k < 2:
        raise ValueError(
            f"need at least 2 subjects and 2 raters, got {n} and {k}.")
    if y.size != n * k:
        raise ValueError(
            f"the design must be complete and crossed: {n} subjects x {k} "
            f"raters needs {n * k} observations, got {y.size}. An unbalanced "
            "table makes the mean squares non-orthogonal and every ICC "
            "ill-defined.")
    M = np.full((n, k), np.nan)
    si = {v: i for i, v in enumerate(subs)}
    ri = {v: j for j, v in enumerate(rats)}
    for idx in range(y.size):
        M[si[s[idx]], ri[r[idx]]] = y[idx]
    if np.any(np.isnan(M)):
        raise ValueError("the subject-by-rater table has empty cells.")
    grand = M.mean()
    row_m = M.mean(axis=1)
    col_m = M.mean(axis=0)
    ss_r = k * np.sum((row_m - grand) ** 2)
    ss_c = n * np.sum((col_m - grand) ** 2)
    ss_t = np.sum((M - grand) ** 2)
    ss_e = ss_t - ss_r - ss_c
    ss_w = ss_t - ss_r
    return {
        "MSR": ss_r / (n - 1),
        "MSC": ss_c / (k - 1),
        "MSE": ss_e / ((n - 1) * (k - 1)),
        "MSW": ss_w / (n * (k - 1)),
        "n": int(n), "k": int(k), "matrix": M,
    }


def spearman_brown(icc1, k):
    r"""Spearman-Brown: the reliability of a mean of :math:`k`
    measurements, :math:`k\rho/(1 + (k-1)\rho)`. Monotone in
    :math:`k` and always at least :math:`\rho` -- which is why every
    average-measure ICC exceeds its single-measure counterpart, and
    why quoting the average-measure figure for a single rating
    overstates reliability."""
    r = float(icc1)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    den = 1.0 + (k - 1) * r
    if den == 0:
        return np.nan
    return k * r / den


def logistic_3pl(theta, a, b, c=0.0):
    r"""The three-parameter logistic item response function

    .. math:: P(\theta) = c + \frac{1-c}
              {1 + \exp\{-a(\theta - b)\}} .

    ``c`` is the lower asymptote (guessing): with ``c > 0`` the
    function never reaches 0, which is what makes the 3PL
    likelihood's tail behaviour differ from the 2PL's.
    """
    th = np.atleast_1d(np.asarray(theta, dtype=float))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    z = np.clip(a * (th[:, None] - b), -500, 500)
    return c + (1.0 - c) / (1.0 + np.exp(-z))


def logistic_3pl_deriv(theta, a, b, c=0.0):
    """dP/dtheta for the 3PL."""
    P = logistic_3pl(theta, a, b, c)
    star = (P - c) / np.maximum(1.0 - c, 1e-12)
    return a * (1.0 - c) * star * (1.0 - star)


def gauss_hermite(n_nodes=41, mu=0.0, sigma=1.0):
    """Nodes and normalised weights for expectation against a normal
    prior, by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    nodes = mu + sigma * x
    weights = w / np.sum(w)
    return nodes, weights


def fixed_effect_pool(yi, vi):
    r"""Inverse-variance (fixed-effect) pooling: weights
    :math:`1/v_i`, estimate :math:`\sum w_i y_i/\sum w_i`, variance
    :math:`1/\sum w_i`, and Cochran's :math:`Q`."""
    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    if y.size != v.size:
        raise ValueError(f"yi has {y.size} entries and vi has {v.size}.")
    if y.size < 2:
        raise ValueError(f"need at least 2 studies, got {y.size}.")
    if np.any(v <= 0):
        raise ValueError("every within-study variance must be positive.")
    w = 1.0 / v
    mu = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - mu) ** 2))
    return mu, 1.0 / float(np.sum(w)), Q, w


def dersimonian_laird(yi, vi):
    r"""The DerSimonian-Laird (1986) moment estimator

    DerSimonian, R. & Laird, N. (1986) "Meta-analysis in clinical
    trials", *Controlled Clinical Trials* 7(3), 177-188,
    doi:10.1016/0197-2456(86)90046-2.

    .. math:: \hat\tau^2_{DL} = \max\left(0,
              \frac{Q - (k-1)}{\sum w_i - \sum w_i^2/\sum w_i}\right).

    Closed-form and therefore ubiquitous, and downward-biased --
    which matters because the truncation at zero is not symmetric:
    a downward-biased tau^2 that hits the floor understates
    heterogeneity AND overstates precision, in the same direction.
    """
    mu, _, Q, w = fixed_effect_pool(yi, vi)
    k = np.asarray(yi).size
    denom = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    if denom <= 0:
        return 0.0
    return max(0.0, (Q - (k - 1)) / denom)


def cheatsheet():
    return ("_psycho: the ICC case, the IRT estimator and the tau^2 method "
            "are all part of the answer, not implementation details")
