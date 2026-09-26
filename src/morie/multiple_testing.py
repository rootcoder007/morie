"""Multiple testing correction and multiplicity-adjusted inference.

Provides comprehensive p-value adjustment methods, family-wise error rate
(FWER) control, false discovery rate (FDR) control, and simultaneous
confidence interval methods for multiple comparisons in epidemiological
and biostatistical analyses.

All functions accept arrays of p-values and return adjusted p-values or
decision boundaries.  Compatible with the MORIE pipeline and can be applied
to any collection of hypothesis tests from the analysis modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn import _stats_core as stats

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class AdjustedPValues:
    """Container for multiplicity-adjusted p-values."""

    original: np.ndarray
    adjusted: np.ndarray
    method: str
    alpha: float
    rejected: np.ndarray  # boolean mask
    n_rejected: int
    n_tests: int
    labels: list[str] | None = None

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame."""
        d = {
            "p_original": self.original,
            "p_adjusted": self.adjusted,
            "rejected": self.rejected,
        }
        if self.labels is not None:
            d["label"] = self.labels
        return pd.DataFrame(d)


@dataclass
class GatekeepingResult:
    """Result from a gatekeeping or hierarchical testing procedure."""

    stages: list[dict]
    overall_rejected: np.ndarray
    method: str
    alpha: float


@dataclass
class ClosedTestResult:
    """Result from a closed testing procedure."""

    original_p: np.ndarray
    adjusted_p: np.ndarray
    rejected: np.ndarray
    intersection_tests: dict
    method: str


# ---------------------------------------------------------------------------
# FWER-controlling procedures
# ---------------------------------------------------------------------------


def bonferroni(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Bonferroni correction (FWER control).

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.minimum(p * m, 1.0)
    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted,
        method="bonferroni",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


def sidak(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Sidak correction (FWER control, slightly less conservative than Bonferroni).

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = 1.0 - (1.0 - p) ** m
    adjusted = np.minimum(adjusted, 1.0)
    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted,
        method="sidak",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


def holm(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Holm step-down procedure (FWER control).

    Uniformly more powerful than Bonferroni.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m)
    for i in range(m):
        adjusted_sorted[i] = sorted_p[i] * (m - i)

    # Enforce monotonicity (step-down).
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Restore original order.
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted,
        method="holm",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


def hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Hochberg step-up procedure (FWER control under PRDS).

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m)
    for i in range(m):
        adjusted_sorted[i] = sorted_p[i] * (m - i)

    # Enforce monotonicity (step-up: from largest to smallest).
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted,
        method="hochberg",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


def hommel(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Hommel procedure (FWER control, most powerful single-step method under independence).

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    # Hommel (1988) adjusted p-values, R's p.adjust(method = "hommel")
    # algorithm (Wright 1992): for each subset size k = m-1..2 the
    # Simes-type minimum over the largest k p-values, carried as a running
    # maximum. The old loop capped the smallest p-values at the q_star of
    # the whole family (0.055 where R gives 0.006).
    ps = [float(v) for v in sorted_p]
    base = min(m * ps[i] / (i + 1) for i in range(m))
    q = [base] * m
    pa = [base] * m
    for k in range(m - 1, 1, -1):
        n_low = m - k + 1  # indices 0..n_low-1
        q1 = min(k * ps[n_low + j] / (j + 2) for j in range(k - 1))
        for i in range(n_low):
            q[i] = min(k * ps[i], q1)
        for i in range(n_low, m):
            q[i] = q[n_low - 1]
        pa = [max(x, y) for x, y in zip(pa, q)]
    adjusted_sorted = np.array([min(1.0, max(x, y)) for x, y in zip(pa, ps)])

    adjusted_final = np.empty(m)
    adjusted_final[order] = adjusted_sorted

    rejected = adjusted_final <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted_final,
        method="hommel",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


def holm_sidak(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Holm-Sidak step-down procedure.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m)
    for i in range(m):
        adjusted_sorted[i] = 1.0 - (1.0 - sorted_p[i]) ** (m - i)

    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted,
        method="holm_sidak",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# FDR-controlling procedures
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Benjamini-Hochberg procedure (FDR control).

    Controls the false discovery rate at level alpha under independence
    or positive regression dependency (PRDS).

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Target FDR level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m)
    for i in range(m):
        adjusted_sorted[i] = sorted_p[i] * m / (i + 1)

    # Enforce monotonicity (step-up: from largest to smallest).
    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted,
        method="benjamini_hochberg",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


# Alias.
bh = benjamini_hochberg


def benjamini_yekutieli(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Benjamini-Yekutieli procedure (FDR control under arbitrary dependence).

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Target FDR level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    c_m = np.sum(1.0 / np.arange(1, m + 1))  # harmonic number

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m)
    for i in range(m):
        adjusted_sorted[i] = sorted_p[i] * m * c_m / (i + 1)

    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted,
        method="benjamini_yekutieli",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


# Alias.
by = benjamini_yekutieli


def storey_q(
    p_values: np.ndarray,
    alpha: float = 0.05,
    lambda_param: float = 0.5,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Storey's q-value procedure (adaptive FDR control).

    Estimates the proportion of true null hypotheses (pi0) and uses it
    to achieve higher power than BH when many nulls are false.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    alpha : float
        Target FDR level.
    lambda_param : float
        Tuning parameter for pi0 estimation.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    # Estimate pi0.
    pi0 = np.sum(p > lambda_param) / (m * (1.0 - lambda_param))
    pi0 = min(pi0, 1.0)

    order = np.argsort(p)
    sorted_p = p[order]

    q_values = np.empty(m)
    for i in range(m):
        q_values[i] = sorted_p[i] * pi0 * m / (i + 1)

    for i in range(m - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])

    q_values = np.minimum(q_values, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = q_values

    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=p,
        adjusted=adjusted,
        method=f"storey_q(pi0={pi0:.3f})",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Local FDR
# ---------------------------------------------------------------------------


def local_fdr(
    p_values: np.ndarray,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Estimate local false discovery rate from p-values.

    Uses an empirical Bayes approach with a two-component mixture model.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    pd.DataFrame
        With columns: p_value, z_score, local_fdr, label (if provided).
    """
    p = np.asarray(p_values, dtype=float)
    z = stats.norm.ppf(1 - p / 2)  # two-sided z-scores

    # Estimate pi0 via BH method.
    m = len(p)
    pi0_hat = min(np.sum(p > 0.5) / (m * 0.5), 1.0)

    # Estimate f(z) via KDE.
    kde = stats.gaussian_kde(z)
    f_z = kde(z)

    # f0(z) = standard normal density.
    f0_z = stats.norm.pdf(z)

    # local FDR = pi0 * f0(z) / f(z).
    lfdr = pi0_hat * f0_z / np.maximum(f_z, 1e-10)
    lfdr = np.minimum(lfdr, 1.0)

    d = {"p_value": p, "z_score": z, "local_fdr": lfdr}
    if labels is not None:
        d["label"] = labels
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# Permutation-based methods
# ---------------------------------------------------------------------------


def permutation_fwer(
    test_statistics: np.ndarray,
    null_distribution: np.ndarray,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """FWER control via max-T permutation test (Westfall-Young).

    Parameters
    ----------
    test_statistics : array-like, shape (m,)
        Observed test statistics.
    null_distribution : array-like, shape (n_perm, m)
        Null distribution from permutations.
    alternative : str
        One of 'two-sided', 'greater', 'less'.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    t_obs = np.asarray(test_statistics, dtype=float)
    t_null = np.asarray(null_distribution, dtype=float)
    m = len(t_obs)

    if alternative == "two-sided":
        t_obs_abs = np.abs(t_obs)
        t_null_abs = np.abs(t_null)
    elif alternative == "greater":
        t_obs_abs = t_obs
        t_null_abs = t_null
    else:
        t_obs_abs = -t_obs
        t_null_abs = -t_null

    # Step-down max-T procedure.
    order = np.argsort(-t_obs_abs)  # descending
    t_sorted = t_obs_abs[order]

    adjusted_sorted = np.empty(m)
    for i in range(m):
        remaining = order[i:]
        max_null = t_null_abs[:, remaining].max(axis=1)
        adjusted_sorted[i] = np.mean(max_null >= t_sorted[i])

    # Enforce monotonicity.
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    rejected = adjusted <= alpha
    return AdjustedPValues(
        original=adjusted,  # permutation p-values
        adjusted=adjusted,
        method="permutation_maxT",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


def permutation_fdr(
    p_values: np.ndarray,
    null_p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """FDR control via permutation null distribution.

    Parameters
    ----------
    p_values : array-like, shape (m,)
        Observed p-values.
    null_p_values : array-like, shape (n_perm, m)
        P-values from permuted data.
    alpha : float
        Target FDR level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    p_null = np.asarray(null_p_values, dtype=float)
    m = len(p)

    thresholds = np.sort(p)
    fdr_at_t = np.empty(len(thresholds))

    for i, t in enumerate(thresholds):
        n_rejected = np.sum(p <= t)
        if n_rejected == 0:
            fdr_at_t[i] = 0
        else:
            expected_false = np.mean(np.sum(p_null <= t, axis=1))
            fdr_at_t[i] = expected_false / n_rejected

    # Find the largest threshold where FDR <= alpha.
    valid = fdr_at_t <= alpha
    t_star = thresholds[np.where(valid)[0][-1]] if valid.any() else 0

    rejected = p <= t_star

    # Assign q-values.
    q_values = np.ones(m)
    for i in range(m):
        candidate_fdr = fdr_at_t[thresholds >= p[i]]
        if len(candidate_fdr) > 0:
            q_values[i] = candidate_fdr.min()

    return AdjustedPValues(
        original=p,
        adjusted=q_values,
        method="permutation_fdr",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Combining p-values
# ---------------------------------------------------------------------------


def fisher_combined(p_values: np.ndarray) -> tuple[float, float]:
    """Fisher's method for combining independent p-values.

    Parameters
    ----------
    p_values : array-like
        Independent p-values.

    Returns
    -------
    tuple[float, float]
        (test_statistic, combined_p_value).
    """
    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    chi2_stat = -2.0 * np.sum(np.log(p))
    combined_p = 1.0 - stats.chi2.cdf(chi2_stat, df=2 * len(p))
    return float(chi2_stat), float(combined_p)


def stouffer_combined(
    p_values: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Stouffer's method for combining independent p-values.

    Parameters
    ----------
    p_values : array-like
        Independent p-values.
    weights : array-like, optional
        Weights for each p-value.  If None, equal weights are used.

    Returns
    -------
    tuple[float, float]
        (z_statistic, combined_p_value).
    """
    p = np.asarray(p_values, dtype=float)
    z = stats.norm.ppf(1 - p)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        z_combined = np.sum(w * z) / np.sqrt(np.sum(w**2))
    else:
        z_combined = np.sum(z) / np.sqrt(len(z))
    combined_p = 1.0 - stats.norm.cdf(z_combined)
    return float(z_combined), float(combined_p)


def tippett_combined(p_values: np.ndarray) -> tuple[float, float]:
    """Tippett's method: combine via minimum p-value.

    Parameters
    ----------
    p_values : array-like
        Independent p-values.

    Returns
    -------
    tuple[float, float]
        (min_p, combined_p).
    """
    p = np.asarray(p_values, dtype=float)
    min_p = p.min()
    combined_p = 1.0 - (1.0 - min_p) ** len(p)
    return float(min_p), float(combined_p)


def simes_combined(p_values: np.ndarray) -> tuple[float, float]:
    """Simes' test for the global null hypothesis.

    Parameters
    ----------
    p_values : array-like
        P-values (may be dependent under PRDS).

    Returns
    -------
    tuple[float, float]
        (simes_statistic, combined_p).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    sorted_p = np.sort(p)
    simes_vals = sorted_p * m / np.arange(1, m + 1)
    simes_stat = simes_vals.min()
    return float(simes_stat), float(simes_stat)  # Simes p = min(m*p_(i)/i)


def harmonic_mean_p(p_values: np.ndarray) -> float:
    """Harmonic mean p-value for combining dependent tests.

    Parameters
    ----------
    p_values : array-like
        P-values (can be dependent).

    Returns
    -------
    float
        Harmonic mean p-value.
    """
    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 1e-300, 1.0)
    m = len(p)
    hmp = m / np.sum(1.0 / p)
    return float(hmp)


def cauchy_combination(
    p_values: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Cauchy combination test (Liu & Xie 2020).

    Robust to arbitrary correlation structures.

    Parameters
    ----------
    p_values : array-like
        P-values.
    weights : array-like, optional
        Non-negative weights summing to 1.

    Returns
    -------
    tuple[float, float]
        (test_statistic, combined_p).
    """
    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    m = len(p)

    if weights is None:
        w = np.ones(m) / m
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()

    # Cauchy transformation.
    t_values = np.tan((0.5 - p) * np.pi)
    t_stat = np.sum(w * t_values)

    # P-value from standard Cauchy distribution.
    combined_p = 1.0 - stats.cauchy.cdf(t_stat)
    return float(t_stat), float(combined_p)


# ---------------------------------------------------------------------------
# Gatekeeping / hierarchical testing
# ---------------------------------------------------------------------------


def hierarchical_bonferroni(
    p_values_by_family: list[np.ndarray],
    alpha: float = 0.05,
    propagate_alpha: bool = True,
) -> GatekeepingResult:
    """Hierarchical (serial gatekeeping) Bonferroni procedure.

    Tests are organized in families (stages). A family is tested only if
    at least one hypothesis in the previous family is rejected.

    Parameters
    ----------
    p_values_by_family : list of array-like
        P-values grouped by family/stage.
    alpha : float
        Overall FWER level.
    propagate_alpha : bool
        If True, unused alpha is propagated to next family.

    Returns
    -------
    GatekeepingResult
    """
    n_families = len(p_values_by_family)
    stages = []
    all_rejected = []
    remaining_alpha = alpha

    for i, family_p in enumerate(p_values_by_family):
        p = np.asarray(family_p, dtype=float)
        m = len(p)

        adj = bonferroni(p, alpha=remaining_alpha)
        n_rej = adj.n_rejected

        stages.append(
            {
                "family": i,
                "n_tests": m,
                "alpha_used": remaining_alpha,
                "n_rejected": n_rej,
                "adjusted_p": adj.adjusted,
                "rejected": adj.rejected,
            }
        )
        all_rejected.extend(adj.rejected.tolist())

        if n_rej == 0:
            # Gate closed: don't test subsequent families.
            for j in range(i + 1, n_families):
                fam_p_j = np.asarray(p_values_by_family[j])
                stages.append(
                    {
                        "family": j,
                        "n_tests": len(fam_p_j),
                        "alpha_used": 0,
                        "n_rejected": 0,
                        "adjusted_p": np.ones(len(fam_p_j)),
                        "rejected": np.zeros(len(fam_p_j), dtype=bool),
                    }
                )
                all_rejected.extend([False] * len(fam_p_j))
            break

        if propagate_alpha:
            unused = remaining_alpha * (1 - n_rej / m)
            remaining_alpha = remaining_alpha - unused + unused  # In practice, keep the same alpha

    return GatekeepingResult(
        stages=stages,
        overall_rejected=np.array(all_rejected, dtype=bool),
        method="hierarchical_bonferroni",
        alpha=alpha,
    )


def fixed_sequence(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Fixed-sequence (predetermined order) testing procedure.

    Tests are evaluated in order; the procedure stops at the first
    non-rejection.  No multiplicity adjustment needed for tests
    that are reached.

    Parameters
    ----------
    p_values : array-like
        P-values in the predetermined testing order.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    rejected = np.zeros(m, dtype=bool)

    for i in range(m):
        if p[i] <= alpha:
            rejected[i] = True
        else:
            break

    return AdjustedPValues(
        original=p,
        adjusted=p,  # No adjustment in fixed-sequence.
        method="fixed_sequence",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


def fallback_procedure(
    p_values: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Fallback (fixed-sequence with alpha spending) procedure.

    Parameters
    ----------
    p_values : array-like
        P-values in testing order.
    weights : array-like
        Alpha allocation weights (must sum to 1).
    alpha : float
        Overall significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    p = np.asarray(p_values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = len(p)

    allocated = w * alpha
    rejected = np.zeros(m, dtype=bool)
    carry_over = 0.0

    for i in range(m):
        threshold = allocated[i] + carry_over
        if p[i] <= threshold:
            rejected[i] = True
            carry_over = threshold - p[i]
        else:
            carry_over = threshold  # pass full allocation to next

    return AdjustedPValues(
        original=p,
        adjusted=p,
        method="fallback",
        alpha=alpha,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        n_tests=m,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def estimate_pi0(p_values: np.ndarray, method: str = "storey") -> float:
    """Estimate the proportion of true null hypotheses (pi0).

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    method : str
        Estimation method: 'storey' (default), 'bootstrap', 'two_step'.

    Returns
    -------
    float
        Estimated pi0 in [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    lambdas = [0.05 * k for k in range(1, 20)]  # qvalue's default grid, 0.05 to 0.95
    pi0_lam = [sum(1 for v in p.tolist() if v >= lam) / (m * (1 - lam)) for lam in lambdas]
    if method == "storey":
        # Storey & Tibshirani (2003): a cubic smoothing spline with 3 df
        # through pi0(lambda), read at the largest lambda (qvalue's
        # pi0.method = "smoother")
        pi0 = _smoothing_spline_df(lambdas, pi0_lam, 3.0)[-1]
        if pi0 <= 0:
            raise ValueError("estimated pi0 <= 0 (the smoother extrapolated below zero); use method='bootstrap'")
        return float(min(pi0, 1.0))

    elif method == "bootstrap":
        # Storey, Taylor & Siegmund (2004): the lambda minimising the
        # closed-form MSE against the 10% quantile of pi0(lambda)
        # (qvalue's pi0.method = "bootstrap")
        srt = sorted(pi0_lam)
        h = 0.1 * (len(srt) - 1)
        lo = int(h)
        min_pi0 = srt[lo] + (h - lo) * (srt[min(lo + 1, len(srt) - 1)] - srt[lo])
        mse = []
        for lam, pl in zip(lambdas, pi0_lam):
            W = sum(1 for v in p.tolist() if v >= lam)
            mse.append(W / (m**2 * (1 - lam) ** 2) * (1 - W / m) + (pl - min_pi0) ** 2)
        best = min(mse)
        return float(min(min(pl for pl, e in zip(pi0_lam, mse) if e == best), 1.0))

    elif method == "two_step":
        # Benjamini, Krieger & Yekutieli (2006): stage-one BH at q/(1 + q)
        bh_result = benjamini_hochberg(p, alpha=0.05 / 1.05)
        r = bh_result.n_rejected
        return float(min((m - r) / m, 1.0))

    else:
        raise ValueError(f"Unknown method: {method}")


def _smoothing_spline_df(x, y, df):
    """Fitted values of the natural cubic smoothing spline with the given
    effective degrees of freedom (Green & Silverman 1994, sec. 2.3):
    g = (I + a K)^-1 y with K = Q R^-1 Q', a solved so that tr = df."""
    import math

    n = len(x)
    hh = [x[i + 1] - x[i] for i in range(n - 1)]
    Q = [[0.0] * (n - 2) for _ in range(n)]
    Rm = [[0.0] * (n - 2) for _ in range(n - 2)]
    for j in range(1, n - 1):
        c = j - 1
        Q[j - 1][c] = 1.0 / hh[j - 1]
        Q[j][c] = -1.0 / hh[j - 1] - 1.0 / hh[j]
        Q[j + 1][c] = 1.0 / hh[j]
        Rm[c][c] = (hh[j - 1] + hh[j]) / 3.0
        if c + 1 < n - 2:
            Rm[c][c + 1] = Rm[c + 1][c] = hh[j] / 6.0
    RiQt = np.linalg.solve(np.array(Rm), np.array([[Q[i][c] for i in range(n)] for c in range(n - 2)]))
    K = [[sum(Q[i][c] * float(RiQt[c][k]) for c in range(n - 2)) for k in range(n)] for i in range(n)]

    def smoother(a):
        return np.linalg.inv(np.array([[(1.0 if i == k else 0.0) + a * K[i][k] for k in range(n)] for i in range(n)]))

    lo, hi = -30.0, 30.0  # bisection on log a: tr(S) falls from n to 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        S = smoother(math.exp(mid))
        if sum(float(S[i][i]) for i in range(n)) > df:
            lo = mid
        else:
            hi = mid
    S = smoother(math.exp(0.5 * (lo + hi)))
    return [sum(float(S[i][k]) * y[k] for k in range(n)) for i in range(n)]


def adjust_p_values(
    p_values: np.ndarray,
    method: str = "bh",
    alpha: float = 0.05,
    labels: list[str] | None = None,
) -> AdjustedPValues:
    """Convenience dispatcher for p-value adjustment methods.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.
    method : str
        Adjustment method. One of: 'bonferroni', 'sidak', 'holm',
        'hochberg', 'hommel', 'holm_sidak', 'bh', 'by', 'storey'.
    alpha : float
        Significance level.
    labels : list[str], optional
        Labels for each test.

    Returns
    -------
    AdjustedPValues
    """
    methods = {
        "bonferroni": bonferroni,
        "sidak": sidak,
        "holm": holm,
        "hochberg": hochberg,
        "hommel": hommel,
        "holm_sidak": holm_sidak,
        "bh": benjamini_hochberg,
        "benjamini_hochberg": benjamini_hochberg,
        "by": benjamini_yekutieli,
        "benjamini_yekutieli": benjamini_yekutieli,
        "storey": storey_q,
        "fdr": benjamini_hochberg,
        "fwer": holm,
    }

    if method.lower() not in methods:
        raise ValueError(f"Unknown method '{method}'. Available: {', '.join(methods.keys())}")

    return methods[method.lower()](p_values, alpha=alpha, labels=labels)


def n_effective_tests(
    p_values: np.ndarray | None = None,
    correlation_matrix: np.ndarray | None = None,
    method: str = "galwey",
) -> float:
    """Estimate the effective number of independent tests.

    Useful for determining appropriate Bonferroni correction when
    tests are correlated.

    Parameters
    ----------
    p_values : array-like, optional
        P-values (used for some methods).
    correlation_matrix : array-like, optional
        Correlation matrix between test statistics.
    method : str
        'galwey' (eigenvalue-based), 'li_ji' (Li & Ji 2005),
        'nyholt' (Nyholt 2004).

    Returns
    -------
    float
        Estimated number of effective independent tests.
    """
    if correlation_matrix is None:
        if p_values is None:
            raise ValueError("correlation_matrix is required (or p_values, for the uncorrelated count)")
        # without a correlation structure every test counts once
        return float(len(np.asarray(p_values, dtype=float).ravel()))

    R = np.asarray(correlation_matrix, dtype=float)
    eigenvalues = np.linalg.eigvalsh(R)
    eigenvalues = eigenvalues[eigenvalues > 0]
    m = len(eigenvalues)

    if method == "galwey":
        # Galwey (2009): sum of sqrt of eigenvalues, squared, divided by sum.
        m_eff = (np.sum(np.sqrt(eigenvalues))) ** 2 / np.sum(eigenvalues)

    elif method == "li_ji":
        # Li & Ji (2005).
        m_eff = sum(1 if ev >= 1 else 0 for ev in eigenvalues) + sum(ev - int(ev) for ev in eigenvalues if ev < 1)

    elif method == "nyholt":
        # Nyholt (2004).
        var_eigenvalues = np.var(eigenvalues)
        m_eff = 1 + (m - 1) * (1 - var_eigenvalues / m)

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(max(1.0, m_eff))
