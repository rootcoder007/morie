# morie.fn -- function file (rootcoder007/morie)
"""K-sample Anderson-Darling test."""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats
from ._containers import TestResult


def k_sample_anderson_darling(
    *samples: np.ndarray,
) -> TestResult:
    """K-sample Anderson-Darling test.

    Tests whether k independent samples come from a common
    (unspecified) continuous distribution.  Uses scipy's
    implementation of the Scholz & Stephens (1987) k-sample AD.

    :param samples: Two or more 1-D arrays of observations.
    :return: TestResult with AD statistic and approximate p-value.

    Examples
    --------
    >>> r = k_sample_anderson_darling([0.1, 1.2, 0.5, 2.2, 1.9], [3.1, 2.4, 4.2, 3.3, 2.9])
    >>> round(r.statistic, 10), round(r.p_value, 10)
    (4.7260154673, 0.004355978)
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples.")
    arrays = [np.asarray(s, dtype=float).ravel() for s in samples]
    result = stats.anderson_ksamp(arrays)
    stat = float(result.statistic)
    pval = float(result.pvalue) if hasattr(result, "pvalue") else float(result.significance_level)
    n_total = sum(len(a) for a in arrays)

    return TestResult(
        test_name="k-sample Anderson-Darling",
        statistic=stat,
        p_value=pval,
        method="Scholz-Stephens k-sample AD",
        n=n_total,
        extra={"k": len(arrays), "sample_sizes": [len(a) for a in arrays]},
    )


ksamp = k_sample_anderson_darling


def cheatsheet() -> str:
    return "k_sample_anderson_darling({}) -> K-sample Anderson-Darling test."
