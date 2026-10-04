# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P18: selective labels (``research/lean/P18Selective.lean``; Lakkaraju et al. 2017; Kleinberg et al. 2018).

* ``Research.P18.observed_rate_is_conditional`` / ``nested_rate_identified``
* ``Research.P18.unobserved_bounds`` / ``unobserved_width`` / ``unobserved_ends_attained``

R parity: ``rmorie`` ``R/selective_labels.R`` (``morie_selective_labels``).
"""

from __future__ import annotations

from morie.fn import _array_core as np

__all__ = ["selective_labels"]


def selective_labels(y, released, rule_released, weights=None) -> dict:
    """What a release rule's failure rate can be known from the cases a judge released.

    Examples
    --------
    >>> y = [0, 1, 0, 1, 1, 0]; released = [True, True, True, True, False, False]
    >>> r = selective_labels(y, released, [True, True, True, False, False, False])
    >>> (r["identified"], round(r["rule_rate"], 12))
    (True, 0.333333333333)
    >>> r2 = selective_labels(y, released, [True, True, True, True, True, False])
    >>> ({k: round(v, 12) for k, v in r2["bounds"].items()}, round(r2["width"], 12))
    ({'lower': 0.4, 'upper': 0.6}, 0.2)
    """
    y = np.asarray(y, dtype=float)
    rel = np.asarray([bool(v) for v in released])
    rule = np.asarray([bool(v) for v in rule_released])
    n = y.shape[0]
    if rel.shape[0] != n or rule.shape[0] != n:
        raise ValueError("y, released and rule_released must have equal length")
    if not np.all((y[rel] == 0) | (y[rel] == 1)):
        raise ValueError("y must be 0/1 on the released")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape[0] != n or np.any(w < 0):
        raise ValueError("weights must be non-negative")
    wy = np.where(rel, w * y, 0.0)
    mass_r = float(np.sum(w[rel]))
    mass_m = float(np.sum(w[rule]))
    if mass_r <= 0 or mass_m <= 0:
        raise ValueError("both the released set and the rule's set must have positive mass")
    obs_rate = float(np.sum(wy[rel])) / mass_r
    inter = rel & rule
    fails_inter = float(np.sum(wy[inter]))
    mass_unobs = float(np.sum(w[rule & ~rel]))
    identified = mass_unobs == 0
    lower = fails_inter / mass_m
    upper = (fails_inter + mass_unobs) / mass_m
    mass_inter = float(np.sum(w[inter]))
    return {
        "observed_rate": obs_rate,
        "rule_share_unobserved": mass_unobs / mass_m,
        "identified": identified,
        "rule_rate": lower if identified else float("nan"),
        "bounds": {"lower": lower, "upper": upper},
        "width": upper - lower,
        "naive_rate": fails_inter / mass_inter if mass_inter > 0 else float("nan"),
        "theorems": [
            "Research.P18.observed_rate_is_conditional",
            "Research.P18.nested_rate_identified",
            "Research.P18.unobserved_bounds",
            "Research.P18.unobserved_width",
            "Research.P18.unobserved_ends_attained",
        ],
    }
