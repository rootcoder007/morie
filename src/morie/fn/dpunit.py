# morie.fn -- function file (rootcoder007/morie)
"""Privacy unit: what one record actually protects."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["dp_unit_definition"]


def dp_unit_definition(records, unit=None):
    r"""Report the contribution structure that sets the effective privacy unit.

    A DP guarantee protects a **unit**, and the unit is whatever "one record"
    means in the sensitivity calculation. If sensitivity was computed per row
    but one person contributes many rows, the person is not protected -- only
    each row is, and an adversary who can link rows recovers the person.

    Given a mapping from records to units, this reports the maximum number of
    records any single unit contributes. That number is the factor by which
    sensitivity must be multiplied for the guarantee to apply at the unit
    level:

    .. math::
        \Delta_{\text{unit}} = \max_u |\{i : \text{unit}(i) = u\}|
                              \times \Delta_{\text{record}}.

    User-level privacy on event-level data is the usual case -- one user with
    a thousand log lines needs a thousand times the noise, or their
    contribution capped at :math:`k` records first, which is why bounded
    contribution is standard in production systems.

    Parameters
    ----------
    records : array-like
        Unit label for each record.
    unit : hashable, optional
        Report on this unit specifically.

    Returns
    -------
    RichResult
        ``n_records``, ``n_units``, ``max_contribution``,
        ``sensitivity_multiplier``, ``contributions``.

    References
    ----------
    Wilson, R. J., Zhang, C. Y., Lam, W., Desfontaines, D., Simmons-Marengo,
        D., & Gipson, B. (2020). Differentially private SQL with bounded user
        contribution. *PoPETs*, 2020(2), 230-250.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-407.

    Examples
    --------
    When each unit contributes once, record-level and unit-level privacy
    coincide and the multiplier is 1.

    >>> r = dp_unit_definition([1, 2, 3, 4])
    >>> int(r["max_contribution"]), int(r["sensitivity_multiplier"])
    (1, 1)

    When one user contributes many rows, the multiplier is the factor by which
    sensitivity was understated.

    >>> r = dp_unit_definition([1, 1, 1, 1, 1, 2, 3])
    >>> int(r["max_contribution"])
    5
    >>> bool(r.warnings)
    True

    Per-unit counts are returned so contribution capping can be planned.

    >>> sorted(int(v) for v in dp_unit_definition([1, 1, 2])["contributions"])
    [1, 2]

    >>> dp_unit_definition([])
    Traceback (most recent call last):
        ...
    ValueError: records must be non-empty
    """
    r = np.asarray(records).ravel()
    if r.size == 0:
        raise ValueError("records must be non-empty")
    units, counts = np.unique(r, return_counts=True)
    mx = int(counts.max())
    warn = []
    if mx > 1:
        warn.append(
            f"one unit contributes {mx} records, so a sensitivity computed per "
            f"record understates the unit-level sensitivity by {mx}x; either "
            "multiply the noise or cap contributions first"
        )
    payload = {
        "n_records": int(r.size), "n_units": int(units.size),
        "max_contribution": mx, "sensitivity_multiplier": mx,
        "units": units, "contributions": counts,
        "mean_contribution": float(counts.mean()),
        "method": "dp_unit_definition",
    }
    if unit is not None:
        payload["unit_contribution"] = int(np.sum(r == unit))
    return RichResult(
        title="Privacy unit",
        summary_lines=[("records", int(r.size)), ("units", int(units.size)),
                       ("max contribution", mx)],
        warnings=warn,
        payload=payload,
    )


def cheatsheet():
    return "dpunit: per-ROW sensitivity does not protect a PERSON with many rows -- multiply or cap"
