# morie.fn -- function file (rootcoder007/morie)
"""Research P11: sentencing effects as intervals.

Python twin of ``R/sentence_bounds.R`` (``morie_sentence_effect_bounds``,
``morie_contaminated_bounds``, ``morie_sentence_effect_mtr``);
machine-checked in ``research/lean/P11Bounds.lean``,
``P11Contaminated.lean`` and ``P11Monotone.lean`` (Manski, Identification
for Prediction and Decision, sec. 5.2 and 7.2; Manski & Nagin 1998):

- ``Research.P11.Pop.outcome_bounds``: P(y=1, z=t) <= P[y(t)=1] <= P(y=1, z=t) + P(z != t), ends attained
- ``Research.P11.Pop.ate_width_one`` / ``ate_contains_zero``: the worst-case contrast has width 1 and contains 0
- ``Research.P11.clean_bounds`` and companions: contaminated-sample bounds
- ``Research.P11.Pop.mtr_lower`` / ``mtr_upper`` / ``mtr_upper_attained``: monotone treatment response
"""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["sentence_effect_bounds", "contaminated_bounds", "sentence_effect_mtr"]


def _prep(y, z, weights, contrast):
    n = len(y)
    if len(z) != n:
        raise ValueError("y and z must have equal length")
    if not all(v in (0, 1) for v in y):
        raise ValueError("y must be 0/1")
    z = [str(v) for v in z]
    lv = sorted(set(z))
    if len(lv) != 2:
        raise ValueError("z must take exactly two distinct values")
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n or any(w < 0 for w in weights) or math.fsum(weights) <= 0:
        raise ValueError("weights must be non-negative with positive total")
    tw = math.fsum(weights)
    w = [v / tw for v in weights]
    if contrast is None:
        trt = lv[1]
    else:
        if str(contrast) not in lv:
            raise ValueError("contrast must be one of the sentence values")
        trt = str(contrast)
    ctl = lv[0] if trt == lv[1] else lv[1]
    joint = {}
    pz = {}
    for lev in (ctl, trt):
        joint[lev] = math.fsum(w[i] * y[i] for i in range(n) if z[i] == lev)
        pz[lev] = math.fsum(w[i] for i in range(n) if z[i] == lev)
    return ctl, trt, joint, pz


def sentence_effect_bounds(y, z, weights=None, contrast=None):
    """Worst-case identification bounds for a binary outcome under two sentences.

    With no assumption on how sentences were assigned, the probability of
    the outcome under sentence ``t`` is identified only up to
    ``[P(y=1, z=t), P(y=1, z=t) + P(z != t)]``, both ends attainable; the
    contrast between the two sentences lies in an interval of width exactly
    one that always contains zero.

    Parameters
    ----------
    y : sequence of int
        Binary outcome per person (0/1).
    z : sequence
        Sentence per person, exactly two distinct values.
    weights : sequence of float, optional
        Non-negative weights.
    contrast : optional
        Which sentence value is the treatment (default the second in sorted
        order); the other is the comparison.

    Returns
    -------
    RichResult
        ``levels`` (``comparison``, ``treatment``), ``joint`` and ``pz`` per
        sentence, ``outcome_bounds`` (``{level: (lower, upper)}``),
        ``ate_bounds`` (``lower``, ``upper``), ``ate_width`` (always 1),
        ``naive_difference`` and ``theorems``.

    Examples
    --------
    >>> y = [1, 0, 1, 1, 0, 0, 1, 0]
    >>> z = ["custody", "community", "custody", "community", "community", "custody", "custody", "community"]
    >>> b = sentence_effect_bounds(y, z)
    >>> b.levels
    {'comparison': 'community', 'treatment': 'custody'}
    >>> b.ate_bounds, b.ate_width, b.naive_difference
    ({'lower': -0.25, 'upper': 0.75}, 1.0, 0.5)
    """
    ctl, trt, joint, pz = _prep(y, z, weights, contrast)
    ob = {lev: (joint[lev], joint[lev] + (1 - pz[lev])) for lev in (ctl, trt)}
    ate = {"lower": ob[trt][0] - ob[ctl][1], "upper": ob[trt][1] - ob[ctl][0]}
    return RichResult(
        title="Worst-case sentencing-effect bounds",
        payload={
            "levels": {"comparison": ctl, "treatment": trt},
            "joint": joint,
            "pz": pz,
            "outcome_bounds": ob,
            "ate_bounds": ate,
            "ate_width": ate["upper"] - ate["lower"],
            "naive_difference": joint[trt] / pz[trt] - joint[ctl] / pz[ctl],
            "theorems": [
                "Research.P11.Pop.outcome_bounds",
                "Research.P11.Pop.lower_attained",
                "Research.P11.Pop.upper_attained",
                "Research.P11.Pop.ate_width_one",
                "Research.P11.Pop.ate_contains_zero",
            ],
        },
    )


def contaminated_bounds(q, p):
    """Contaminated-sample bounds for a recorded proportion.

    A recorded distribution mixes the clean distribution with an unknown
    contaminant in known share ``p``. For any event with recorded
    probability ``q`` the clean probability lies in
    ``[max(0, (q-p)/(1-p)), min(1, q/(1-p))]``; both ends are attained,
    the width before clipping is ``p/(1-p)``, and the interval is narrower
    than [0, 1] exactly when ``p < q`` or ``p < 1 - q``.

    Parameters
    ----------
    q : float or sequence of float
        Recorded probability of the event.
    p : float
        Known contamination share in [0, 1).

    Returns
    -------
    dict
        Columns ``q``, ``lower``, ``upper``, ``width``, ``informative`` (lists)
        and ``theorems``.

    Examples
    --------
    >>> b = contaminated_bounds([0.05, 0.5, 0.97], 0.1)
    >>> [round(v, 12) for v in b["lower"]], [round(v, 12) for v in b["upper"]]
    ([0.0, 0.444444444444, 0.966666666667], [0.055555555556, 0.555555555556, 1.0])
    >>> round(b["width"][0], 12), b["informative"]
    (0.111111111111, [True, True, True])
    """
    if not isinstance(p, (int, float)) or math.isnan(p) or p < 0 or p >= 1:
        raise ValueError("p must be a single number in [0, 1)")
    qs = [float(q)] if isinstance(q, (int, float)) else [float(v) for v in q]
    if any(math.isnan(v) or v < 0 or v > 1 for v in qs):
        raise ValueError("q must lie in [0, 1]")
    return {
        "q": qs,
        "lower": [max(0.0, (v - p) / (1 - p)) for v in qs],
        "upper": [min(1.0, v / (1 - p)) for v in qs],
        "width": [p / (1 - p)] * len(qs),
        "informative": [(p < v) or (p < 1 - v) for v in qs],
        "theorems": [
            "Research.P11.clean_bounds",
            "Research.P11.clean_lower_attained",
            "Research.P11.clean_upper_attained",
            "Research.P11.clean_width",
            "Research.P11.clean_informative",
        ],
    }


def sentence_effect_mtr(y, z, weights=None, contrast=None, direction="non-decreasing"):
    """Monotone-treatment-response bounds for a binary outcome under two sentences.

    If the harsher sentence never lowers anyone's outcome (Manski 1997), the
    contrast ``E[y(b)] - E[y(a)]`` lies in ``[0, P(y=1, z=b) + P(y=0, z=a)]``
    with the upper end attained; the ``"non-increasing"`` direction gives
    ``[-(P(y=0, z=b) + P(y=1, z=a)), 0]``.

    Parameters
    ----------
    y, z, weights, contrast
        As in :func:`sentence_effect_bounds`.
    direction : {"non-decreasing", "non-increasing"}
        Direction of the monotonicity assumption.

    Returns
    -------
    RichResult
        ``levels``, ``direction``, ``bounds`` (``lower``, ``upper``),
        ``width``, ``naive_difference`` and ``theorems``.

    Examples
    --------
    >>> y = [1, 0, 1, 1, 0, 0, 1, 0]
    >>> z = ["custody", "community", "custody", "community", "community", "custody", "custody", "community"]
    >>> sentence_effect_mtr(y, z).bounds
    {'lower': 0.0, 'upper': 0.75}
    >>> sentence_effect_mtr(y, z, direction="non-increasing").bounds
    {'lower': -0.25, 'upper': 0.0}
    """
    if direction not in ("non-decreasing", "non-increasing"):
        raise ValueError("direction must be 'non-decreasing' or 'non-increasing'")
    ctl, trt, joint, pz = _prep(y, z, weights, contrast)
    if direction == "non-decreasing":
        bounds = {"lower": 0.0, "upper": joint[trt] + (pz[ctl] - joint[ctl])}
    else:
        bounds = {"lower": -((pz[trt] - joint[trt]) + joint[ctl]), "upper": 0.0}
    return RichResult(
        title="Monotone-treatment-response sentencing bounds",
        payload={
            "levels": {"comparison": ctl, "treatment": trt},
            "direction": direction,
            "bounds": bounds,
            "width": bounds["upper"] - bounds["lower"],
            "naive_difference": joint[trt] / pz[trt] - joint[ctl] / pz[ctl],
            "theorems": [
                "Research.P11.Pop.mtr_lower",
                "Research.P11.Pop.mtr_upper",
                "Research.P11.Pop.mtr_upper_attained",
            ],
        },
    )


def cheatsheet() -> str:
    return (
        "sentence_effect_bounds(y, z) -> worst-case outcome and contrast bounds, width 1 (Research P11)\n"
        "contaminated_bounds(q, p) -> clean-probability interval under contamination share p\n"
        "sentence_effect_mtr(y, z, direction='non-decreasing') -> MTR bounds on the contrast"
    )
