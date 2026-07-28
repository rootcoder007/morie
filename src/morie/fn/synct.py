# morie.fn -- function file (rootcoder007/morie)
"""Synthetic control for comparative case studies."""

import numpy as np

from ._did import as_panel, simplex_lstsq
from ._richresult import RichResult

__all__ = ["synthetic_control"]


def _fit_one(Y, treated_row, t0):
    donors = np.delete(np.arange(Y.shape[0]), treated_row)
    A = Y[donors][:, :t0].T
    b = Y[treated_row, :t0]
    w, _, _ = simplex_lstsq(A, b)
    fit = Y[donors].T @ w
    return w, donors, fit


def _rmspe(gap, sl):
    return float(np.sqrt(np.mean(gap[sl] ** 2)))


def synthetic_control(Y, unit_id, time_id, treated_unit, treatment_time,
                      n_placebo=None):
    r"""Weight the donor pool to reproduce the treated unit's past.

    Abadie and Gardeazabal's method for the case where there is ONE
    treated unit and no obvious comparison. Choose weights
    :math:`w \geq 0` with :math:`\sum_j w_j = 1` minimising the
    pre-treatment fit

    .. math:: \min_w \sum_{t < T_0}
              \Big(Y_{1t} - \sum_{j \neq 1} w_j Y_{jt}\Big)^2,

    and read the effect off the post-treatment gap
    :math:`Y_{1t} - \sum_j w_j Y_{jt}`.

    The two constraints are the method. Non-negativity and summing to
    one force the synthetic unit to be a weighted AVERAGE of real
    units, so it cannot extrapolate outside the donor pool's range --
    an unconstrained regression would fit the pre-period better and
    routinely produce a counterfactual no combination of real units
    could ever be.

    Inference is by PERMUTATION, not by a standard error. There is
    one treated unit, so there is no sampling distribution to appeal
    to; instead the same estimator is applied pretending each donor
    was treated, and the treated unit's post/pre RMSPE ratio is ranked
    among those placebos (Abadie, Diamond and Hainmueller 2010,
    section 5). The resulting p-value cannot be finer than
    :math:`1/(J+1)`, and that floor is reported so a "p = 0.05" from
    19 donors is not mistaken for precision.

    A good pre-treatment fit is a precondition, not a result:
    ``pre_rmspe`` should be small relative to the gaps being claimed,
    and ``fit_quality`` says so in words when it is not.

    Parameters
    ----------
    Y : array-like
        Outcomes: long format with ``unit_id``/``time_id``, or a
        (n_units, n_periods) matrix.
    unit_id, time_id : array-like
        Identifiers matching ``Y`` in long format.
    treated_unit : scalar
        The treated unit's identifier.
    treatment_time : scalar
        First treated period.
    n_placebo : int, optional
        Number of donors to use as placebos, largest pre-fit first.
        All of them by default.

    Returns
    -------
    RichResult
        ``estimate`` (mean post-treatment gap), ``att`` (the gap path),
        ``weights``, ``donors``, ``synthetic``, ``pre_rmspe``,
        ``post_rmspe``, ``rmspe_ratio``, ``placebo_p``,
        ``placebo_ratios``, ``p_value_floor``, ``fit_quality``.

    References
    ----------
    Abadie and Gardeazabal (2003), *AER* 93:113-132.
    Abadie, Diamond and Hainmueller (2010), *JASA* 105:493-505.
    Abadie (2021), *Journal of Economic Literature* 59:391-425.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> f = rng.normal(size=20)
    >>> load = np.array([1.0, 0.5, 1.5, 0.2, 1.0])
    >>> Y = np.outer(load, f) + rng.normal(scale=0.01, size=(5, 20))
    >>> Y[0, 12:] += 3.0
    >>> out = synthetic_control(Y, None, None, 0, 12)
    >>> bool(abs(out["estimate"] - 3.0) < 0.1)
    True
    """
    Ya = np.asarray(Y, dtype=float)
    if unit_id is None and time_id is None:
        M = np.atleast_2d(Ya)
        units = np.arange(M.shape[0])
        periods = np.arange(M.shape[1])
    else:
        M, units, periods = as_panel(Ya, unit_id, time_id)
    n_u, T = M.shape
    if n_u < 3:
        raise ValueError(
            "need at least 3 units (one treated, two donors), got %d." % n_u
        )
    idx = np.nonzero(units == treated_unit)[0]
    if idx.size != 1:
        raise ValueError(
            "treated_unit %r is not in the unit set." % (treated_unit,)
        )
    row = int(idx[0])
    tt = np.nonzero(periods >= treatment_time)[0]
    if tt.size == 0:
        raise ValueError(
            "treatment_time %r is after the last period." % (treatment_time,)
        )
    t0 = int(tt[0])
    if t0 < 2:
        raise ValueError(
            "only %d pre-treatment period(s); the weights are fitted on the "
            "pre-period, so there is nothing to fit." % t0
        )
    if t0 >= T:
        raise ValueError("no post-treatment period.")

    w, donors, synth = _fit_one(M, row, t0)
    gap = M[row] - synth
    pre = slice(0, t0)
    post = slice(t0, T)
    pre_r, post_r = _rmspe(gap, pre), _rmspe(gap, post)
    ratio = post_r / pre_r if pre_r > 0 else np.inf

    placebo = {}
    cand = list(donors)
    for j in cand:
        wj, _, sj = _fit_one(M, int(j), t0)
        gj = M[int(j)] - sj
        pj = _rmspe(gj, pre)
        placebo[float(units[int(j)])] = {
            "pre_rmspe": pj,
            "post_rmspe": _rmspe(gj, post),
            "ratio": (_rmspe(gj, post) / pj) if pj > 0 else np.inf,
        }
    if n_placebo is not None:
        keep = sorted(placebo, key=lambda k: placebo[k]["pre_rmspe"])
        keep = set(keep[: int(n_placebo)])
        placebo = {k: v for k, v in placebo.items() if k in keep}
    ratios = np.array([v["ratio"] for v in placebo.values()])
    p = float((1 + np.sum(ratios >= ratio)) / (1 + ratios.size))

    scale = float(np.mean(np.abs(M[row, pre])))
    quality = (
        "good: the pre-treatment fit is small relative to the outcome"
        if pre_r < 0.1 * max(scale, 1e-12)
        else "poor: the synthetic unit does not track the treated unit "
        "before treatment, so the post-treatment gap is not evidence of an "
        "effect"
    )

    return RichResult(
        payload={
            "estimate": float(gap[post].mean()),
            "att": gap[post],
            "gap": gap,
            "weights": w,
            "donors": units[donors],
            "n_donors_used": int((w > 1e-8).sum()),
            "synthetic": synth,
            "treated_path": M[row],
            "pre_rmspe": pre_r,
            "post_rmspe": post_r,
            "rmspe_ratio": ratio,
            "placebo_p": p,
            "placebo_ratios": placebo,
            "p_value_floor": float(1.0 / (1 + ratios.size)),
            "inference_note": (
                "permutation inference over donors: with one treated unit "
                "there is no sampling distribution, so the p-value cannot be "
                "finer than 1/(J+1)"
            ),
            "convexity_note": (
                "weights are non-negative and sum to one, so the synthetic "
                "unit is an average of real units and cannot extrapolate"
            ),
            "fit_quality": quality,
            "t0_index": t0,
            "n_units": int(n_u),
            "n_periods": int(T),
            "method": "Synthetic control (Abadie-Diamond-Hainmueller)",
        }
    )


def cheatsheet():
    return (
        "synct: simplex-weighted synthetic control with placebo permutation "
        "inference and an explicit pre-fit quality check"
    )
