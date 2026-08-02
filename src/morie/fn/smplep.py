# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dual-frame estimation with sample overlap (Hartley 1962)."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sample_overlap", "optimal_overlap_weight"]

_METHOD = "Hartley dual-frame estimator with overlap weighting"


def _z(q):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _domain_total(y, w, mask):
    """Weighted total over one domain, with its sampling variance."""
    if not np.any(mask):
        return 0.0, 0.0
    contrib = w[mask] * y[mask]
    total = float(np.sum(contrib))
    m = contrib.size
    if m < 2:
        return total, 0.0
    # with-replacement approximation: Var(sum) = m * Var(contribution)
    return total, float(m * np.var(contrib, ddof=1))


def optimal_overlap_weight(var_a, var_b):
    r"""The variance-minimising split of the overlap domain.

    Hartley's estimator is unbiased for *every* :math:`\theta` in
    :math:`[0, 1]`, so the choice is purely one of efficiency. Treating
    the two frames' samples as independent, the variance of
    :math:`\theta \hat Y_{ab}^A + (1-\theta) \hat Y_{ab}^B` is
    :math:`\theta^2 V_A + (1-\theta)^2 V_B`, minimised at

    .. math:: \theta^* = \frac{V_B}{V_A + V_B}.

    Note the direction: weight goes to the frame that measures the
    overlap *more* precisely, since :math:`\theta` multiplies frame A's
    estimate and rises as frame B's variance rises.
    """
    va, vb = float(var_a), float(var_b)
    if va < 0 or vb < 0:
        raise ValueError("variances must be non-negative.")
    if va + vb <= 0:
        return 0.5
    return vb / (va + vb)


def sample_overlap(frame_a, frame_b, overlap_a, overlap_b,
                   weights_a=None, weights_b=None, theta=None, alpha=0.05):
    r"""Combine two overlapping sampling frames into one estimate.

    Two frames :math:`A` and :math:`B` together cover the population but
    neither alone does, and they overlap. Splitting into the three
    domains :math:`a` (frame A only), :math:`ab` (both) and :math:`b`
    (frame B only), Hartley's estimator is

    .. math::
        \hat Y = \hat Y_a + \theta \hat Y_{ab}^A
                 + (1 - \theta)\hat Y_{ab}^B + \hat Y_b .

    The overlap domain is measured twice, once through each sample, and
    the two measurements are blended rather than added.

    The failure mode this guards against is the one that looks like
    success. **Pooling the two samples double-counts the overlap.**
    Units in :math:`ab` could have been selected through either frame,
    so adding the frame totals inflates the estimate by roughly the
    size of the overlap domain -- and the inflated estimate is
    perfectly stable across replications, because it is a bias, not
    noise. More data makes it more precisely wrong.
    ``naive_pooled_total`` is returned so the gap is visible rather
    than argued about.

    What the estimator needs and cannot check is that the overlap
    indicators are correct: every sampled unit must know whether it was
    reachable through the other frame. Misclassifying overlap units as
    frame-only reintroduces exactly the double count.

    Parameters
    ----------
    frame_a, frame_b : array-like
        Observed values in the sample drawn from each frame.
    overlap_a, overlap_b : array-like
        1 if that unit also belongs to the other frame, else 0.
    weights_a, weights_b : array-like, optional
        Design weights (inverse selection probabilities). Default 1.
    theta : float, optional
        Overlap weight. Defaults to the variance-minimising value.
    alpha : float
        Two-sided level.

    Returns
    -------
    RichResult
        ``estimate``, ``theta``, ``theta_optimal``, ``se``,
        ``ci_lower``/``ci_upper``, the four domain totals,
        ``naive_pooled_total``, ``overlap_double_count``,
        ``variance_ratio_vs_optimal``.

    References
    ----------
    Hartley HO (1962) Multiple frame surveys, *Proceedings of the
    Social Statistics Section*, American Statistical Association,
    203-206. Lohr SL, Rao JNK (2000) *JASA* 95(449):271-280.

    Examples
    --------
    >>> ya = [1.0, 1.0, 1.0, 1.0]
    >>> yb = [1.0, 1.0, 1.0]
    >>> out = sample_overlap(ya, yb, [0, 0, 1, 1], [1, 1, 0], theta=0.5)
    >>> out["estimate"]
    5.0
    >>> out["naive_pooled_total"]
    7.0
    """
    ya = np.asarray(frame_a, dtype=float).ravel()
    yb = np.asarray(frame_b, dtype=float).ravel()
    da = np.asarray(overlap_a, dtype=float).ravel()
    db = np.asarray(overlap_b, dtype=float).ravel()
    if da.size != ya.size:
        raise ValueError(
            f"overlap_a has length {da.size} but frame_a has {ya.size}."
        )
    if db.size != yb.size:
        raise ValueError(
            f"overlap_b has length {db.size} but frame_b has {yb.size}."
        )
    if ya.size < 1 or yb.size < 1:
        raise ValueError("both frames must contribute at least one unit.")
    if not (np.all(np.isin(da, (0.0, 1.0)))
            and np.all(np.isin(db, (0.0, 1.0)))):
        raise ValueError("overlap indicators must be binary 0/1.")
    wa = (np.ones_like(ya) if weights_a is None
          else np.asarray(weights_a, dtype=float).ravel())
    wb = (np.ones_like(yb) if weights_b is None
          else np.asarray(weights_b, dtype=float).ravel())
    if wa.size != ya.size or wb.size != yb.size:
        raise ValueError("weights must match their frame's length.")
    if np.any(wa <= 0) or np.any(wb <= 0):
        raise ValueError("design weights must be positive.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")

    t_a, v_a = _domain_total(ya, wa, da == 0)         # frame A only
    t_b, v_b = _domain_total(yb, wb, db == 0)         # frame B only
    t_ab_a, v_ab_a = _domain_total(ya, wa, da == 1)   # overlap, via A
    t_ab_b, v_ab_b = _domain_total(yb, wb, db == 1)   # overlap, via B

    th_opt = optimal_overlap_weight(v_ab_a, v_ab_b)
    if theta is None:
        th, chosen = th_opt, "optimal"
    else:
        th = float(theta)
        if not 0 <= th <= 1:
            raise ValueError(f"theta must lie in [0, 1]; got {theta}.")
        chosen = "user"

    overlap_est = th * t_ab_a + (1.0 - th) * t_ab_b
    est = t_a + overlap_est + t_b
    var = v_a + v_b + th ** 2 * v_ab_a + (1.0 - th) ** 2 * v_ab_b
    var_opt = (v_a + v_b + th_opt ** 2 * v_ab_a
               + (1.0 - th_opt) ** 2 * v_ab_b)
    se = math.sqrt(max(var, 0.0))

    naive = t_a + t_ab_a + t_ab_b + t_b
    inflation = naive - est

    zc = _z(1 - alpha / 2)
    out = RichResult(
        title="Dual-frame total with sample overlap",
        summary_lines=[
            ("Combined total", est),
            ("theta used", th),
            ("theta optimal", th_opt),
            ("SE", se),
            ("Naive pooled total", naive),
        ],
        tables=[{
            "title": "Domain totals",
            "headers": ["Domain", "Total", "Variance"],
            "rows": [
                ["a  (frame A only)", t_a, v_a],
                ["ab (both, via A)", t_ab_a, v_ab_a],
                ["ab (both, via B)", t_ab_b, v_ab_b],
                ["b  (frame B only)", t_b, v_b],
            ],
        }],
        payload={
            "estimate": est,
            "theta": th,
            "theta_optimal": th_opt,
            "theta_source": chosen,
            "se": se,
            "variance": var,
            "variance_optimal": var_opt,
            "variance_ratio_vs_optimal": (var / var_opt if var_opt > 0
                                          else float("nan")),
            "ci_lower": est - zc * se,
            "ci_upper": est + zc * se,
            "total_a_only": t_a,
            "total_b_only": t_b,
            "total_overlap_via_a": t_ab_a,
            "total_overlap_via_b": t_ab_b,
            "overlap_estimate": overlap_est,
            "naive_pooled_total": naive,
            "overlap_double_count": inflation,
            "n_a": int(ya.size),
            "n_b": int(yb.size),
            "n_overlap_a": int(np.sum(da == 1)),
            "n_overlap_b": int(np.sum(db == 1)),
            "n": int(ya.size + yb.size),
            "method": _METHOD,
        },
        interpretation=(
            "Pooling the two samples without the overlap weighting would "
            f"return {naive:.6g} against {est:.6g}, an inflation of "
            f"{inflation:.6g} that no amount of extra data would reveal."
        ),
    )
    if np.sum(da == 1) == 0 or np.sum(db == 1) == 0:
        out.warnings.append(
            "One frame contributed no overlap units, so the overlap domain "
            "rests on a single frame and theta has no effect. If the frames "
            "really do overlap, the indicators are wrong."
        )
    if theta is not None and var_opt > 0 and var / var_opt > 1.05:
        out.warnings.append(
            f"The supplied theta = {th:g} carries {var / var_opt:.2f} times "
            f"the variance of the optimal theta = {th_opt:.4g}. The estimate "
            "is still unbiased; it is only less precise."
        )
    return out


def cheatsheet():
    return (
        "smplep: Hartley dual-frame total, blending the twice-measured "
        "overlap domain at the variance-minimising weight"
    )
