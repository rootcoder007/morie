# morie.fn -- function file (rootcoder007/morie)
"""Ferro-Segers intervals estimator of the extremal index."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ev_extremal_intervals", "evt_extremal_index_intervals"]


def ev_extremal_intervals(x, threshold):
    r"""The intervals estimator of Ferro and Segers (2003),

    .. math:: \hat\theta = \begin{cases}
              \min\!\left(1, \dfrac{2(\sum T_i)^2}
              {N\sum T_i^2}\right) & \max T_i \le 2 \\[1ex]
              \min\!\left(1, \dfrac{2\{\sum(T_i-1)\}^2}
              {N\sum(T_i-1)(T_i-2)}\right) & \max T_i > 2,
              \end{cases}

    with :math:`T_i` the interexceedance times (their Eqs. (4) and
    (34)). Its selling point over the runs estimator is exactly what
    the paper's title says: NO tuning parameter. The interexceedance
    times of a process with extremal index :math:`\theta` converge to
    a mixture -- a point mass of within-cluster short gaps and an
    exponential of between-cluster long ones -- and the moment ratio
    above identifies :math:`\theta` from that mixture without ever
    declaring which gap is which.

    The case split matters: the first form is biased when clusters
    are present (it ignores the -1/-2 corrections) and the second is
    undefined when every gap is 1 or 2; Ferro and Segers' rule uses
    whichever is valid, and this module follows their rule rather
    than picking one form.

    Parameters
    ----------
    x : array-like
        Stationary series.
    threshold : float
        Exceedance threshold.

    Returns
    -------
    RichResult
        keys: ``theta``, ``n_exceedances``, ``form_used``,
        ``mean_interexceedance``, ``max_interexceedance``,
        ``implied_mean_cluster_size``, ``threshold``, ``n``,
        ``method``.

    References
    ----------
    Ferro, C. A. T. and Segers, J. (2003), "Inference for clusters
    of extreme values", *JRSS-B* 65:545-556, Eqs. (4) and (34).
    """
    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 20:
        raise ValueError(f"need at least 20 observations, got {n}.")
    u = float(threshold)
    exc = np.flatnonzero(xv > u)
    N = exc.size
    if N < 3:
        raise ValueError(
            f"only {N} exceedance(s) of {u}; lower the threshold.")
    T = np.diff(exc).astype(float)
    if T.max() <= 2:
        theta = min(1.0, 2.0 * T.sum() ** 2 / ((N - 1) * np.sum(T ** 2)))
        form = "Eq. (4): max gap <= 2, uncorrected moments"
    else:
        num = 2.0 * np.sum(T - 1.0) ** 2
        den = (N - 1) * np.sum((T - 1.0) * (T - 2.0))
        theta = min(1.0, num / den)
        form = "Eq. (34): gaps beyond 2 present, corrected moments"
    return RichResult(payload={
        "theta": float(theta), "n_exceedances": int(N),
        "form_used": form,
        "mean_interexceedance": float(T.mean()),
        "max_interexceedance": float(T.max()),
        "implied_mean_cluster_size": float(1.0 / theta) if theta > 0
        else np.inf,
        "no_tuning_note": "unlike the runs estimator there is no run-length "
                          "parameter: the interexceedance-time mixture "
                          "identifies theta by a moment ratio",
        "threshold": u, "n": int(n),
        "method": "Intervals estimator of the extremal index (Ferro-Segers 2003)"})


def cheatsheet():
    return "evextint: theta from interexceedance-time moments -- no run length to tune"


#: Catalogue alias for :func:`ev_extremal_intervals`.
evt_extremal_index_intervals = ev_extremal_intervals
