# morie.fn -- function file (rootcoder007/morie)
"""Stratified proportion estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["stratified_proportion"]


def stratified_proportion(y, stratum, weights=None, N_h=None):
    r"""Stratified estimator of a proportion:

    .. math:: \hat p_{st} = \sum_h W_h \hat p_h,
              \qquad
              \widehat{\operatorname{Var}}(\hat p_{st})
              = \sum_h W_h^2 \frac{\hat p_h(1-\hat p_h)}{n_h - 1}.

    The variance is a sum WITHIN strata, with no between-stratum
    term at all, and that is the entire point of stratification:
    variation between strata is removed by design rather than
    estimated. A stratified estimate is never less precise than
    simple random sampling of the same size, whatever the strata,
    provided the weights are the true population shares.

    The weights :math:`W_h` must be POPULATION shares, not sample
    shares. Using sample shares silently reproduces the unstratified
    estimate and throws the design away, so they are validated to
    sum to one and ``weights_are_population_shares`` records the
    requirement.

    Parameters
    ----------
    y : array-like of {0, 1}
        Binary responses.
    stratum : array-like
        Stratum labels.
    weights : array-like, optional
        Population share per stratum, in label order. Derived from
        ``N_h`` or from sample shares otherwise.
    N_h : mapping or array-like, optional
        Population sizes per stratum.

    Returns
    -------
    RichResult
        keys: ``proportion``, ``variance``, ``se``, ``strata``,
        ``p_h``, ``n_h``, ``W_h``, ``weights_are_population_shares``,
        ``n``, ``method``.
    """
    yv = np.asarray(y, dtype=float).ravel()
    st = np.asarray(stratum).ravel()
    if st.size != yv.size:
        raise ValueError(f"stratum has {st.size} entries for {yv.size} of y.")
    if not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError("y must be binary 0/1 for a proportion.")
    labs = np.unique(st)
    if labs.size < 2:
        raise ValueError("need at least 2 strata.")
    ph, nh = [], []
    for lab in labs:
        sel = st == lab
        m = int(sel.sum())
        if m < 2:
            raise ValueError(f"stratum {lab!r} has {m} units; need at least 2 "
                             "to estimate a within-stratum variance.")
        nh.append(m)
        ph.append(float(yv[sel].mean()))
    ph = np.array(ph); nh = np.array(nh, dtype=float)
    if weights is not None:
        W = np.atleast_1d(np.asarray(weights, dtype=float)).ravel()
        if W.size != labs.size:
            raise ValueError(f"weights has {W.size} entries for {labs.size} strata.")
        pop = True
    elif N_h is not None:
        Nv = np.atleast_1d(np.asarray(
            [N_h[l] for l in labs] if hasattr(N_h, "__getitem__")
            and not isinstance(N_h, (list, tuple, np.ndarray)) else N_h,
            dtype=float)).ravel()
        W = Nv / Nv.sum()
        pop = True
    else:
        W = nh / nh.sum()
        pop = False
    if not np.isclose(W.sum(), 1.0):
        raise ValueError(f"stratum weights must sum to 1, got {W.sum()}.")
    p = float(np.sum(W * ph))
    var = float(np.sum(W ** 2 * ph * (1 - ph) / (nh - 1)))
    return RichResult(payload={
        "proportion": p, "variance": var, "se": float(np.sqrt(max(var, 0.0))),
        "strata": labs, "p_h": ph, "n_h": nh.astype(int), "W_h": W,
        "weights_are_population_shares": pop,
        "variance_note": "within-stratum only: between-stratum variation is "
                         "removed by DESIGN, not estimated",
        "n": int(yv.size),
        "method": "Stratified proportion; W_h must be POPULATION shares or the design is discarded"})


def cheatsheet():
    return "straprp: no between-stratum variance term exists -- that is what stratification buys"
