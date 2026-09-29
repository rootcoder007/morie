"""Stratified mean estimator."""

from __future__ import annotations

import math
from statistics import NormalDist

from ._containers import DescriptiveResult


def stratified_mean(
    data,
    *,
    y: str = "y",
    strata: str = "stratum",
    pop_sizes: dict | None = None,
) -> DescriptiveResult:
    r"""Stratified population mean with its standard error (Cochran 1977, ch. 5).

    ``ybar_st = sum_h W_h ybar_h`` with ``W_h = N_h / N`` when the stratum
    population sizes ``pop_sizes`` are given, and the proportional weights
    ``W_h = n_h / n`` otherwise. The variance estimator under stratified
    simple random sampling without replacement is
    ``v(ybar_st) = sum_h W_h^2 (1 - n_h / N_h) s_h^2 / n_h`` (Cochran eq. 5.7);
    without ``pop_sizes`` the finite population correction is taken as 1.
    The confidence limits are ``ybar_st -/+ z_{0.975} se``. This is
    ``survey::svymean`` on ``svydesign(ids = ~1, strata = ~stratum, fpc =
    ~N_h)``.

    Parameters
    ----------
    data : mapping of column name to values (a dict of lists or a data frame).
    y : outcome column. Default ``"y"``.
    strata : stratum column. Default ``"stratum"``.
    pop_sizes : dict mapping stratum to its population size ``N_h`` (optional).

    Returns
    -------
    DescriptiveResult
        ``value`` (the stratified mean); ``extra`` has ``se``, ``ci_lower``,
        ``ci_upper``, ``weights``, ``strata_means`` and ``n_strata``.

    References
    ----------
    Cochran, W. G. (1977). *Sampling Techniques*, 3rd edn. Wiley, sections 5.2-5.3.

    Examples
    --------
    >>> d = {"y": [1, 2, 3, 10, 11, 12, 14], "stratum": ["a", "a", "a", "b", "b", "b", "b"]}
    >>> r = stratified_mean(d, pop_sizes={"a": 30, "b": 10})
    >>> r.value, round(r.extra["se"], 12)
    (4.4375, 0.442824739598)
    """
    try:
        yv = [float(v) for v in data[y]]
        sv = list(data[strata])
    except KeyError as e:
        raise ValueError(f"data lacks column {e}") from None
    if len(yv) != len(sv) or not yv:
        raise ValueError("outcome and stratum columns must be non-empty and of equal length")
    groups: dict = {}
    for v, s in zip(yv, sv):
        groups.setdefault(s, []).append(v)
    n_tot = len(yv)
    if pop_sizes is not None:
        missing = [s for s in groups if s not in pop_sizes]
        if missing:
            raise ValueError(f"pop_sizes lacks strata {missing}")
        N = math.fsum(float(pop_sizes[s]) for s in groups)
    means, W, var_terms = {}, {}, []
    for s, g in groups.items():
        n_h = len(g)
        m = math.fsum(g) / n_h
        s2 = math.fsum((v - m) ** 2 for v in g) / (n_h - 1) if n_h > 1 else float("nan")
        means[s] = m
        if pop_sizes is None:
            W[s], fpc = n_h / n_tot, 1.0
        else:
            N_h = float(pop_sizes[s])
            W[s], fpc = N_h / N, 1.0 - n_h / N_h
        var_terms.append(W[s] * W[s] * fpc * s2 / n_h)
    est = math.fsum(W[s] * means[s] for s in groups)
    se = math.sqrt(math.fsum(var_terms))
    z = NormalDist().inv_cdf(0.975)
    return DescriptiveResult(
        name="Stratified mean",
        value=est,
        extra={
            "se": se,
            "ci_lower": est - z * se,
            "ci_upper": est + z * se,
            "weights": W,
            "strata_means": means,
            "n_strata": len(groups),
        },
    )


strat = stratified_mean


def cheatsheet() -> str:
    return "stratified_mean(data, y, strata, pop_sizes) -> stratified mean, SE with fpc (Cochran ch. 5)."


# compact alias per ledger/NAMING.md
stratifiedmean = stratified_mean
