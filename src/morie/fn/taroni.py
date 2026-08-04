"""Tarone-Ware and the rest of the weighted log-rank family."""

from math import sqrt

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import hypothesis_test_result

__all__ = ["taroni_ware"]

_WEIGHTS = ("tarone-ware", "logrank", "gehan", "peto")


def _risk_table(time, event, group):
    t = [float(v) for v in np.asarray(time, dtype=float).ravel().tolist()]
    e = [float(v) for v in np.asarray(event, dtype=float).ravel().tolist()]
    g = [v for v in np.asarray(group).ravel().tolist()]
    if not (len(t) == len(e) == len(g)):
        raise ValueError("time, event and group must have the same length.")
    labs = list(dict.fromkeys(g))
    if len(labs) != 2:
        raise ValueError("need exactly 2 groups; got %d." % len(labs))
    a = labs[0]
    rows = []
    for tt in sorted({t[i] for i in range(len(t)) if e[i] == 1}):
        n = sum(1 for i in range(len(t)) if t[i] >= tt)
        n1 = sum(1 for i in range(len(t)) if t[i] >= tt and g[i] == a)
        d = sum(1 for i in range(len(t)) if t[i] == tt and e[i] == 1)
        d1 = sum(1 for i in range(len(t)) if t[i] == tt and e[i] == 1 and g[i] == a)
        rows.append((tt, n, n1, d, d1))
    if not rows:
        raise ValueError("no events.")
    return rows, labs


def taroni_ware(time, event, group, weight="tarone-ware"):
    r"""Two-sample weighted log-rank test.

    At each distinct event time :math:`t_j` with :math:`n_j` at risk,
    :math:`n_{1j}` of them in group 1, and :math:`d_j` deaths of which
    :math:`d_{1j}` fall in group 1, the hypergeometric mean and
    variance are

    .. math::

       e_{1j}=\frac{d_jn_{1j}}{n_j},\qquad
       V_j=\frac{d_j(n_j-d_j)\,n_{1j}(n_j-n_{1j})}{n_j^2\,(n_j-1)} .

    The family is then

    .. math::

       \chi^2=\frac{\bigl[\sum_j w_j(d_{1j}-e_{1j})\bigr]^2}
                   {\sum_j w_j^2 V_j}\;\sim\;\chi^2_1 ,

    and the members differ only in :math:`w_j`:

    ================  ==================  ==============================
    ``weight``        :math:`w_j`         emphasises
    ================  ==================  ==============================
    ``"logrank"``     1                   late differences equally
    ``"gehan"``       :math:`n_j`         early differences (Wilcoxon)
    ``"tarone-ware"`` :math:`\sqrt{n_j}`  a compromise between the two
    ``"peto"``        :math:`\hat S(t_j)` early, but less size-dependent
    ================  ==================  ==============================

    Tarone-Ware's :math:`\sqrt{n_j}` sits deliberately between the
    log-rank and Gehan weights: the log-rank is most powerful under
    proportional hazards, Gehan under early separation, and
    :math:`\sqrt{n_j}` gives up a little of each rather than betting on
    one alternative. Choosing the weight *after* seeing the curves
    invalidates the p-value.

    Parameters
    ----------
    time : array-like
        Follow-up times.
    event : array-like of {0, 1}
        1 for an event, 0 for right-censored.
    group : array-like
        Exactly two distinct labels.
    weight : {"tarone-ware", "logrank", "gehan", "peto"}

    Returns
    -------
    RichResult
        Keys ``statistic``, ``pvalue``, ``df``, ``observed``,
        ``expected``, ``variance``, ``n_events``, ``groups``, ``weight``.

    References
    ----------
    Tarone, R. E. & Ware, J. (1977). On distribution-free tests for
    equality of survival distributions. *Biometrika*, 64(1), 156-160.
    Gehan, E. A. (1965). *Biometrika*, 52, 203-223.
    Weight definitions cross-checked against the reference
    implementation in survMisc (``comp.ten``), where the Tarone-Ware
    column is ``sqrt(n)``.
    """
    if weight not in _WEIGHTS:
        raise ValueError("weight must be one of %s" % (_WEIGHTS,))
    rows, labs = _risk_table(time, event, group)
    # Peto weight is the left-continuous modified KM estimate, so it is
    # built by a running product over the same event times.
    s, peto = 1.0, []
    for (_, n, _, d, _) in rows:
        peto.append(s)
        s *= 1.0 - d / (n + 1.0)
    num = den = obs = exp = 0.0
    for j, (_, n, n1, d, d1) in enumerate(rows):
        if n <= 1:
            continue
        w = {"logrank": 1.0, "gehan": float(n),
             "tarone-ware": sqrt(n), "peto": peto[j]}[weight]
        e1 = d * n1 / n
        v = d * (n - d) * n1 * (n - n1) / (n * n * (n - 1.0))
        num += w * (d1 - e1)
        den += w * w * v
        obs += d1
        exp += e1
    if den <= 0:
        raise ValueError("zero variance; the groups cannot be compared.")
    stat = num * num / den
    return hypothesis_test_result(
        test_name="Weighted log-rank test (%s)" % weight,
        statistic=float(stat),
        pvalue=float(stats.chi2.sf(stat, 1)),
        extra_summary=[("observed", obs), ("expected", exp)],
        extra_payload={
            "df": 1,
            "observed": float(obs),
            "expected": float(exp),
            "score": float(num),
            "variance": float(den),
            "n_events": int(sum(r[3] for r in rows)),
            "n_event_times": len(rows),
            "groups": [str(v) for v in labs],
            "weight": weight,
            "method": "Tarone-Ware (1977) family of weighted log-rank tests",
        },
    )


def cheatsheet():
    return "taroni: Tarone-Ware / log-rank / Gehan / Peto weighted log-rank test"
