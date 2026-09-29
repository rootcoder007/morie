"""Test of the monotonicity assumption (instrument validity)."""

import math

from ._richresult import RichResult
from ._rng import random_uniform


def _stats(y, d, z, sets, xi):
    """phi-hat, sigma-hat and T_n for every (h, g) of the binary-instrument class."""
    n = len(y)
    m0 = sum(1 for v in z if v == 0)
    m1 = n - m0
    Tn = m0 * m1 / n
    p0, p1 = m0 / n, m1 / n
    phi, sig = [], []
    for kind, a, b, dd in sets:
        # h = sign * indicator; P-hat(h g_k) / P-hat(g_k) with g_1 = 1{Z=0}, g_2 = 1{Z=1}
        if kind == "B":
            ind = [1.0 if (a <= y[i] <= b and d[i] == dd) else 0.0 for i in range(n)]
            sgn = -1.0 if dd == 1 else 1.0
        else:
            ind = [1.0 if d[i] == 0 else 0.0 for i in range(n)]
            sgn = 1.0
        q1 = sum(ind[i] for i in range(n) if z[i] == 1) / n
        q0 = sum(ind[i] for i in range(n) if z[i] == 0) / n
        phi.append(sgn * (q1 / p1 - q0 / p0))
        # h^2 = indicator; (21) of Sun (2023)
        v = Tn / n * (q1 / p1**2 - q1**2 / p1**3 + q0 / p0**2 - q0**2 / p0**3)
        sig.append(math.sqrt(max(v, 0.0)))
    return phi, sig, Tn


def bound_monotone_test(y, D, Z, xi=0.07, tau=2.0, n_boot=500, seed=0):
    r"""Kitagawa (2015) test of instrument validity (exclusion + monotonicity), in Sun's (2023) bootstrap form.

    With a binary treatment ``D`` and a binary instrument ``Z`` (``Z = 1``
    raising take-up), IV validity implies, for every closed interval ``B``,
    ``P(Y in B, D = 1 | Z = 1) >= P(Y in B, D = 1 | Z = 0)``, ``P(Y in B, D = 0
    | Z = 0) >= P(Y in B, D = 0 | Z = 1)`` and ``P(D = 0 | Z = 1) <= P(D = 0 |
    Z = 0)`` (Kitagawa 2015; Mourifie and Wan 2017). Writing each as
    ``phi(h, g) <= 0``, the statistic is the variance-weighted KS

    ``TS = sup sqrt(T_n) phi-hat / max(xi, sigma-hat)``, ``T_n = m0 m1 / n``,

    over intervals with endpoints at the observed ``Y`` values, with
    ``sigma-hat`` the plug-in standard error of Sun (2023, eq. 21) and
    trimming ``xi`` (0.07 suggested by Kitagawa). The critical value comes
    from ``n_boot`` nonparametric bootstrap draws (Philox stream ``b`` of
    ``seed``) of ``sup sqrt(T_n*) (phi* - phi-hat) / max(xi, sigma*)`` over
    the estimated contact set ``{sqrt(T_n)|phi-hat| / max(0.001, sigma-hat) <=
    tau}`` (Sun 2023, eqs. 27-30, ``tau = 2`` recommended below n = 3000);
    for binary ``D`` and ``Z`` the statistic is Kitagawa's. ``p_value`` is
    the share of bootstrap statistics at least ``TS``.

    References
    ----------
    Kitagawa, T. (2015). A test for instrument validity. *Econometrica* 83,
    2043-2063.
    Sun, Z. (2023). Instrument validity for heterogeneous causal effects.
    *Journal of Econometrics* 237, 105523.

    Examples
    --------
    >>> import math
    >>> z = [i % 2 for i in range(200)]
    >>> d = [1 if (math.sin(2.7 * i) + 0.8 * z[i] > 0.3) else 0 for i in range(200)]
    >>> y = [round(2 * (math.cos(1.3 * i) + d[i])) / 2 for i in range(200)]
    >>> round(bound_monotone_test(y, d, z, n_boot=99)["statistic"], 10)
    0.0
    >>> r = bound_monotone_test(y, [1 - v for v in d], z, n_boot=99)
    >>> round(r["statistic"], 10), r["p_value"]
    (3.8111861231, 0.0)
    """
    yv = [float(v) for v in y]
    dv = [int(v) for v in D]
    zv = [int(v) for v in Z]
    n = len(yv)
    if set(dv) - {0, 1} or set(zv) - {0, 1} or len(set(zv)) < 2:
        raise ValueError("D and Z must be binary and Z must take both values")
    grid = sorted(set(yv))
    sets = [("B", grid[a], grid[b], dd) for dd in (0, 1) for a in range(len(grid)) for b in range(a, len(grid))]
    sets.append(("C", None, None, None))
    phi, sig, Tn = _stats(yv, dv, zv, sets, xi)
    ts = max(math.sqrt(Tn) * p / max(xi, s) for p, s in zip(phi, sig))
    contact = [k for k in range(len(sets)) if math.sqrt(Tn) * abs(phi[k]) / max(0.001, sig[k]) <= tau]
    boot = []
    for bb in range(int(n_boot)):
        u = random_uniform(n, seed=seed, stream=bb)
        idx = [min(n - 1, int(math.floor(float(v) * n))) for v in u]
        zb = [zv[i] for i in idx]
        if len(set(zb)) < 2:
            continue
        pb, sb, Tb = _stats([yv[i] for i in idx], [dv[i] for i in idx], zb, [sets[k] for k in contact], xi)
        boot.append(
            max(math.sqrt(Tb) * (pb[j] - phi[k]) / max(xi, sb[j]) for j, k in enumerate(contact)) if contact else 0.0
        )
    pval = sum(1 for v in boot if v >= ts) / len(boot)
    srt = sorted(boot)
    crit = srt[min(len(srt) - 1, int(math.ceil(0.95 * len(srt))) - 1)]
    return RichResult(
        payload={
            "statistic": ts,
            "p_value": pval,
            "critical_value_05": crit,
            "n_boot": len(boot),
            "contact_set_size": len(contact),
            "xi": xi,
            "tau": tau,
            "T_n": Tn,
            "method": "Kitagawa (2015) / Sun (2023) variance-weighted KS test of IV validity",
        }
    )


def cheatsheet():
    return "bndmnt: Kitagawa (2015) IV-validity (monotonicity) test, Sun (2023) bootstrap"
