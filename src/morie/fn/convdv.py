# morie.fn -- function file (rootcoder007/morie)
"""Csiszar f-divergence between two discrete distributions.

Source: Csiszar, I. (1967), "Information-type measures of difference of
probability distributions and indirect observations", *Studia
Scientiarum Mathematicarum Hungarica* 2:299-318.  For a convex f with
f(1) = 0,

    D_f(p || q) = sum_i q_i f(p_i / q_i)

which is the expression the ledger records and the one implemented.

The two boundary conventions are those that make D_f lower
semicontinuous, and are applied here explicitly rather than left to
floating-point accident:

    q_i = 0, p_i = 0  contributes 0
    q_i = 0, p_i > 0  contributes p_i * f'(inf), the recession constant,
                      supplied by the caller as ``f_inf``; the default
                      is +inf, so a p not absolutely continuous with
                      respect to q gives +inf, as it should.

``f`` is a caller-supplied convex function of one non-negative
argument.  Named choices are accepted as strings so that the common
divergences do not have to be rewritten by every caller:

    "kl"      f(t) = t log t              Kullback-Leibler D(p||q)
    "rkl"     f(t) = -log t               reverse KL D(q||p)
    "tv"      f(t) = |t - 1| / 2          total variation
    "chi2"    f(t) = (t - 1)^2            Pearson chi-square
    "hellinger"  f(t) = (sqrt(t) - 1)^2   squared Hellinger
    "js"      f(t) = t log t - (t+1) log((t+1)/2)   Jensen-Shannon

f(1) = 0 is checked for callables, since a shifted f changes the value
by a constant and is the most common way to get a wrong answer here.
Convexity is NOT checked: it cannot be from finitely many evaluations,
and a claim to have checked it would be false.
"""

import math

from ._richresult import RichResult

__all__ = ["convex_divergence"]

_NAMED = {
    "kl": lambda t: t * math.log(t) if t > 0.0 else 0.0,
    "rkl": lambda t: -math.log(t) if t > 0.0 else float("inf"),
    "tv": lambda t: 0.5 * abs(t - 1.0),
    "chi2": lambda t: (t - 1.0) ** 2,
    "hellinger": lambda t: (math.sqrt(t) - 1.0) ** 2,
    "js": lambda t: (t * math.log(t) if t > 0.0 else 0.0)
    - (t + 1.0) * math.log((t + 1.0) / 2.0),
}
_NAMED_INF = {"kl": float("inf"), "rkl": 0.0, "tv": 0.5, "chi2": float("inf"),
              "hellinger": 1.0, "js": math.log(2.0)}


def convex_divergence(p, q, f="kl", f_inf=None, normalise=True):
    """f-divergence D_f(p || q) = sum q_i f(p_i / q_i).

    Parameters
    ----------
    p, q : array-like
        Non-negative weights of equal length.
    f : callable or {"kl","rkl","tv","chi2","hellinger","js"}
        Convex generator with f(1) = 0.
    f_inf : float, optional
        Recession constant lim_{t->inf} f(t)/t, charged against mass
        that p places where q is zero.  Defaults to the exact value for
        a named generator and to +inf for a callable.
    normalise : bool
        Divide p and q by their totals first.  Turn off only when the
        inputs are already probability vectors.

    Returns
    -------
    RichResult
        ``divergence``, ``estimate`` (the same number), ``terms``,
        ``support`` (count of i contributing a finite non-zero term),
        ``n``.
    """
    a = [float(v) for v in p]
    b = [float(v) for v in q]
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("p and q must be non-empty and of equal length")
    for v in a + b:
        if v < 0.0:
            raise ValueError("p and q must be non-negative")
    if isinstance(f, str):
        key = f.lower()
        if key not in _NAMED:
            raise ValueError("unknown generator %r" % (f,))
        fn = _NAMED[key]
        if f_inf is None:
            f_inf = _NAMED_INF[key]
        name = key
    else:
        fn = f
        if f_inf is None:
            f_inf = float("inf")
        if abs(float(fn(1.0))) > 1e-12:
            raise ValueError("generator must satisfy f(1) = 0")
        name = "callable"
    if normalise:
        sa = sum(a)
        sb = sum(b)
        if sa <= 0.0 or sb <= 0.0:
            raise ValueError("p and q must each carry positive total mass")
        a = [v / sa for v in a]
        b = [v / sb for v in b]
    terms = []
    total = 0.0
    support = 0
    for i in range(n):
        if b[i] > 0.0:
            t = float(fn(a[i] / b[i])) * b[i]
        elif a[i] > 0.0:
            t = a[i] * float(f_inf)
        else:
            t = 0.0
        terms.append(t)
        total += t
        if t != 0.0:
            support += 1
    return RichResult(payload={
        "divergence": float(total), "estimate": float(total),
        "terms": [float(v) for v in terms], "support": support,
        "generator": name, "f_inf": float(f_inf), "n": n,
        "method": "Csiszar (1967) f-divergence, sum q f(p/q)"})


def cheatsheet():
    return "convdv: Csiszar (1967) f-divergence"
