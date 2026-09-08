# morie.fn -- function file (rootcoder007/morie)
"""Bayesian-nonparametrics scalar formulas (Ghosal & van der Vaart shelf).

Python arm of the seventeen ``morie_gh_*`` entry points in
``R/bnp_ghosal.R``.  Each function here reproduces the R arm to the
last bit; where the R arm carries a book-level reference only, so does
this module -- no equation number is asserted here that was not read
in the source.

The shared machinery (stick breaking, discrete hazards, dyadic Polya
paths) already lives in :mod:`morie.fn._bnp_core` and is reused rather
than restated.

References
----------
Ghosal, S. and van der Vaart, A. (2017). *Fundamentals of
    Nonparametric Bayesian Inference*. Cambridge University Press.
"""

from __future__ import annotations

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult

__all__ = [
    "ghcrmlg", "ghdirmom", "ghdpmedc", "ghdpnk", "ghdppost", "ghdppred",
    "ghdptrn", "ghewensl", "ghhell2", "ghibpk", "ghkldiv", "ghncrmlap",
    "ghptbits", "ghptdens", "ghpyeppf", "ghrenyi", "ghwnpost",
]


def _vec(x, what):
    if isinstance(x, (int, float)):
        return [float(x)]
    try:
        out = [float(v) for v in x]
    except TypeError:
        raise ValueError(f"{what} must be a number or a sequence of numbers")
    return out


def _probs(p, what):
    ps = _vec(p, what)
    if not ps:
        raise ValueError(f"{what} must not be empty")
    if any(v < 0 for v in ps):
        raise ValueError(f"{what} must be non-negative")
    tot = math.fsum(ps)
    if tot <= 0:
        raise ValueError(f"{what} must have positive total mass")
    return [v / tot for v in ps]


# ------------------------------------------------ completely random measures
def ghcrmlg(f, a):
    r"""Laplace exponent :math:`a \log(1 + f)` of a unit-rate gamma
    completely random measure at the test-function value ``f``.

    The Laplace transform itself is :math:`e^{-a\log(1+f)}`, returned
    alongside.

    Parameters
    ----------
    f : float or sequence of float
        Test-function value(s), each greater than ``-1``.
    a : float
        Base-measure mass of the gamma CRM.

    Returns
    -------
    RichResult
        ``exponent`` (list), ``laplace`` (list), ``a``, ``method``.
    """
    fs = _vec(f, "f")
    if any(v <= -1.0 for v in fs):
        raise ValueError("f must exceed -1 for log(1 + f) to be finite")
    a = float(a)
    ex = [a * math.log(1.0 + v) for v in fs]
    return RichResult(payload={
        "exponent": ex, "laplace": [math.exp(-v) for v in ex],
        "a": a,
        "method": "gamma CRM Laplace exponent a log(1 + f) "
                  "(Ghosal & van der Vaart 2017)"})


def ghncrmlap(f, m, u):
    r"""Laplace transform :math:`\exp\{-\sum_i m_i (1 - e^{-f_i u})\}` of a
    completely random measure whose Levy intensity has been discretised
    into atoms of mass ``m`` at test-function values ``f``.

    The R arm ``morie_gh_ncrm_laplace`` lets R's recycling rule pad the
    shorter of ``f`` and ``m``; this arm raises instead.  Ragged input
    therefore errors here where R silently returned a number computed
    from repeated entries.

    Parameters
    ----------
    f : sequence of float
        Test-function values at the discretisation atoms.
    m : sequence of float
        Atom masses; must be the same length as ``f``.
    u : float
        Argument of the Laplace transform.

    Returns
    -------
    RichResult
        ``laplace``, ``exponent``, ``n_atoms``, ``u``, ``method``.
    """
    fs = _vec(f, "f")
    ms = _vec(m, "m")
    if len(fs) != len(ms):
        raise ValueError(
            f"f has {len(fs)} entries and m has {len(ms)}; "
            "they must align (R would recycle silently)")
    u = float(u)
    ex = math.fsum(mi * (1.0 - math.exp(-fi * u))
                   for fi, mi in zip(fs, ms))
    return RichResult(payload={
        "laplace": math.exp(-ex), "exponent": ex,
        "n_atoms": len(fs), "u": u,
        "method": "discretised CRM Laplace transform "
                  "(Ghosal & van der Vaart 2017)"})


# -------------------------------------------------- Dirichlet process
def ghdppost(g0a, alpha, nina, n):
    r"""Beta posterior of the Dirichlet-process mass :math:`G(A)` of one set.

    The posterior mean is the precision-weighted mixture
    :math:`\frac{\alpha}{\alpha+n}G_0(A) + \frac{n}{\alpha+n}\hat p_n`
    and the variance that of a beta with precision :math:`\alpha + n`.

    Parameters
    ----------
    g0a : float
        Prior base-measure mass :math:`G_0(A)`, in ``[0, 1]``.
    alpha : float
        Dirichlet-process precision, positive.
    nina : float
        Number of observations falling in ``A``.
    n : float
        Total number of observations; the empirical mass is taken as 0
        when ``n`` is zero.

    Returns
    -------
    RichResult
        ``mean``, ``var``, ``precision``, ``method``.
    """
    g0a = float(g0a)
    alpha = float(alpha)
    nina = float(nina)
    n = float(n)
    if not 0.0 <= g0a <= 1.0:
        raise ValueError("g0a must lie in [0, 1]")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if n < 0 or nina < 0 or nina > n:
        raise ValueError("need 0 <= nina <= n")
    pn = nina / n if n > 0 else 0.0
    m = alpha / (alpha + n) * g0a + n / (alpha + n) * pn
    return RichResult(payload={
        "mean": m, "var": m * (1.0 - m) / (1.0 + alpha + n),
        "precision": alpha + n,
        "method": "Dirichlet-process posterior for one set "
                  "(Ghosal & van der Vaart 2017)"})


def ghdpnk(n, alpha):
    r"""Mean and variance of the number of distinct values in ``n`` draws
    from a Dirichlet process of precision ``alpha``:
    :math:`E K_n = \sum_{i=1}^n \alpha/(\alpha+i-1)` and
    :math:`\operatorname{Var} K_n = \sum_{i=1}^n \alpha(i-1)/(\alpha+i-1)^2`.

    Parameters
    ----------
    n : int
        Sample size, positive.
    alpha : float
        Dirichlet-process precision, positive.

    Returns
    -------
    RichResult
        ``mean``, ``var``, ``n``, ``alpha``, ``method``.
    """
    n = int(n)
    alpha = float(alpha)
    if n < 1:
        raise ValueError("n must be a positive integer")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    mean = math.fsum(alpha / (alpha + i - 1) for i in range(1, n + 1))
    var = math.fsum(alpha * (i - 1) / (alpha + i - 1) ** 2
                    for i in range(1, n + 1))
    return RichResult(payload={
        "mean": mean, "var": var, "n": n, "alpha": alpha,
        "method": "expected distinct values in a DP sample "
                  "(Ghosal & van der Vaart 2017)"})


def ghdppred(alpha, n):
    r"""Polya-urn predictive weights after ``n`` observations: a fresh draw
    from the base measure carries :math:`\alpha/(\alpha+n)`, each past
    observation :math:`1/(\alpha+n)`.

    Parameters
    ----------
    alpha : float
        Dirichlet-process precision, positive.
    n : float
        Number of observations already drawn.

    Returns
    -------
    RichResult
        ``weight_fresh``, ``weight_per_obs``, ``method``.
    """
    alpha = float(alpha)
    n = float(n)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    return RichResult(payload={
        "weight_fresh": alpha / (alpha + n),
        "weight_per_obs": 1.0 / (alpha + n),
        "method": "Polya-urn predictive weights of a Dirichlet process "
                  "(Ghosal & van der Vaart 2017)"})


def ghdptrn(eps, alpha):
    r"""Stick-breaking truncation level :math:`2 + \alpha\log(1/\varepsilon)`
    for an expected discarded tail mass of order ``eps``.  Not rounded.

    Parameters
    ----------
    eps : float
        Residual mass tolerance in ``(0, 1)``.
    alpha : float
        Dirichlet-process precision, positive.

    Returns
    -------
    RichResult
        ``level``, ``eps``, ``alpha``, ``method``.
    """
    eps = float(eps)
    alpha = float(alpha)
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie in (0, 1)")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return RichResult(payload={
        "level": 2.0 + alpha * (-math.log(eps)),
        "eps": eps, "alpha": alpha,
        "method": "stick-breaking truncation level "
                  "(Ghosal & van der Vaart 2017)"})


def ghdpmedc(gx, alpha, ngrid=4000):
    r""":math:`P(\operatorname{median}(G) \le x)` for a Dirichlet process,
    where the base measure has value ``gx`` at ``x``.

    :math:`G(x)` is :math:`\mathrm{Beta}(\alpha G_0(x), \alpha(1-G_0(x)))`
    and the median of :math:`G` is at most ``x`` exactly when
    :math:`G(x) \ge 1/2`, so the answer is the beta tail above one half,
    taken here by a midpoint rule with ``ngrid`` panels on ``(1/2, 1)``.

    Parameters
    ----------
    gx : float
        Base-measure value :math:`G_0(x)`, strictly inside ``(0, 1)``.
    alpha : float
        Dirichlet-process precision, positive.
    ngrid : int
        Midpoint panels on ``(1/2, 1)``.

    Returns
    -------
    RichResult
        ``prob``, ``a``, ``b``, ``ngrid``, ``method``.
    """
    gx = float(gx)
    alpha = float(alpha)
    ngrid = int(ngrid)
    if not 0.0 < gx < 1.0:
        raise ValueError("gx must lie strictly in (0, 1)")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if ngrid < 1:
        raise ValueError("ngrid must be a positive integer")
    a = alpha * gx
    b = alpha * (1.0 - gx)
    h = 0.5 / ngrid
    lognorm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    terms = []
    for i in range(ngrid):
        u = 0.5 + (i + 0.5) * h
        if u >= 1.0:
            continue
        terms.append(math.exp(lognorm + (a - 1.0) * math.log(u)
                              + (b - 1.0) * math.log(1.0 - u)))
    return RichResult(payload={
        "prob": math.fsum(terms) * h, "a": a, "b": b, "ngrid": ngrid,
        "method": "midpoint beta-tail rule for the DP median CDF "
                  "(Ghosal & van der Vaart 2017)"})


def ghdirmom(alpha, j, jp):
    r"""Mean, variance and covariance of two cells of a Dirichlet
    distribution: with :math:`A = \sum_i \alpha_i`,
    :math:`E\pi_j = \alpha_j/A`,
    :math:`\operatorname{Var}\pi_j = \alpha_j(A-\alpha_j)/(A^2(A+1))`
    and :math:`\operatorname{Cov}(\pi_j,\pi_{j'}) =
    -\alpha_j\alpha_{j'}/(A^2(A+1))` for :math:`j \ne j'`.

    ``j`` and ``jp`` are **zero-based** here and one-based in the R arm;
    the values returned for the corresponding cells agree.  When
    ``j == jp`` the covariance is the variance, matching the R arm's
    explicit self-covariance branch.

    Parameters
    ----------
    alpha : sequence of float
        Dirichlet concentration parameters, all positive.
    j, jp : int
        Zero-based cell indices.

    Returns
    -------
    RichResult
        ``mean``, ``var``, ``cov``, ``total``, ``method``.
    """
    a = _vec(alpha, "alpha")
    if not a:
        raise ValueError("alpha must not be empty")
    if any(v <= 0 for v in a):
        raise ValueError("alpha entries must be positive")
    j = int(j)
    jp = int(jp)
    if not 0 <= j < len(a) or not 0 <= jp < len(a):
        raise ValueError(f"j and jp must lie in 0..{len(a) - 1}")
    A = math.fsum(a)
    vr = a[j] * (A - a[j]) / (A * A * (A + 1.0))
    cv = vr if j == jp else -a[j] * a[jp] / (A * A * (A + 1.0))
    return RichResult(payload={
        "mean": a[j] / A, "var": vr, "cov": cv, "total": A,
        "method": "Dirichlet first two moments "
                  "(Ghosal & van der Vaart 2017)"})


# ------------------------------------------------------ partitions
def ghewensl(mult, alpha):
    r"""Log Ewens sampling formula for the partition described by
    ``mult``, whose ``i``-th entry counts the blocks of size ``i + 1``:

    .. math:: \log\Big\{\frac{n!}{(\alpha)_n}
              \prod_i \frac{\alpha^{m_i}}{i^{m_i} m_i!}\Big\},

    with :math:`n = \sum_i i\, m_i` recovered from the multiplicities.

    Parameters
    ----------
    mult : sequence of float
        Block-size multiplicities, ``mult[i]`` counting blocks of size
        ``i + 1``.
    alpha : float
        Dirichlet-process precision, positive.

    Returns
    -------
    RichResult
        ``logprob``, ``prob``, ``n``, ``method``.
    """
    ms = _vec(mult, "mult")
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if any(v < 0 for v in ms):
        raise ValueError("multiplicities must be non-negative")
    n = int(round(math.fsum((i + 1) * m for i, m in enumerate(ms))))
    if n < 1:
        raise ValueError("the multiplicities describe an empty partition")
    lp = math.lgamma(n + 1.0) \
        - math.fsum(math.log(alpha + i) for i in range(n))
    lp += math.fsum(m * math.log(alpha) - m * math.log(i + 1)
                    - math.lgamma(m + 1.0) for i, m in enumerate(ms))
    return RichResult(payload={
        "logprob": lp, "prob": math.exp(lp), "n": n,
        "method": "log Ewens sampling formula "
                  "(Ghosal & van der Vaart 2017)"})


def ghpyeppf(sizes, d, theta):
    r"""Log exchangeable partition probability function of a Pitman-Yor
    process for blocks of the given ``sizes``:

    .. math:: \log\Big\{\frac{\prod_{j=1}^{k-1}(\theta + jd)}
              {(\theta+1)_{n-1}} \prod_j (1-d)_{n_j-1}\Big\}.

    Blocks are exchangeable, so only the multiset of ``sizes`` matters.

    Parameters
    ----------
    sizes : sequence of int
        Block sizes, each at least 1.
    d : float
        Discount parameter, typically in ``[0, 1)``.
    theta : float
        Concentration parameter, typically greater than ``-d``.

    Returns
    -------
    RichResult
        ``logprob``, ``prob``, ``n``, ``k``, ``method``.
    """
    ns = [int(v) for v in _vec(sizes, "sizes")]
    if not ns:
        raise ValueError("sizes must not be empty")
    if any(v < 1 for v in ns):
        raise ValueError("block sizes must be at least 1")
    d = float(d)
    theta = float(theta)
    n = sum(ns)
    k = len(ns)
    lp = 0.0
    for j in range(1, k):
        arg = theta + j * d
        if arg <= 0:
            raise ValueError("theta + j d must stay positive")
        lp += math.log(arg)
    for i in range(1, n):
        if theta + i <= 0:
            raise ValueError("theta + i must stay positive")
        lp -= math.log(theta + i)
    for nj in ns:
        for lvl in range(nj - 1):
            arg = 1.0 - d + lvl
            if arg <= 0:
                raise ValueError("1 - d + l must stay positive")
            lp += math.log(arg)
    return RichResult(payload={
        "logprob": lp, "prob": math.exp(lp), "n": n, "k": k,
        "method": "log Pitman-Yor EPPF "
                  "(Ghosal & van der Vaart 2017)"})


def ghibpk(n, alpha):
    r"""Expected number of features sampled by ``n`` customers of a
    one-parameter Indian buffet process, :math:`\alpha H_n`.

    Parameters
    ----------
    n : int
        Number of customers, positive.
    alpha : float
        Mass parameter, positive.

    Returns
    -------
    RichResult
        ``expected``, ``harmonic``, ``n``, ``alpha``, ``method``.
    """
    n = int(n)
    alpha = float(alpha)
    if n < 1:
        raise ValueError("n must be a positive integer")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    hn = math.fsum(1.0 / i for i in range(1, n + 1))
    return RichResult(payload={
        "expected": alpha * hn, "harmonic": hn, "n": n, "alpha": alpha,
        "method": "expected Indian-buffet features alpha H_n "
                  "(Ghosal & van der Vaart 2017)"})


# ------------------------------------------------------ Polya trees
def ghptbits(x, depth):
    r"""The leading ``depth`` bits of the binary expansion of ``x``, which
    index the nested dyadic Polya-tree partition containing ``x``.

    ``x`` is first clamped to :math:`[0, 1 - 10^{-15}]`, so a value
    outside the unit interval is silently pulled to the nearest end
    rather than rejected -- the R arm's behaviour, kept.

    Parameters
    ----------
    x : float
        Point of the unit interval.
    depth : int
        Number of Polya-tree levels, positive.

    Returns
    -------
    RichResult
        ``bits`` (list of 0/1, most significant first), ``depth``,
        ``clamped`` (whether ``x`` lay outside ``[0, 1)``), ``method``.
    """
    x = float(x)
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be a positive integer")
    return RichResult(payload={
        "bits": _bnp._bits(x, depth), "depth": depth,
        "clamped": not 0.0 <= x < 1.0,
        "method": "dyadic Polya-tree path bits "
                  "(no book citation: a plain binary expansion)"})


def ghptdens(x, data, depth=4):
    r"""Polya-tree posterior density at ``x`` relative to the uniform base
    measure.

    ``x`` is located in the dyadic partition, ``data`` are counted along
    each of its first ``depth`` bit prefixes, and the level-wise
    posterior branching probabilities
    :math:`(2a_m + 2N_m)/(2a_m + N_{m-1})` are multiplied, with the
    canonical level weights :math:`a_m = m^2` and :math:`N_0` the sample
    size.

    Parameters
    ----------
    x : float
        Point of ``[0, 1)`` at which to evaluate the density.
    data : sequence of float
        Observations in ``[0, 1)``.
    depth : int
        Number of Polya-tree levels.

    Returns
    -------
    RichResult
        ``density``, ``counts``, ``bits``, ``depth``, ``n``, ``method``.
    """
    x = float(x)
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be a positive integer")
    ds = _vec(data, "data")
    if not ds:
        raise ValueError("data must not be empty")
    n = len(ds)
    counts = _bnp.pt_path_counts(x, ds, depth)
    dens = _bnp.pt_density_posterior(x, lambda m: m * m, counts, n, depth)
    return RichResult(payload={
        "density": dens, "counts": counts,
        "bits": _bnp._bits(x, depth), "depth": depth, "n": n,
        "method": "Polya-tree posterior density, level weights a_m = m^2 "
                  "(Ghosal & van der Vaart 2017)"})


# --------------------------------------------- information quantities
def ghhell2(p, q):
    r"""``1 - sum_i sqrt(p_i q_i)`` for two discrete distributions, each
    renormalised to sum to one.

    Convention, stated because both are current: this is **half** the
    squared Hellinger distance under
    :math:`h^2(p,q) = \sum_i(\sqrt{p_i}-\sqrt{q_i})^2`, and equals the
    squared distance under the :math:`1/\sqrt2`-scaled definition.  It
    lies in ``[0, 1]``.  Double the returned value for the unscaled
    :math:`h^2`.  The R arm ``morie_gh_hellinger2`` returns exactly
    this quantity.

    Parameters
    ----------
    p, q : sequence of float
        Non-negative weights on a common support; renormalised here.

    Returns
    -------
    RichResult
        ``half_h2`` (the returned quantity), ``h2`` (twice it, the
        unscaled squared Hellinger), ``affinity``, ``method``.
    """
    ps = _probs(p, "p")
    qs = _probs(q, "q")
    if len(ps) != len(qs):
        raise ValueError(
            f"p has {len(ps)} entries and q has {len(qs)}")
    rho = math.fsum(math.sqrt(a * b) for a, b in zip(ps, qs))
    half = 1.0 - rho
    return RichResult(payload={
        "half_h2": half, "h2": 2.0 * half, "affinity": rho,
        "method": "1 - Bhattacharyya affinity = half the squared "
                  "Hellinger distance (no book citation: a generic "
                  "information-theory quantity)"})


def ghkldiv(p, q):
    r"""Kullback-Leibler divergence :math:`\sum_i p_i\log(p_i/q_i)` in nats
    over the support of ``p``, both arguments renormalised to sum to one.

    Zeros in ``q`` are floored at ``1e-300`` rather than returning
    infinity, so the result is always finite -- the R arm's behaviour.

    Parameters
    ----------
    p : sequence of float
        Non-negative weights, the reference measure.
    q : sequence of float
        Non-negative weights on the same support.

    Returns
    -------
    RichResult
        ``kl``, ``support`` (cells of ``p`` with positive mass),
        ``floored`` (cells where ``q`` was floored), ``method``.
    """
    ps = _probs(p, "p")
    qs = _probs(q, "q")
    if len(ps) != len(qs):
        raise ValueError(
            f"p has {len(ps)} entries and q has {len(qs)}")
    keep = [i for i, v in enumerate(ps) if v > 0]
    floored = sum(1 for i in keep if qs[i] < 1e-300)
    kl = math.fsum(ps[i] * math.log(ps[i] / max(qs[i], 1e-300))
                   for i in keep)
    return RichResult(payload={
        "kl": kl, "support": len(keep), "floored": floored,
        "method": "Kullback-Leibler divergence in nats, q floored at "
                  "1e-300 (no book citation: a generic "
                  "information-theory quantity)"})


def ghrenyi(p, q, alpha=0.5):
    r"""Renyi divergence of order ``alpha``,
    :math:`(\alpha-1)^{-1}\log\sum_i p_i^{\alpha} q_i^{1-\alpha}`, in
    nats, both arguments renormalised to sum to one.

    Parameters
    ----------
    p, q : sequence of float
        Non-negative weights on a common support.
    alpha : float
        Order, not equal to 1.

    Returns
    -------
    RichResult
        ``divergence``, ``alpha``, ``chernoff`` (the summed affinity),
        ``method``.
    """
    ps = _probs(p, "p")
    qs = _probs(q, "q")
    if len(ps) != len(qs):
        raise ValueError(
            f"p has {len(ps)} entries and q has {len(qs)}")
    alpha = float(alpha)
    if abs(alpha - 1.0) < 1e-15:
        raise ValueError("alpha must differ from 1; use ghkldiv for the "
                         "alpha -> 1 limit")
    rho = math.fsum(a ** alpha * b ** (1.0 - alpha)
                    for a, b in zip(ps, qs))
    if rho <= 0:
        raise ValueError("p and q have disjoint support")
    return RichResult(payload={
        "divergence": math.log(rho) / (alpha - 1.0), "alpha": alpha,
        "chernoff": rho,
        "method": "Renyi divergence of order alpha (no book citation: a "
                  "generic information-theory quantity)"})


# ----------------------------------------------------- white noise
def ghwnpost(x, n, priorvar):
    r"""Conjugate posterior of one white-noise coordinate under a
    mean-zero normal prior of variance ``priorvar``: posterior mean
    :math:`nX/(n + 1/\lambda)`, variance :math:`1/(n + 1/\lambda)`.

    The noise variance is taken as **one**.  The R arm
    ``morie_gh_wn_posterior`` has no noise-variance argument at all and
    likewise assumes unit noise; this arm matches it rather than
    generalising, so a non-unit noise level must be folded into ``n``
    by the caller.

    Parameters
    ----------
    x : float
        Observed coordinate value.
    n : float
        Effective sample size / noise-precision multiplier, positive.
    priorvar : float
        Prior variance of the coordinate, positive.

    Returns
    -------
    RichResult
        ``mean``, ``var``, ``shrinkage``, ``method``.
    """
    x = float(x)
    n = float(n)
    lam = float(priorvar)
    if n <= 0:
        raise ValueError("n must be positive")
    if lam <= 0:
        raise ValueError("priorvar must be positive")
    prec = n + 1.0 / lam
    return RichResult(payload={
        "mean": n * x / prec, "var": 1.0 / prec, "shrinkage": n / prec,
        "method": "conjugate white-noise coordinate posterior, unit noise "
                  "variance assumed (Ghosal & van der Vaart 2017)"})


def cheatsheet():
    return ("bnpgh: Ghosal-van der Vaart scalar formulas -- DP posterior, "
            "Ewens/PY partitions, Polya trees, CRM Laplace transforms, "
            "Hellinger/KL/Renyi")
