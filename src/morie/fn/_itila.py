# morie.fn -- shelf core (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Information-theory / Bayesian-inference shelf core.

Spec: MacKay, D.J.C. (2003), *Information Theory, Inference, and
Learning Algorithms*, Cambridge University Press.  Every routine here
carries the printed chapter/equation/page it implements.

Everything is closed form or a fixed-iteration recurrence -- no RNG,
no tolerance-driven early exit -- so the R mirror in R/info_mackay.R
reproduces each number bit-for-bit.
"""

from __future__ import annotations

import math

from . import _array_core as np

_LN2 = math.log(2.0)


def _log2(x: float) -> float:
    return math.log(float(x)) / _LN2


# --- ch. 1: the repetition codes R3 / RN (pp. 5-20) -------------------

def r3post(r, f):
    """(1.18) p.9 -- posterior over the source bit of an R3 codeword."""
    r = [int(b) for b in r]
    if len(r) != 3 or any(b not in (0, 1) for b in r):
        raise ValueError("r must be three bits, got %r" % (r,))
    f = float(f)
    if not 0.0 < f < 1.0:
        raise ValueError("f must lie strictly in (0, 1), got %r" % (f,))
    lik = []
    for s in (0, 1):
        p = 1.0
        for b in r:
            p *= f if b != s else (1.0 - f)
        lik.append(p)
    tot = lik[0] + lik[1]
    return {
        "p0": lik[0] / tot,
        "p1": lik[1] / tot,
        "decoded": 1 if lik[1] > lik[0] else 0,
        "gamma": (1.0 - f) / f,
        "evidence": tot / 2.0,
    }


def cbcapx(n):
    """(1.40) p.17 -- Gaussian approximation to the central binomial."""
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1, got %r" % (n,))
    approx = 2.0 ** n / math.sqrt(2.0 * math.pi * n / 4.0)
    exact = float(math.comb(n, n // 2))
    return {
        "approx": approx,
        "exact": exact,
        "relerr": approx / exact - 1.0,
        "logapprox": n * _LN2 - 0.5 * math.log(2.0 * math.pi * n / 4.0),
    }


def binsumga(n):
    """(1.41) p.17 -- the Gaussian sum that proves (1.40).

    MacKay writes ``sigma = sqrt(N/4)`` in the running text but then
    uses ``sqrt(2 pi sigma)``; only the reading ``sigma = N/4`` (the
    VARIANCE) makes (1.41) agree with (1.40).  This routine takes the
    variance reading and reports both quantities so the discrepancy is
    visible rather than silently resolved.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1, got %r" % (n,))
    var = n / 4.0
    gsum = math.sqrt(2.0 * math.pi * var)
    exact = float(math.comb(n, n // 2))
    return {
        "var": var,
        "sd": math.sqrt(var),
        "gsum": gsum,
        "total": 2.0 ** (-n) * exact * gsum,
        "cbcapprox": 2.0 ** n / gsum,
    }


def repcpb(n, f):
    """(1.42)-(1.43) p.17 -- block error probability of RN, odd N."""
    n = int(n)
    f = float(f)
    if n < 1 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer, got %r" % (n,))
    if not 0.0 < f < 1.0:
        raise ValueError("f must lie strictly in (0, 1), got %r" % (f,))
    k = (n + 1) // 2
    leading = math.comb(n, k) * f ** k * (1.0 - f) ** (n - k)
    half = (n - 1) / 2.0
    approx1 = (2.0 ** n / math.sqrt(math.pi * n / 2.0)) * f * (f * (1.0 - f)) ** half
    approx2 = (1.0 / math.sqrt(math.pi * n / 8.0)) * f * (4.0 * f * (1.0 - f)) ** half
    return {
        "leading": leading,
        "approx1": approx1,
        "approx2": approx2,
        "logapprox2": _log2(approx2),
    }


def repcn(pb, f, n0=68.0, iters=3):
    """(1.44)-(1.45) p.17 -- blocklength N reaching a target pb.

    Fixed ``iters`` sweeps of the iteration printed under (1.44),
    started from the book value N-hat_1 = 68; there is no convergence
    test, so both language arms take identical steps.
    """
    pb = float(pb)
    f = float(f)
    if not 0.0 < pb < 1.0:
        raise ValueError("pb must lie strictly in (0, 1), got %r" % (pb,))
    if not 0.0 < f < 0.5:
        raise ValueError("f must lie strictly in (0, 0.5), got %r" % (f,))
    n = float(n0)
    denom = math.log10(4.0 * f * (1.0 - f))
    half = float("nan")
    for _ in range(int(iters)):
        half = (math.log10(pb) + math.log10(math.sqrt(math.pi * n / 8.0) / f)) / denom
        n = 2.0 * half + 1.0
    return {"n": n, "half": half, "denom": denom, "iters": int(iters)}


# --- ch. 2: inferring which urn (pp. 27-31) ---------------------------

def urnpost(nb, ntot, nurns=10):
    """(2.25)-(2.26) p.28 -- posterior over the urn index u."""
    nb = int(nb)
    ntot = int(ntot)
    nurns = int(nurns)
    if nurns < 1 or ntot < 0 or not 0 <= nb <= ntot:
        raise ValueError("need 0 <= nb <= ntot and nurns >= 1")
    prior = 1.0 / (nurns + 1.0)
    joint = []
    for u in range(nurns + 1):
        fu = u / float(nurns)
        joint.append(prior * math.comb(ntot, nb) * fu ** nb * (1.0 - fu) ** (ntot - nb))
    evid = sum(joint)
    post = [j / evid for j in joint]
    return {
        "posterior": post,
        "evidence": evid,
        "map": max(range(nurns + 1), key=lambda u: post[u]),
        "prior": prior,
    }


def urnpred(nb, ntot, nurns=10):
    """(2.29)-(2.31) p.29 -- predictive P(next ball black)."""
    post = urnpost(nb, ntot, nurns)["posterior"]
    nurns = int(nurns)
    p = sum((u / float(nurns)) * post[u] for u in range(nurns + 1))
    imap = max(range(nurns + 1), key=lambda u: post[u])
    return {"p": p, "pnot": 1.0 - p, "pmap": imap / float(nurns)}


# --- ch. 3: the bent coin and model comparison (pp. 50-53, 63) --------

def bcoinlik(pa, fa, fb):
    """(3.8) p.51 -- bent-coin likelihood P(s | p_a, F, H1)."""
    pa = float(pa)
    fa = int(fa)
    fb = int(fb)
    if not 0.0 <= pa <= 1.0 or fa < 0 or fb < 0:
        raise ValueError("need pa in [0, 1] and non-negative counts")
    lik = pa ** fa * (1.0 - pa) ** fb
    return {
        "likelihood": lik,
        "loglik": (fa * math.log(pa) if fa else 0.0)
        + (fb * math.log1p(-pa) if fb else 0.0),
        "fa": fa,
        "fb": fb,
    }


def bcoinpri(pa):
    """(3.9) p.51 -- the uniform prior density P(p_a | H1) = 1."""
    pa = float(pa)
    inside = 0.0 <= pa <= 1.0
    return {
        "density": 1.0 if inside else 0.0,
        "inside": bool(inside),
        "logdensity": 0.0 if inside else float("-inf"),
    }


def sucrule(fa, fb):
    """(3.16) p.52 -- the rule of succession, (Fa + 1)/(Fa + Fb + 2)."""
    fa = int(fa)
    fb = int(fb)
    if fa < 0 or fb < 0:
        raise ValueError("counts must be non-negative")
    p = (fa + 1.0) / (fa + fb + 2.0)
    return {
        "p": p,
        "pnot": 1.0 - p,
        "mle": (fa / float(fa + fb)) if (fa + fb) else float("nan"),
    }


def evidmix(evidences, priors):
    """(3.19) p.53 -- total evidence P(s | F) by the sum rule."""
    ev = [float(e) for e in evidences]
    pr = [float(p) for p in priors]
    if len(ev) != len(pr) or not ev:
        raise ValueError("evidences and priors must be non-empty and equal length")
    terms = [e * p for e, p in zip(ev, pr)]
    tot = sum(terms)
    return {"evidence": tot, "terms": terms, "posterior": [t / tot for t in terms]}


def postodds(lik1, lik0, prior1=0.5, prior0=0.5):
    """(3.21) p.53 -- posterior odds = likelihood ratio x prior odds."""
    lik1 = float(lik1)
    lik0 = float(lik0)
    odds = (lik1 * float(prior1)) / (lik0 * float(prior0))
    return {
        "odds": odds,
        "logodds": math.log(odds),
        "p1": odds / (1.0 + odds),
        "bayesfactor": lik1 / lik0,
    }


def bcoinbf(fa, fb, p0=1.0 / 6.0):
    """(3.12), (3.20), (3.22) pp.52-53 -- H1 (free p_a) vs H0 (p_a = p0)."""
    fa = int(fa)
    fb = int(fb)
    if fa < 0 or fb < 0:
        raise ValueError("counts must be non-negative")
    p0 = float(p0)
    lge1 = math.lgamma(fa + 1.0) + math.lgamma(fb + 1.0) - math.lgamma(fa + fb + 2.0)
    lge0 = (fa * math.log(p0) if fa else 0.0) + (fb * math.log1p(-p0) if fb else 0.0)
    return {
        "evidence1": math.exp(lge1),
        "evidence0": math.exp(lge0),
        "ratio": math.exp(lge1 - lge0),
        "logratio": lge1 - lge0,
    }


def lrprod(num, den):
    """(3.31) p.63 -- posterior odds as a running product of ratios."""
    num = [float(a) for a in num]
    den = [float(b) for b in den]
    if len(num) != len(den) or not num:
        raise ValueError("num and den must be non-empty and equal length")
    ratio = 1.0
    for a, b in zip(num, den):
        ratio *= a / b
    return {
        "ratio": ratio,
        "p1": ratio / (1.0 + ratio),
        "logratio": math.log(ratio),
        "n": len(num),
    }


# --- ch. 4: the typical set (p. 80) -----------------------------------

def typset(p, n, h, beta):
    """(4.29) p.80 -- membership test for the typical set T_{N beta}."""
    p = float(p)
    n = int(n)
    if not 0.0 < p <= 1.0 or n < 1:
        raise ValueError("need 0 < p <= 1 and n >= 1")
    info = -_log2(p)
    rate = info / n
    dev = rate - float(h)
    return {
        "info": info,
        "rate": rate,
        "deviation": dev,
        "member": bool(abs(dev) < float(beta)),
    }


# --- ch. 11: the Gaussian channel (p. 182) ----------------------------

def gchpost(y, v, s2):
    """(11.27)-(11.29) p.182 -- posterior over a Gaussian channel input."""
    y = float(y)
    v = float(v)
    s2 = float(s2)
    if v <= 0.0 or s2 <= 0.0:
        raise ValueError("v and s2 must be positive")
    var = 1.0 / (1.0 / v + 1.0 / s2)
    mean = v / (v + s2) * y
    return {
        "mean": mean,
        "var": var,
        "sd": math.sqrt(var),
        "wdata": (1.0 / s2) / (1.0 / v + 1.0 / s2),
        "marginalvar": v + s2,
    }


# --- ch. 19: why have sex? (pp. 271-273) ------------------------------

_ETA = math.sqrt(2.0 / (math.pi + 2.0))


def sexbeta(gamma):
    """(19.7) p.271 -- dynamic-equilibrium variance factor 1/(1 - gamma)."""
    gamma = float(gamma)
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1), got %r" % (gamma,))
    onep = 1.0 / (1.0 - gamma)
    return {"onepbeta": onep, "beta": onep - 1.0, "gamma": gamma}


def sexdfdt(f, g, eta=None):
    """(19.13) p.273 -- dF/dt = eta sqrt(f (1 - f) G) under sexual mixing."""
    f = float(f)
    g = float(g)
    if not 0.0 <= f <= 1.0 or g <= 0.0:
        raise ValueError("need f in [0, 1] and G > 0")
    eta = _ETA if eta is None else float(eta)
    return {"dfbardt": eta * math.sqrt(f * (1.0 - f) * g), "eta": eta, "g": g}


def sexfsol(t, g, f0, eta=None, c=None):
    """(19.14) p.273 -- f(t) = (1 + sin(eta (t + c)/sqrt(G)))/2.

    The book states ``c = asin(2 f(0) - 1)``, which does NOT satisfy its
    own f(0) = f_0 unless the sine argument is read as
    ``eta t/sqrt(G) + c``.  The default ``c`` here is the
    self-consistent one, ``sqrt(G)/eta * asin(2 f_0 - 1)``; the printed
    value is returned as ``cbook`` so the disagreement stays visible.
    """
    t = float(t)
    g = float(g)
    f0 = float(f0)
    if g <= 0.0 or not 0.0 <= f0 <= 1.0:
        raise ValueError("need G > 0 and f0 in [0, 1]")
    eta = _ETA if eta is None else float(eta)
    cbook = math.asin(2.0 * f0 - 1.0)
    c = (math.sqrt(g) / eta) * cbook if c is None else float(c)
    return {
        "f": 0.5 * (1.0 + math.sin(eta * (t + c) / math.sqrt(g))),
        "c": c,
        "cbook": cbook,
        "tperfect": (math.pi / eta) * math.sqrt(g),
    }


# --- ch. 24: exact marginalization in Gaussians (pp. 319-320) ---------

def gllsuff(xbar, s, n, mu, sigma):
    """(24.5)-(24.6) p.319 -- Gaussian log likelihood via (xbar, S)."""
    n = int(n)
    sigma = float(sigma)
    if n < 1 or sigma <= 0.0:
        raise ValueError("need n >= 1 and sigma > 0")
    s = float(s)
    ll = -n * math.log(math.sqrt(2.0 * math.pi) * sigma) - (
        n * (float(mu) - float(xbar)) ** 2 + s
    ) / (2.0 * sigma ** 2)
    return {"loglik": ll, "n": n, "s": s, "sigman": math.sqrt(s / n)}


def mupostsg(xbar, n, sigma):
    """(24.9)-(24.11) p.320 -- P(mu | data, sigma) = Normal(xbar, sigma^2/n)."""
    n = int(n)
    sigma = float(sigma)
    if n < 1 or sigma <= 0.0:
        raise ValueError("need n >= 1 and sigma > 0")
    var = sigma ** 2 / n
    return {"mean": float(xbar), "var": var, "se": math.sqrt(var), "n": n}


def sigevid(s, n, sigma, sigmamu=1.0):
    """(24.13) p.320 -- log evidence for sigma, mu marginalized out."""
    n = int(n)
    sigma = float(sigma)
    sigmamu = float(sigmamu)
    if n < 1 or sigma <= 0.0 or sigmamu <= 0.0:
        raise ValueError("need n >= 1, sigma > 0, sigmamu > 0")
    s = float(s)
    bestfit = -n * math.log(math.sqrt(2.0 * math.pi) * sigma) - s / (2.0 * sigma ** 2)
    occam = math.log(math.sqrt(2.0 * math.pi) * sigma / math.sqrt(n) / sigmamu)
    return {"logevidence": bestfit + occam, "bestfit": bestfit, "logoccam": occam}


# --- ch. 28: model comparison and the razor (pp. 344-352) -------------

def postgapx(dw, a):
    """(28.5) p.344 -- quadratic (Gaussian) approximation to the posterior."""
    dw = np.atleast_1d(np.asarray(dw, dtype=float))
    a = np.atleast_2d(np.asarray(a, dtype=float))
    k = int(np.shape(dw)[0])
    sh = np.shape(a)
    if sh[0] != k or sh[1] != k:
        raise ValueError("A must be %d x %d to match dw" % (k, k))
    quad = float(np.dot(dw, np.dot(a, dw)))
    cov = np._pinv(a)
    sd = [math.sqrt(abs(float(cov[i][i]))) for i in range(k)]
    return {
        "quadform": quad,
        "logratio": -0.5 * quad,
        "ratio": math.exp(-0.5 * quad),
        "errorbars": sd,
    }


def evratio(factors):
    """(28.13)-(28.14) p.351 -- posterior ratio as a product of penalties."""
    factors = [float(x) for x in factors]
    if not factors or any(x <= 0.0 for x in factors):
        raise ValueError("factors must be positive and non-empty")
    prod = 1.0
    for x in factors:
        prod *= x
    return {
        "ratio": 1.0 / prod,
        "product": prod,
        "logratio": -math.log(prod),
        "n": len(factors),
    }


def msglen(p=None, length=None):
    """(28.15) p.352 -- P(x) = 2^-L(x), L(x) = -log2 P(x)."""
    if (p is None) == (length is None):
        raise ValueError("give exactly one of p or length")
    if p is not None:
        p = float(p)
        if not 0.0 < p <= 1.0:
            raise ValueError("p must lie in (0, 1]")
        length = -_log2(p)
    else:
        length = float(length)
        p = 2.0 ** (-length)
    return {"length": length, "p": p, "nats": length * _LN2}


def mdlpost(ph, pdh, deltad=1.0):
    """(28.16)-(28.17) p.352 -- two-part MDL message length, in bits."""
    ph = float(ph)
    pdh = float(pdh)
    deltad = float(deltad)
    if not 0.0 < ph <= 1.0 or pdh <= 0.0 or deltad <= 0.0:
        raise ValueError("need 0 < ph <= 1 and pdh, deltad > 0")
    model = -_log2(ph)
    data = -_log2(pdh * deltad)
    return {"total": model + data, "model": model, "data": data}


def linevid(x, t, sigma=1.0, slope=True, priorsd=1.0):
    """(28.22) p.352, Exercise 28.2 -- evidence for a straight-line model.

    ``slope=False`` is H1 (horizontal line, w1 = 0); ``slope=True`` is
    H2 (w1 free with a Normal(0, priorsd^2) prior).  Closed form:
    t ~ Normal(0, priorsd^2 X X' + sigma^2 I).
    """
    xs = [float(v) for v in np.atleast_1d(np.asarray(x, dtype=float))]
    ts = [float(v) for v in np.atleast_1d(np.asarray(t, dtype=float))]
    n = len(xs)
    if len(ts) != n or n < 1:
        raise ValueError("x and t must be non-empty and the same length")
    sigma = float(sigma)
    priorsd = float(priorsd)
    if sigma <= 0.0 or priorsd <= 0.0:
        raise ValueError("sigma and priorsd must be positive")
    cols = [[1.0] * n, xs] if slope else [[1.0] * n]
    pv = priorsd ** 2
    cov = [
        [
            pv * sum(c[i] * c[j] for c in cols) + (sigma ** 2 if i == j else 0.0)
            for j in range(n)
        ]
        for i in range(n)
    ]
    cova = np.asarray(cov, dtype=float)
    _sign, logdet = np._lu_slogdet(cova)
    sol = np.dot(np._pinv(cova), np.asarray(ts, dtype=float))
    quad = float(np.dot(np.asarray(ts, dtype=float), sol))
    lev = -0.5 * (n * math.log(2.0 * math.pi) + float(logdet) + quad)
    return {
        "logevidence": lev,
        "evidence": math.exp(lev),
        "quadform": quad,
        "logdet": float(logdet),
        "k": len(cols),
    }


# --- ch. 29: why uniform sampling fails (p. 366) ----------------------

def rminsamp(n, h):
    """(29.19) p.366 -- uniform draws needed to hit the typical set once."""
    n = float(n)
    h = float(h)
    if n <= 0.0 or h < 0.0 or h > n:
        raise ValueError("need 0 < n and 0 <= h <= n")
    log2r = n - h
    return {
        "log2rmin": log2r,
        "log10rmin": log2r * _LN2 / math.log(10.0),
        "rmin": 2.0 ** log2r if log2r < 1000.0 else float("inf"),
    }
