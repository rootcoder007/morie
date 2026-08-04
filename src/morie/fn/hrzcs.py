# morie.fn -- function file (rootcoder007/morie)
"""The curse of dimensionality in nonparametric estimation.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Appendix A.1 (page 238) and A.2 (page 241).

With p (or g) s times continuously differentiable and K an order s
kernel, the fastest possible rate of convergence in probability of a
d-dimensional kernel density or mean-regression estimator is

    n^{-s/(2s+d)}

achieved at bandwidth h_n = c n^{-1/(2s+d)}.  Squaring gives the
mean-square-error rate n^{-2s/(2s+d)}, which is the familiar
n^{-4/(4+d)} when s = 2.  The rate degrades as d grows, so the sample
size needed for fixed precision grows rapidly -- the curse of
dimensionality (Huber 1985).  The book cites Silverman's (1986)
illustration in which n must rise by a factor of nearly 200 to hold
precision constant as d goes from 1 to 5.

Everything here is the closed-form rate arithmetic.  No estimation, no
randomness.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["nprate", "horowitz_curse_dimensionality"]


def nprate(d, n, s=2, c=1.0, dref=1):
    """Convergence rate, optimal bandwidth and the sample-size penalty.

    Parameters
    ----------
    d : int
        Dimension of the covariate.
    n : int
        Sample size.
    s : int, default 2
        Order of the kernel; also the assumed number of continuous
        derivatives.  s = 2 is the ordinary second-order kernel.
    c : float, default 1.0
        Bandwidth constant in h_n = c n^{-1/(2s+d)}.
    dref : int, default 1
        Reference dimension for the sample-size penalty.

    Returns
    -------
    RichResult
        payload keys: exponent, rate, mseexponent, mse, bandwidth,
        nequiv, penalty, d, s, n, method.
    """
    d = int(d)
    n = int(n)
    s = int(s)
    if d < 1 or n < 1 or s < 1:
        raise ValueError("d, n and s must all be positive integers.")
    expo = s / float(2 * s + d)
    rate = float(n ** (-expo))
    bw = float(c) * float(n ** (-1.0 / (2 * s + d)))
    expref = s / float(2 * s + int(dref))
    # sample size in dimension d giving the same rate as n in dref
    nequiv = float(n ** (expref / expo))
    return RichResult(
        title="Nonparametric convergence rate and the curse of dimensionality",
        payload={"exponent": expo, "rate": rate,
                 "mseexponent": 2.0 * expo, "mse": float(rate * rate),
                 "bandwidth": bw, "nequiv": nequiv,
                 "penalty": float(nequiv / n), "d": d, "s": s, "n": n,
                 "method": "Horowitz (2009) Appendix A.1/A.2 optimal rates"},
    )


horowitz_curse_dimensionality = nprate


def cheatsheet():
    return "hrzcs: nonparametric rate n^{-s/(2s+d)} and the curse of dimensionality"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    r = nprate(1, 1000)
    assert abs(r["exponent"] - 2.0 / 5.0) < 1e-12, r["exponent"]
    assert abs(r["mseexponent"] - 4.0 / 5.0) < 1e-12
    r5 = nprate(5, 1000)
    assert abs(r5["mseexponent"] - 4.0 / 9.0) < 1e-12, r5["mseexponent"]
    assert r5["rate"] > r["rate"]          # slower convergence in d = 5
    assert r5["penalty"] > 1.0
    print("ok", r["rate"], r5["rate"], r5["penalty"])
