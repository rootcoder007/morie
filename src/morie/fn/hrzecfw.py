# morie.fn -- function file (rootcoder007/morie)
"""Empirical characteristic function of W, the deconvolution sample analogue.

Horowitz, J. L. (2009), Semiparametric and Nonparametric Methods in
Econometrics, Springer, Section 5.1, page 137 (volume [Pages 135-188],
read as a rendered page image).  In the measurement-error model
W = U + epsilon of equation (5.6), the inversion formula (5.7) is

    f_U(u) = (1 / 2 pi) integral exp(-i tau u) psi_W(tau) / psi_eps(tau) d tau,

and the only unknown on the right is psi_W, replaced by the empirical
characteristic function printed unnumbered immediately below (5.7):

    psi_nW(tau) = (1 / n) sum_{j=1}^n exp(i tau W_j).

The book uses j to index observations because i is the imaginary unit;
the same convention is kept here.  Only the empirical characteristic
function itself is computed: the book notes on the same page that
substituting psi_nW directly into (5.7) generally gives a divergent
integral, which is why the smoothed estimators of Section 5.1.1 exist.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["horowitz_empirical_cf"]


def horowitz_empirical_cf(w, tau):
    """Empirical characteristic function (1/n) sum exp(i tau W_j).

    Parameters
    ----------
    w : array-like
        The sample {W_j}.
    tau : array-like or float
        Frequencies at which to evaluate psi_nW.

    Returns
    -------
    characteristic_function : list of (real, imaginary) pairs
    re, im, modulus, argument : the same thing split out
    """
    ww = core.vec(w)
    tt = core.vec(tau)
    n = len(ww)
    if n == 0:
        raise ValueError("horowitz_empirical_cf: w is empty")
    if len(tt) == 0:
        raise ValueError("horowitz_empirical_cf: tau is empty")
    re = []
    im = []
    mod = []
    arg = []
    cf = []
    for t in tt:
        a = 0.0
        b = 0.0
        for j in range(n):
            a += math.cos(t * ww[j])
            b += math.sin(t * ww[j])
        a /= n
        b /= n
        re.append(a)
        im.append(b)
        mod.append(math.sqrt(a * a + b * b))
        arg.append(math.atan2(b, a))
        cf.append([a, b])
    return RichResult(
        title="Empirical characteristic function for deconvolution",
        summary_lines=[("n", n), ("tau", len(tt))],
        payload={
            "estimate": mod[0],
            "characteristic_function": cf,
            "re": re,
            "im": im,
            "modulus": mod,
            "argument": arg,
            "tau": tt,
            "n": n,
            "method": "Horowitz (2009) Section 5.1 p.137, psi_nW(tau) = n^-1 sum exp(i tau W_j)",
        },
    )


def cheatsheet():
    return "hrzecfw: Empirical characteristic function for deconvolution"
