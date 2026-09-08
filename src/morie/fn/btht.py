# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Bootstrap two-sided p-value for H0: theta = theta0.

Source: Davison, A. C. and Hinkley, D. V. (1997), *Bootstrap Methods and
their Application*, Cambridge University Press, Chapter 4.  The
resampling must be done under the null, not under the fitted model: the
data are shifted so their statistic equals theta0,

    x0_i = x_i - t + theta0,

resampled with replacement, and the two-sided p-value is

    p = 2 min( (1 + #{T*_b >= T_hat})/(B + 1) ,
               (1 + #{T*_b <= T_hat})/(B + 1) ),

capped at one.  The +1 in numerator and denominator is Davison and
Hinkley's convention (Section 4.2): the observed statistic is itself one
of the B + 1 values under the null, so a p-value of exactly zero -- which
the raw proportion would report the moment no replicate reaches T_hat --
is not attainable, and the smallest reportable value is 2/(B + 1).  A
p-value of zero from a finite simulation is a statement the simulation
cannot support.

Shifting is the step that is easy to skip and fatal to skip: resampling
the unshifted data estimates the distribution of T around t, so the test
would compare T_hat against its own centre and return a p-value near one
for every data set no matter how far t is from theta0.  The degenerate
anchor checks exactly that -- with theta0 set to t the p-value must come
out near one, and with theta0 far away it must fall to the resolution
floor 2/B.

Resampling is deterministic, from the Park-Miller generator
s <- 16807 s mod (2^31 - 1), so both language arms see the same
resamples.  The shift is additive, so it is exact only for
location-equivariant statistics; ``method`` says so.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_test_hypothesis"]

_M = 2147483647


def boot_test_hypothesis(x, theta0=0.0, stat=None, B=999, seed=1):
    """Two-sided bootstrap p-value.

    Parameters
    ----------
    x : array-like
        The sample.
    theta0 : float
        The null value.
    stat : callable, optional
        The statistic; defaults to the sample mean.
    B : int
        Number of resamples.
    seed : int
        Seed for the deterministic index generator.

    Returns
    -------
    p : the two-sided p-value
    T_hat : the statistic on the original data
    T_b : the B null replicates
    """
    v = core.vec(x)
    n = len(v)
    if n == 0:
        raise ValueError("boot_test_hypothesis: x is empty")
    Bn = int(B)
    if Bn < 1:
        raise ValueError("boot_test_hypothesis: B must be at least one")
    f = (lambda z: core.mean(z)) if stat is None else stat
    t = float(f(v))
    th0 = float(theta0)
    shifted = [u - t + th0 for u in v]
    s = int(seed) % _M
    if s <= 0:
        s += _M - 1
    reps = []
    ge = 0
    le = 0
    for _ in range(Bn):
        samp = []
        for _ in range(n):
            s = (16807 * s) % _M
            u = (s - 1.0) / (_M - 1.0)
            j = int(u * n)
            if j >= n:
                j = n - 1
            samp.append(shifted[j])
        tb = float(f(samp))
        reps.append(tb)
        if tb >= t:
            ge += 1
        if tb <= t:
            le += 1
    pge = (1.0 + ge) / (Bn + 1.0)
    ple = (1.0 + le) / (Bn + 1.0)
    p = 2.0 * (pge if pge < ple else ple)
    if p > 1.0:
        p = 1.0
    return RichResult(
        title="Bootstrap hypothesis test",
        summary_lines=[("p", p), ("T_hat", t)],
        payload={
            "p": p,
            "estimate": p,
            "T_hat": t,
            "theta0": th0,
            "T_b": reps,
            "p_upper": pge,
            "p_lower": ple,
            "B": Bn,
            "n": n,
            "method": "null resampling of the shifted data, p = 2 min((1+#)/(B+1)); D&H Sect. 4.2",
        },
    )


def cheatsheet():
    return "btht: Bootstrap two-sided p-value for H0: theta = theta0"


# compact alias per ledger/NAMING.md
boottesthypothesis = boot_test_hypothesis
