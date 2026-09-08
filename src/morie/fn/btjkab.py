# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Jackknife-after-bootstrap influence diagnostic.

Source: Efron, B. (1992), "Jackknife-after-bootstrap standard errors and
influence functions", *Journal of the Royal Statistical Society Series B*
54(1), 83-111, and Davison and Hinkley (1997), *Bootstrap Methods and
their Application*, Section 3.10, which is the treatment consulted here.

The trick is that the resamples already drawn contain, for free, the
resamples of every leave-one-out data set: those replicates whose index
set never mentions observation i are exactly a bootstrap sample of
x with i removed.  So

    theta_bar_{-i} = mean{ t*_b : i not in the b-th index set },
    infl_i         = theta_bar_{-i} - mean{ t*_b },

no refitting required.  A large |infl_i| says observation i moves the
whole bootstrap distribution.

The counting matters: an observation that appears in every resample has
no leave-i-out subset at all, and the honest answer there is a missing
value, not zero.  ``n_out`` reports the subset sizes so a diagnostic
computed off two replicates is not mistaken for a stable one.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_jackknife_after_boot"]


def boot_jackknife_after_boot(x, theta_b, B_idx):
    """Influence of each observation on the bootstrap distribution.

    Parameters
    ----------
    x : array-like
        The original sample of length n.
    theta_b : array-like
        The B replicates.
    B_idx : sequence of sequences
        B index sets, 0-based, one per replicate.

    Returns
    -------
    infl_i : the n influence values (NaN where every resample used i)
    theta_minus : the n leave-i-out replicate means
    n_out : how many replicates omitted each observation
    """
    xx = core.vec(x)
    n = len(xx)
    if n == 0:
        raise ValueError("boot_jackknife_after_boot: x is empty")
    tb = core.vec(theta_b)
    B = len(tb)
    if B == 0:
        raise ValueError("boot_jackknife_after_boot: no bootstrap replicates")
    idx = [[int(i) for i in r] for r in B_idx]
    if len(idx) != B:
        raise ValueError("boot_jackknife_after_boot: B_idx and theta_b have different lengths")
    used = []
    for r in idx:
        seen = [False] * n
        for i in r:
            if i < 0 or i >= n:
                raise ValueError("boot_jackknife_after_boot: an index is out of range")
            seen[i] = True
        used.append(seen)
    grand = core.mean(tb)
    infl = []
    tm = []
    nout = []
    nan = float("nan")
    for i in range(n):
        s = 0.0
        c = 0
        for b in range(B):
            if not used[b][i]:
                s += tb[b]
                c += 1
        nout.append(c)
        if c == 0:
            tm.append(nan)
            infl.append(nan)
        else:
            m = s / c
            tm.append(m)
            infl.append(m - grand)
    mx = 0.0
    for v in infl:
        if v == v and abs(v) > mx:
            mx = abs(v)
    return RichResult(
        title="Jackknife after bootstrap",
        summary_lines=[("n", n), ("B", B)],
        payload={
            "infl_i": infl,
            "estimate": mx,
            "theta_minus": tm,
            "n_out": nout,
            "grand_mean": grand,
            "max_abs_influence": mx,
            "n": n,
            "B": B,
            "method": "Efron (1992) jackknife-after-bootstrap; Davison and Hinkley Sect. 3.10",
        },
    )


def cheatsheet():
    return "btjkab: Jackknife-after-bootstrap influence diagnostic"


# compact alias per ledger/NAMING.md
bootjackknifeafterboot = boot_jackknife_after_boot
