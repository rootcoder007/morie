# morie.fn -- function file (rootcoder007/morie)
"""Split-merge MH move for DPM.

Implements sec. 5.2 (MH on configurations; prior (4.20)) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_splitmerge"]


def _norm_pdf(x, mu, sd):
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def _config_logscore(xs, z, M, sigma):
    """log of prior (4.20) x likelihood with each cluster scored at
    its own mean."""
    labels = sorted(set(z))
    lp = len(labels) * math.log(M)
    for lab in labels:
        members = [xs[j] for j in range(len(xs)) if z[j] == lab]
        lp += math.lgamma(len(members))
        mu = sum(members) / len(members)
        lp += sum(math.log(max(_norm_pdf(v, mu, sigma), 1e-300))
                  for v in members)
    return lp


def ghosal_splitmerge(data, z_current, split_label=None, alpha=1.0,
                      sigma=0.5):
    """MH split/merge on the configuration: propose splitting one
    cluster at its mean (or merging two), accept with
    min(1, score(new)/score(old)) where the score is the Dirichlet
    partition prior (4.20) times the likelihood (GvdV 2017 sec. 5.2).
    Keys: estimate."""
    xs = _bnp._flat(data)
    z = [int(v) for v in _bnp._flat(z_current)]
    M = float(alpha)
    if split_label is None:
        split_label = max(set(z), key=lambda l: z.count(l))
    members = [j for j in range(len(xs)) if z[j] == int(split_label)]
    if len(members) < 2:
        raise ValueError("cluster too small to split")
    mu = sum(xs[j] for j in members) / len(members)
    new_lab = max(z) + 1
    z_prop = list(z)
    for j in members:
        if xs[j] > mu:
            z_prop[j] = new_lab
    if len(set(z_prop)) == len(set(z)):     # degenerate split
        z_prop = z
    old = _config_logscore(xs, z, M, sigma)
    new = _config_logscore(xs, z_prop, M, sigma)
    accept = min(1.0, math.exp(min(new - old, 50.0)))
    res = RichResult(payload={"estimate": accept,
                              "log_score_old": old,
                              "log_score_new": new,
                              "z_proposed": z_prop,
                              "method": "split-merge MH ratio (GvdV 2017 sec. 5.2, prior 4.20)"})
    return with_describe_pointer(res, "gh_c5_4")


def cheatsheet():
    return "gh_c5_4: Split-merge MH move for DPM"
