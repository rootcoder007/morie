# morie.fn -- function file (rootcoder007/morie)
"""k-ary randomised response under local differential privacy.

Each respondent perturbs their value before it is collected, so the
collector never sees the truth.
"""

from . import _robust_core as _rc
from ._richresult import RichResult, with_describe_pointer

__all__ = ["local_dp_planar_mechanism"]


def local_dp_planar_mechanism(y, truth=None, k=2, epsilon=1.0,
                             seed=2):
    """P(report=v|true=u) = e^eps/(k-1+e^eps) if v=u, else
    1/(k-1+e^eps).

    The ratio of any two conditional probabilities is at most e^eps,
    which is exactly the eps-local-differential-privacy guarantee.
    Reporting raw counts would be biased towards uniform, so the
    result also carries the debiased estimate that inverts the
    transition matrix. Keys: estimate."""
    values = truth if truth is not None else y
    r = _rc.local_dp_randomised_response(values, k, epsilon, seed=seed)
    res = RichResult(payload={"estimate": r["estimate"],
                              "reports": r["reports"],
                              "observed": r["observed"],
                              "p_keep": r["p_keep"],
                              "p_flip": r["p_flip"],
                              "epsilon": r["epsilon"], "k": r["k"],
                              "method": r["method"]})
    return with_describe_pointer(res, "ldppm")


def cheatsheet():
    return "ldppm: k-ary randomised response under local differential privacy"
