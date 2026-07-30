# morie.fn -- function file (rootcoder007/morie)
"""Exponential mechanism -- McSherry & Talwar (2007)."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["dp_exponential_mechanism"]


def dp_exponential_mechanism(candidates, utility, epsilon=1.0, sensitivity=1.0,
                             seed=None):
    r"""Privately select from a discrete candidate set.

    Samples candidate :math:`r` with probability proportional to

    .. math::
        \exp\!\left(\frac{\varepsilon\, u(D, r)}{2\,\Delta u}\right),

    where :math:`\Delta u` is the sensitivity of the utility function. This is
    the mechanism to reach for when the output is a *choice* rather than a
    number, and adding noise to the answer makes no sense -- picking the best
    bin, the best model, the best split point.

    The factor of 2 in the denominator is not decoration: it is what makes the
    guarantee hold, because changing one record can move the utility of the
    chosen candidate *and* of its competitors.

    Utility only ever enters through differences, so shifting it by a constant
    changes nothing. The implementation subtracts the maximum before
    exponentiating, which is numerically necessary and free.

    Parameters
    ----------
    candidates : sequence
        The discrete options.
    utility : array-like
        Utility of each candidate; higher is better.
    epsilon : float
        Privacy budget, positive.
    sensitivity : float
        Sensitivity of the utility function, positive.
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``selected``, ``index``, ``probabilities``, ``epsilon``.

    References
    ----------
    McSherry, F., & Talwar, K. (2007). Mechanism design via differential
        privacy. *FOCS 2007*, 94-103.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *Foundations and Trends in Theoretical
        Computer Science*, 9(3-4), 211-407.

    Examples
    --------
    The highest-utility candidate is the most likely, but not certain.

    >>> import numpy as np
    >>> r = dp_exponential_mechanism(["a", "b", "c"], [0.0, 5.0, 1.0],
    ...                              epsilon=2.0, seed=0)
    >>> int(np.argmax(r["probabilities"]))
    1
    >>> bool(r["probabilities"][1] < 1.0)
    True

    Shifting every utility by a constant leaves the distribution unchanged --
    only differences matter.

    >>> p1 = dp_exponential_mechanism("abc", [0.0, 5.0, 1.0], 2.0)["probabilities"]
    >>> p2 = dp_exponential_mechanism("abc", [10.0, 15.0, 11.0], 2.0)["probabilities"]
    >>> bool(np.allclose(p1, p2))
    True

    Larger epsilon concentrates on the best candidate.

    >>> lo = dp_exponential_mechanism("abc", [0.0, 5.0, 1.0], 0.1)["probabilities"][1]
    >>> hi = dp_exponential_mechanism("abc", [0.0, 5.0, 1.0], 10.0)["probabilities"][1]
    >>> bool(hi > lo)
    True

    >>> dp_exponential_mechanism(["a", "b"], [1.0])
    Traceback (most recent call last):
        ...
    ValueError: utility has 1 entries but there are 2 candidates
    """
    epsilon, _ = check_budget(epsilon)
    sensitivity = float(sensitivity)
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    cands = list(candidates)
    u = np.atleast_1d(np.asarray(utility, dtype=float)).ravel()
    if u.size != len(cands):
        raise ValueError(
            f"utility has {u.size} entries but there are {len(cands)} candidates"
        )
    if not np.all(np.isfinite(u)):
        raise ValueError("utility must be finite")
    # The 2 is required: one record can move the chosen candidate's utility
    # and its competitors'.
    logp = epsilon * u / (2.0 * sensitivity)
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    idx = int(np.random.default_rng(seed).choice(len(cands), p=p))
    return RichResult(
        title="Exponential mechanism",
        summary_lines=[("epsilon", epsilon), ("candidates", len(cands)),
                       ("selected", str(cands[idx]))],
        payload={
            "selected": cands[idx], "index": idx, "probabilities": p,
            "epsilon": epsilon, "sensitivity": sensitivity,
            "mechanism": "exponential", "method": "dp_exponential_mechanism",
        },
    )


def cheatsheet():
    return "dpexpm: p ~ exp(eps*u/(2*du)); for choices not numbers. The 2 is load-bearing"
