# morie.fn -- function file (rootcoder007/morie)
"""Randomized response -- Warner (1965)."""

from __future__ import annotations

import numpy as np

from ._dp import check_budget
from ._richresult import RichResult

__all__ = ["randomized_response_dp"]


def randomized_response_dp(truth, epsilon=1.0, seed=None):
    r"""Local differential privacy for binary answers.

    Each respondent reports their true bit with probability

    .. math::
        p = \frac{e^{\varepsilon}}{1 + e^{\varepsilon}},

    and flips it otherwise, which is exactly :math:`\varepsilon`-locally
    differentially private: the likelihood ratio between the two possible
    truths is :math:`p/(1-p) = e^{\varepsilon}`.

    This is the *local* model -- no trusted curator holds the raw data, because
    the randomisation happens before the answer leaves the respondent. The
    price is accuracy: the debiased estimate has standard error of order
    :math:`1/(\sqrt n\,(2p-1))`, which blows up as :math:`\varepsilon \to 0`.

    The raw proportion of reported ones is **biased toward 1/2** and must be
    debiased by
    :math:`\hat\pi = (\bar r - (1-p))/(2p - 1)`; ``estimate`` does this, and
    can fall outside [0, 1] in small samples, which is honest rather than a
    bug to clip away.

    Parameters
    ----------
    truth : array-like
        True bits in {0, 1}.
    epsilon : float
        Local privacy budget, positive.
    seed : int, optional
        Seed; leave ``None`` for a real release.

    Returns
    -------
    RichResult
        ``responses``, ``p_truth``, ``raw_proportion``, ``estimate``
        (debiased), ``se``.

    References
    ----------
    Warner, S. L. (1965). Randomized response: A survey technique for
        eliminating evasive answer bias. *JASA*, 60(309), 63-69.
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *Foundations and Trends in Theoretical
        Computer Science*, 9(3-4), 211-407.

    Examples
    --------
    The debiased estimate recovers the true proportion; the raw one does not.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> truth = (rng.random(20000) < 0.3).astype(int)
    >>> r = randomized_response_dp(truth, epsilon=1.0, seed=1)
    >>> bool(abs(r["estimate"] - 0.3) < 0.05)
    True
    >>> bool(abs(r["raw_proportion"] - 0.3) > 0.05)
    True

    The raw proportion is pulled toward 1/2, which is the bias being corrected.

    >>> bool(0.3 < r["raw_proportion"] < 0.5)
    True

    Truth-telling probability is the logistic of epsilon.

    >>> float(round(randomized_response_dp([1], epsilon=0.0 + 2.0)["p_truth"], 6))
    0.880797

    >>> randomized_response_dp([0, 1, 2], epsilon=1.0)
    Traceback (most recent call last):
        ...
    ValueError: truth must contain only 0 and 1
    """
    epsilon, _ = check_budget(epsilon)
    t = np.atleast_1d(np.asarray(truth, dtype=float)).ravel()
    if not np.all((t == 0) | (t == 1)):
        raise ValueError("truth must contain only 0 and 1")
    p = float(np.exp(epsilon) / (1.0 + np.exp(epsilon)))
    rng = np.random.default_rng(seed)
    keep = rng.random(t.size) < p
    resp = np.where(keep, t, 1.0 - t)
    raw = float(resp.mean())
    est = (raw - (1.0 - p)) / (2.0 * p - 1.0)
    n = t.size
    var = raw * (1 - raw) / max(n, 1) / (2.0 * p - 1.0) ** 2
    return RichResult(
        title="Randomized response (local DP)",
        summary_lines=[("epsilon", epsilon), ("p(truth)", p), ("n", n),
                       ("estimate", est)],
        payload={
            "responses": resp, "p_truth": p, "raw_proportion": raw,
            "estimate": float(est), "se": float(np.sqrt(max(var, 0.0))),
            "n": int(n), "epsilon": epsilon,
            "mechanism": "randomized_response",
            "method": "randomized_response_dp",
        },
    )


def cheatsheet():
    return "dprrep: LOCAL DP, no trusted curator; raw proportion is biased to 1/2 -- use `estimate`"
