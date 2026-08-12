r"""Page's likelihood-ratio CUSUM changepoint detector.

Lai, T. L. (1995) "Sequential changepoint detection in quality control
and dynamical systems", *Journal of the Royal Statistical Society B*
**57**(4), 613-658, section 2.4 and equation (2.3).

The setting is a sequence :math:`X_1, X_2, \ldots` that is i.i.d. with
density :math:`f_0` up to a changepoint :math:`\nu`, and i.i.d. with
:math:`f_1` from :math:`\nu` onward. Page (1954) scores each observation
by its log-likelihood ratio

.. math:: Z_i = \log\\frac{f_1(X_i)}{f_0(X_i)},

writes :math:`S_n = \sum_{i \le n} Z_i` with :math:`S_0 = 0`, and stops at

.. math::
    N = \inf\\left\\{ n : \max_{1 \le k \le n}
        \sum_{i=k}^{n} \log\\frac{f_1(X_i)}{f_0(X_i)} \ge c \\right\\}.

That inner maximum is what this module returns at every ``n``. It equals
:math:`S_n - \min_{0 \le i \le n} S_i`, which is why the detector needs
only one pass and O(1) state: the running minimum of the partial sums is
the best guess at where the change began.

Moustakides (1986) and Ritov (1990) proved (2.3) exactly minimax for the
worst-case expected delay; Lorden (1971) gave the asymptotics

.. math:: E_1(N) \sim \\frac{\log\gamma}{I(f_1, f_0)}, \qquad
          I(f_1, f_0) = E_{f_1}\\left[
              \log\\frac{f_1(X_1)}{f_0(X_1)}\\right],

the Kullback-Leibler information number, reported here as ``kl``. That
identity is the reason the threshold is on a log scale and the reason a
change that barely separates :math:`f_0` from :math:`f_1` takes so long
to find: the delay is inversely proportional to the KL divergence.

This is the simple-versus-simple case, where both densities are named.
Lai's *generalized* likelihood ratio, equation (2.4), replaces
:math:`f_1` by a supremum over an exponential family when the
out-of-control distribution is not specified in advance; it reduces to
(2.3) exactly when the family is the single point ``p1``.

Routes
------
``family`` selects the score, since (2.3) is stated for any pair of
densities and the useful ones differ by field:

``"bernoulli"``
    ``p0``/``p1`` are success probabilities and ``x`` is 0/1. The
    default -- the argument names are probabilities.
``"normal"``
    ``p0``/``p1`` are means of a Gaussian with known ``sd``; the score
    collapses to :math:`(\mu_1-\mu_0)(x - (\mu_0+\mu_1)/2)/\sigma^2`,
    the classical two-sided-chart score.
``"poisson"``
    ``p0``/``p1`` are rates and ``x`` counts.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["glr_test", "page_cusum", "glrtest"]

_FAMILIES = ("bernoulli", "normal", "poisson")


def _score_bernoulli(p0, p1):
    """Log f1/f0 for Bernoulli, plus its KL number under f1."""
    if not (0.0 < p0 < 1.0) or not (0.0 < p1 < 1.0):
        raise ValueError(
            "glr_test: bernoulli p0 and p1 must lie strictly in (0, 1), "
            "got %r and %r" % (p0, p1))
    a = math.log(p1 / p0)
    b = math.log((1.0 - p1) / (1.0 - p0))

    def z(v):
        if v not in (0.0, 1.0):
            raise ValueError(
                "glr_test: bernoulli data must be 0 or 1, got %r" % (v,))
        return a if v == 1.0 else b

    return z, p1 * a + (1.0 - p1) * b


def _score_normal(p0, p1, sd):
    """Log f1/f0 for a Gaussian mean shift with known sd."""
    if sd <= 0.0:
        raise ValueError("glr_test: sd must be positive, got %r" % (sd,))
    d = p1 - p0
    mid = 0.5 * (p0 + p1)
    s2 = sd * sd

    def z(v):
        return d * (v - mid) / s2

    # E_f1[Z] = d * (p1 - mid) / s2 = d^2 / (2 s2), the Gaussian KL.
    return z, d * d / (2.0 * s2)


def _score_poisson(p0, p1):
    """Log f1/f0 for Poisson rates."""
    if p0 <= 0.0 or p1 <= 0.0:
        raise ValueError(
            "glr_test: poisson rates must be positive, got %r and %r"
            % (p0, p1))
    lr = math.log(p1 / p0)

    def z(v):
        if v < 0.0 or v != math.floor(v):
            raise ValueError(
                "glr_test: poisson data must be non-negative integers, "
                "got %r" % (v,))
        return v * lr - (p1 - p0)

    return z, p1 * lr - (p1 - p0)


def glr_test(x, p0, p1, threshold=None, family="bernoulli", sd=1.0):
    r"""Page's likelihood-ratio CUSUM over ``x``.

    Parameters
    ----------
    x : array-like
        The observed sequence, in time order.
    p0, p1 : float
        In-control and out-of-control parameters, read according to
        ``family``.
    threshold : float, optional
        The stopping boundary :math:`c` of equation (2.3). When given,
        ``detected`` and ``stop_index`` report the first crossing.
    family : {"bernoulli", "normal", "poisson"}
        Which pair of densities the parameters name. See the module
        docstring.
    sd : float
        Known standard deviation, ``family="normal"`` only.

    Returns
    -------
    RichResult
        ``statistic`` is the terminal CUSUM
        :math:`\max_k \sum_{i=k}^{n} Z_i`; ``estimate`` is its running
        value at every ``n``; ``changepoint`` is the argmax's left end,
        the maximum-likelihood estimate of :math:`\nu`; ``kl`` is
        Lorden's information number.
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(xv)
    if n == 0:
        raise ValueError("glr_test: x must hold at least one observation")
    fam = str(family).lower()
    if fam not in _FAMILIES:
        raise ValueError(
            "glr_test: family must be one of %s, got %r"
            % (", ".join(_FAMILIES), family))
    p0 = float(p0)
    p1 = float(p1)
    if p0 == p1:
        raise ValueError(
            "glr_test: p0 and p1 must differ -- with one density there is "
            "no change to detect")

    if fam == "bernoulli":
        z, kl = _score_bernoulli(p0, p1)
    elif fam == "normal":
        z, kl = _score_normal(p0, p1, float(sd))
    else:
        z, kl = _score_poisson(p0, p1)

    # One pass. S is the running partial sum, smin the running minimum
    # over S_0 = 0 .. S_n, and S - smin is exactly the inner maximum of
    # equation (2.3) -- the k that attains it is the argmin of S, i.e.
    # the last time the evidence was at its weakest.
    scores = [0.0] * n
    cusum = [0.0] * n
    S = 0.0
    smin = 0.0
    smin_at = 0          # index (0-based) just BEFORE the putative change
    best = -math.inf
    best_k = 0
    stop_index = None
    for i in range(n):
        zi = z(float(xv[i]))
        scores[i] = zi
        S += zi
        # The minimum in Lai (2.3) runs over S_0 .. S_n INCLUSIVE, so it
        # must absorb the new S before the statistic is read. That is
        # what reflects the chart at zero: with the update after the
        # read, an all-in-control stretch would drift negative instead
        # of resting at 0.
        if S < smin:
            smin = S
            smin_at = i + 1
        val = S - smin
        cusum[i] = val
        if val > best:
            best = val
            best_k = smin_at
        if threshold is not None and stop_index is None and \
                val >= float(threshold):
            stop_index = i

    payload = {
        "estimate": cusum,
        "statistic": float(cusum[-1]),
        "max_statistic": float(best),
        "scores": scores,
        "changepoint": int(best_k),
        "kl": float(kl),
        "n": int(n),
        "family": fam,
        "p0": p0,
        "p1": p1,
        "method": "Page likelihood-ratio CUSUM (Lai 1995, eq. 2.3)",
    }
    if threshold is not None:
        payload["threshold"] = float(threshold)
        payload["detected"] = stop_index is not None
        payload["stop_index"] = (-1 if stop_index is None else int(stop_index))
        # Lorden (1971): E_1(N) ~ log(gamma) / I(f1, f0), and the
        # threshold is c = log(gamma), so the ARL to detection is c / kl.
        payload["expected_delay"] = (
            float(threshold) / kl if kl > 0 else float("inf"))
    return RichResult(payload=payload)


def cheatsheet():
    return ("glm: Page likelihood-ratio CUSUM, max_k sum_{i=k}^{n} "
            "log(f1/f0) (Lai 1995 eq. 2.3); families bernoulli/normal/"
            "poisson; KL number gives Lorden's delay log(gamma)/KL.")


page_cusum = glr_test
# compact alias per ledger/NAMING.md
glrtest = glr_test
