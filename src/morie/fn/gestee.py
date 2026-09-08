# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differentially private mean, with the clipping bias made visible.

Dwork and Roth (2014), *The Algorithmic Foundations of Differential
Privacy*, Sec 3.3 and Thm A.1; Karwa and Vadhan (2018),
*Finite Sample Differentially Private Confidence Intervals*,
arXiv:1711.03908.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gauss_subgaussian_estimator", "dp_mean_error_curve"]

_METHOD = "Differentially private clipped mean (Laplace / Gaussian mechanism)"


def _z(q):
    """Standard normal quantile by bisection on the error function."""
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gauss_subgaussian_estimator(y, C=None, epsilon=1.0, n=None,
                                mechanism="laplace", delta=1e-6,
                                alpha=0.05, seed=None, lower=None):
    """Private mean of a bounded sample.

    The estimator has three stages and every one of them costs
    something:

    1. **Clip** each observation to ``[lower, lower + C]``. This is what
       bounds the sensitivity, and it is also the only stage that
       introduces *bias*. Nothing downstream can remove that bias.
    2. **Average.** Under replace-one adjacency the sensitivity of the
       mean of ``n`` clipped values is :math:`\\Delta = C/n`.
    3. **Perturb.** Laplace noise :math:`\\mathrm{Lap}(\\Delta/\\epsilon)`
       gives :math:`\\epsilon`-DP; Gaussian noise with
       :math:`\\sigma = \\Delta\\sqrt{2\\ln(1.25/\\delta)}/\\epsilon`
       gives :math:`(\\epsilon, \\delta)`-DP.

    The reason to return the pieces separately is that the error is a
    sum of two terms that move differently as ``C`` moves: a tight clip
    range makes the noise small and can make the bias large, a loose
    one does the reverse. Whether that amounts to a genuine trade-off
    depends on the shape of the data, and it often does not --
    :func:`dp_mean_error_curve` traces the two terms so the question
    can be settled per dataset rather than assumed.

    Where an interior optimum does exist it depends on the data, which
    is itself private, so choosing ``C`` from the sample and then
    reporting only the private mean leaks. ``C`` should come from prior
    knowledge of the domain, not from ``y``. When ``C`` is left as None
    one is taken from the observed range purely so the function is
    runnable, and the payload flags it.

    Parameters
    ----------
    y : array-like
        Sample.
    C : float, optional
        Width of the clipping interval. See the caveat above.
    epsilon : float
        Privacy parameter, > 0.
    n : int, optional
        Denominator to use in the sensitivity. Defaults to ``len(y)``.
    mechanism : {"laplace", "gaussian"}
    delta : float
        Only used by the Gaussian mechanism.
    alpha : float
        Two-sided level for the intervals.
    seed : int, optional
        Seed for the noise draw. Present for reproducible testing; a
        deployment must not fix it.
    lower : float, optional
        Lower clipping bound. Defaults to ``min(y)`` when ``C`` is also
        defaulted, else ``mean(y) - C/2``.

    Returns
    -------
    RichResult
        ``estimate`` (private), ``non_private_mean``, ``clipped_mean``,
        ``clipping_bias``, ``noise_scale``, ``noise_sd``,
        ``sensitivity``, ``ci_lower``/``ci_upper`` (accounting for the
        noise), ``ci_naive_lower``/``ci_naive_upper`` (not accounting
        for it), ``n_clipped``.

    References
    ----------
    Dwork C, Roth A (2014), Sec 3.3 and Thm A.1.
    Karwa V, Vadhan S (2018) arXiv:1711.03908.

    Examples
    --------
    >>> import numpy as np
    >>> out = gauss_subgaussian_estimator(np.arange(100.0), C=100.0,
    ...                                   epsilon=1.0, lower=0.0, seed=0)
    >>> bool(abs(out["estimate"] - 49.5) < 5)
    True
    >>> out["sensitivity"]
    1.0
    """
    x = np.asarray(y, dtype=float).ravel()
    x = x[np.isfinite(x)]
    m = x.size
    if m < 1:
        raise ValueError("y must contain at least one finite value.")
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"epsilon must be positive and finite; got {epsilon}.")
    if mechanism not in ("laplace", "gaussian"):
        raise ValueError('mechanism must be "laplace" or "gaussian".')
    nn = m if n is None else int(n)
    if nn < 1:
        raise ValueError(f"n must be at least 1; got {n}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")

    c_from_data = C is None
    if c_from_data:
        rng_lo, rng_hi = float(np.min(x)), float(np.max(x))
        width = rng_hi - rng_lo
        C_ = width if width > 0 else 1.0
        lo = rng_lo
    else:
        C_ = float(C)
        if not np.isfinite(C_) or C_ <= 0:
            raise ValueError(f"C must be positive and finite; got {C}.")
        lo = float(np.mean(x)) - C_ / 2
    if lower is not None:
        lo = float(lower)
    hi = lo + C_

    clipped = np.clip(x, lo, hi)
    n_clipped = int(np.sum((x < lo) | (x > hi)))
    mu_clip = float(np.mean(clipped))
    mu_raw = float(np.mean(x))

    sens = C_ / nn
    rng = np.random.default_rng(seed)
    if mechanism == "laplace":
        scale = sens / eps
        noise = float(rng.laplace(0.0, scale))
        noise_sd = float(math.sqrt(2.0) * scale)
    else:
        d = float(delta)
        if not 0 < d < 1:
            raise ValueError(f"delta must lie in (0, 1); got {delta}.")
        scale = sens * math.sqrt(2.0 * math.log(1.25 / d)) / eps
        noise = float(rng.normal(0.0, scale))
        noise_sd = float(scale)
    priv = mu_clip + noise

    s = float(np.std(clipped, ddof=1)) if m > 1 else 0.0
    samp_se = s / math.sqrt(m)
    zc = _z(1 - alpha / 2)
    # naive: pretend the noise was never added
    ci_naive = (priv - zc * samp_se, priv + zc * samp_se)
    # split the level between sampling error and mechanism noise, using
    # the exact Laplace tail rather than a normal approximation to it
    z_half = _z(1 - alpha / 4)
    if mechanism == "laplace":
        noise_half = scale * math.log(2.0 / alpha)
    else:
        noise_half = z_half * scale
    half = z_half * samp_se + noise_half

    out = RichResult(
        title="Differentially private mean",
        summary_lines=[
            ("Private estimate", priv),
            ("Clipped-sample mean", mu_clip),
            ("Clipping bias", mu_clip - mu_raw),
            ("Noise SD", noise_sd),
            ("epsilon", eps),
        ],
        payload={
            "estimate": priv,
            "non_private_mean": mu_raw,
            "clipped_mean": mu_clip,
            "clipping_bias": mu_clip - mu_raw,
            "sensitivity": float(sens),
            "noise_scale": float(scale),
            "noise_sd": noise_sd,
            "noise_drawn": noise,
            "mechanism": mechanism,
            "epsilon": eps,
            "delta": float(delta) if mechanism == "gaussian" else None,
            "clip_lower": lo,
            "clip_upper": hi,
            "clip_width": C_,
            "n_clipped": n_clipped,
            "sampling_se": float(samp_se),
            "total_se": float(math.sqrt(samp_se ** 2 + noise_sd ** 2)),
            "ci_lower": float(priv - half),
            "ci_upper": float(priv + half),
            "ci_naive_lower": float(ci_naive[0]),
            "ci_naive_upper": float(ci_naive[1]),
            "n": m,
            "n_denominator": nn,
            "method": _METHOD,
        },
        interpretation=(
            "The interval widens by the mechanism noise as well as the "
            "sampling error. Reporting the naive interval would understate "
            "the uncertainty by exactly the price of the privacy."
        ),
    )
    if c_from_data:
        out.warnings.append(
            "C was taken from the range of y, which is itself a function of "
            "the private data. The stated guarantee does not hold for that "
            "choice. Supply C from prior domain knowledge."
        )
    if n_clipped:
        out.warnings.append(
            f"{n_clipped} of {m} observations were clipped, biasing the "
            f"estimate by {mu_clip - mu_raw:+.4g} before any noise was added."
        )
    if mechanism == "gaussian" and eps >= 1.0:
        out.warnings.append(
            "The sigma used is the Dwork-Roth Thm A.1 bound, which is only "
            "proved for epsilon < 1. At epsilon >= 1 use the analytic "
            "Gaussian mechanism of Balle and Wang (2018) instead."
        )
    return out


def dp_mean_error_curve(y, widths, epsilon=1.0, reps=200, seed=0,
                        mechanism="laplace", lower=None):
    """Total private error as a function of the clipping width.

    Traces bias, noise and root mean squared error across candidate
    values of ``C`` so the trade-off in
    :func:`gauss_subgaussian_estimator` can be seen rather than
    asserted. The minimising width is a property of the population;
    reading it off a sample and then using it is a privacy leak, so
    this is a study tool, not a tuner.

    The trade-off is often described as though it were automatic --
    tighter clipping always trading bias for noise -- and it is not.
    Two things have to hold for the curve to have an interior minimum:

    * The distribution must be **skewed**. Clipping a symmetric
      distribution with a window centred on its mean removes equal mass
      from both tails, so the bias cancels at every width. Measured on
      a standard normal at ``epsilon = 0.05``, the bias stays below
      0.006 across widths from 0.5 to 200 while the noise grows by a
      factor of 400; the RMSE is then monotone increasing and the best
      width is the smallest one offered.
    * The window must not be free to follow the mean. With ``lower``
      left as None the window is centred on the sample mean, which is
      the most favourable placement there is. A real deployment fixes
      the window from the domain, and pays more.

    On an exponential sample the interior minimum does appear, landing
    strictly inside the same grid. Note also that the bias need not
    be monotone in the width when the window slides: on a lognormal
    sample it runs -0.09, -0.18, -0.31, -0.27, -0.14, -0.03 as the
    width grows, worsening before it improves, because the lower edge
    drops below the support while the upper edge is still inside it.

    Returns
    -------
    RichResult with ``widths``, ``bias``, ``noise_sd``, ``rmse``,
    ``best_width``, ``interior_minimum``.
    """
    x = np.asarray(y, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("y must contain at least two finite values.")
    w = np.asarray(widths, dtype=float).ravel()
    if w.size < 1 or np.any(w <= 0):
        raise ValueError("widths must be positive and non-empty.")
    if mechanism not in ("laplace", "gaussian"):
        raise ValueError('mechanism must be "laplace" or "gaussian".')
    truth = float(np.mean(x))
    n = x.size
    rng = np.random.default_rng(seed)
    bias = np.empty(w.size)
    nsd = np.empty(w.size)
    rmse = np.empty(w.size)
    for i, C_ in enumerate(w):
        lo = truth - C_ / 2 if lower is None else float(lower)
        mu_clip = float(np.mean(np.clip(x, lo, lo + C_)))
        bias[i] = mu_clip - truth
        sens = C_ / n
        if mechanism == "laplace":
            b = sens / epsilon
            draws = rng.laplace(0.0, b, size=reps)
            nsd[i] = math.sqrt(2.0) * b
        else:
            sg = sens * math.sqrt(2.0 * math.log(1.25 / 1e-6)) / epsilon
            draws = rng.normal(0.0, sg, size=reps)
            nsd[i] = sg
        rmse[i] = float(np.sqrt(np.mean((mu_clip + draws - truth) ** 2)))
    j = int(np.argmin(rmse))
    interior = 0 < j < w.size - 1
    out = RichResult(
        title="Private mean error against clipping width",
        summary_lines=[
            ("Best width", float(w[j])),
            ("RMSE there", float(rmse[j])),
            ("Bias there", float(bias[j])),
            ("Interior minimum", interior),
        ],
        payload={
            "widths": w,
            "bias": bias,
            "noise_sd": nsd,
            "rmse": rmse,
            "best_width": float(w[j]),
            "best_rmse": float(rmse[j]),
            "interior_minimum": interior,
            "max_abs_bias": float(np.max(np.abs(bias))),
            "estimate": float(w[j]),
            "n": n,
            "method": "Clipping-width error curve for the DP mean",
        },
        interpretation=(
            "The clipping width trades bias against noise only when the "
            "clipped tail is asymmetric. Where it is not, the bias cancels "
            "and the smallest width offered simply wins."
        ),
    )
    if not interior:
        out.warnings.append(
            f"The minimising width is at the edge of the grid ({w[j]:g}). "
            "Either the grid does not bracket the optimum, or -- when the "
            "largest absolute bias is near zero, here "
            f"{float(np.max(np.abs(bias))):.4g} -- there is no trade-off to "
            "find, because a centred window on a symmetric sample clips both "
            "tails equally."
        )
    return out


def cheatsheet():
    return (
        "gestee: differentially private clipped mean, Laplace or Gaussian "
        "mechanism, reporting the clipping bias and the noise separately"
    )
