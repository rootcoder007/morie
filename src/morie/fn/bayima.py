# morie.fn -- function file (rootcoder007/morie)
"""Importance sampling estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["importance_sampling"]


def importance_sampling(log_weights, values=None, normalized=True):
    r"""Importance-weighted expectation with its own reliability check.

    .. math::
       \hat\mu = \frac{\sum_i w_i h(x_i)}{\sum_i w_i},
       \qquad w_i = \frac{p(x_i)}{q(x_i)}

    Weights are taken on the LOG scale and stabilised by subtracting
    the maximum before exponentiating. This is not a nicety: raw
    likelihood ratios in more than a few dimensions overflow or
    underflow to zero, and the failure is silent -- the estimate comes
    back as ``nan`` or, worse, as the value of whichever single point
    survived.

    The estimator's reliability is entirely a question of weight
    concentration, and there is a standard diagnostic for it. The
    effective sample size

    .. math:: ESS = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}

    says how many independent draws the weighted sample is worth. An
    ESS of 5 out of 10 000 means the answer rests on five points,
    whatever its apparent precision. ``ess_fraction`` and
    ``max_weight_share`` make that visible rather than leaving it to be
    inferred from a standard error that will look fine either way.

    The normalised (self-normalising) form is biased at
    :math:`O(1/n)` but has far lower variance and does not require the
    target density to be normalised. That trade is almost always worth
    taking, which is why it is the default.

    Parameters
    ----------
    log_weights : array-like, shape (n,)
        :math:`\log p(x_i) - \log q(x_i)`.
    values : array-like, shape (n,) or (n, k), optional
        :math:`h(x_i)`. The mean weight is returned when omitted.
    normalized : bool

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ess``, ``ess_fraction``,
        ``max_weight_share``, ``weights``, ``reliable``,
        ``log_normalizer``.

    References
    ----------
    Geweke (1989), *Econometrica* 57:1317-1339.
    Kong (1992) for the effective sample size.
    Owen (2013), *Monte Carlo theory, methods and examples*, chapter 9.

    Examples
    --------
    >>> import numpy as np
    >>> out = importance_sampling(np.zeros(100), np.ones(100))
    >>> float(out["estimate"])
    1.0
    """
    lw = np.asarray(log_weights, dtype=float).ravel()
    n = lw.size
    if n < 1:
        raise ValueError("need at least one draw.")
    if np.all(~np.isfinite(lw)):
        raise ValueError("every log weight is non-finite.")
    m = float(np.max(lw[np.isfinite(lw)]))
    w = np.exp(lw - m)                     # stabilised before exponentiating
    w = np.where(np.isfinite(w), w, 0.0)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("all weights underflowed to zero.")
    ess = float(s ** 2 / np.sum(w ** 2))
    log_norm = float(m + np.log(s / n))

    if values is None:
        est = float(s / n) * np.exp(m)
        se = float(np.std(w * np.exp(m), ddof=1) / np.sqrt(n)) if n > 1 \
            else np.nan
        h = None
    else:
        h = np.asarray(values, dtype=float)
        if h.ndim == 1:
            h = h[:, None]
        if h.shape[0] != n:
            raise ValueError(
                "values has %d rows for %d weights." % (h.shape[0], n)
            )
        if normalized:
            est = (w @ h) / s
            # delta-method variance of a ratio of weighted sums
            resid = h - est[None, :]
            se = np.sqrt(np.sum((w[:, None] * resid) ** 2, axis=0)) / s
        else:
            est = (w @ h) / n * np.exp(m)
            se = np.std(w[:, None] * h * np.exp(m), axis=0, ddof=1) \
                / np.sqrt(n)
        if est.size == 1:
            est = float(est[0])
            se = float(se[0])

    reliable = bool(ess >= max(0.1 * n, 10))
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "weights": w / s,
            "ess": ess,
            "ess_fraction": float(ess / n),
            "max_weight_share": float(w.max() / s),
            "reliable": reliable,
            "reliability_note": (
                None if reliable else
                "the effective sample size is %.1f out of %d draws, so the "
                "answer rests on a handful of points; the standard error "
                "will look fine regardless" % (ess, n)
            ),
            "log_normalizer": log_norm,
            "normalized": bool(normalized),
            "bias_note": (
                "the self-normalising form is O(1/n) biased but has far "
                "lower variance and needs no normalised target; the "
                "unnormalised form is unbiased and usually much worse"
            ),
            "stabilization_note": (
                "weights are exponentiated after subtracting the maximum; "
                "raw likelihood ratios in more than a few dimensions "
                "under- or overflow silently"
            ),
            "n": int(n),
            "method": "Importance sampling (%s)"
                      % ("self-normalised" if normalized else "unnormalised"),
        }
    )


def cheatsheet():
    return (
        "bayima: log-stabilised importance sampling with the effective "
        "sample size that decides whether to believe it"
    )
