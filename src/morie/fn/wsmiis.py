# morie.fn -- function file (rootcoder007/morie)
"""Importance sampling, MacKay Sec. 29.2."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_importance_sampling"]


def wasserman_importance_sampling(f, p, q, n=None, samples=None,
                                  normalised=False, seed=0):
    r"""Importance sampling, MacKay Eqs. (29.21)-(29.22).

    Draw :math:`x^{(r)}` from a sampler density :math:`Q`, weight
    each by

    .. math:: w_r \equiv \frac{P^*(x^{(r)})}{Q^*(x^{(r)})}
              \tag{29.21}

    and estimate :math:`\Phi = \mathbb E_P[\phi]` by

    .. math:: \hat\Phi = \frac{\sum_r w_r \phi(x^{(r)})}{\sum_r w_r}.
              \tag{29.22}

    **(29.22) is self-normalised and that is not a detail.** It
    divides by :math:`\sum_r w_r`, so :math:`P^*` and :math:`Q^*`
    need only be known up to multiplicative constants -- which is the
    usual situation, and the reason importance sampling is worth
    doing. The unnormalised alternative
    :math:`\frac1R\sum_r w_r\phi(x^{(r)})` is only valid when both
    densities are exactly normalised; it is available here through
    ``normalised=True`` and is unbiased, where (29.22) is biased for
    small :math:`R` and consistent as :math:`R` grows (MacKay's
    Exercise 29.1).

    The book is blunt about the failure mode, and so is this
    function's output. The variance of :math:`\hat\Phi` is hard to
    estimate, because "the empirical variances of the quantities
    :math:`w_r` and :math:`w_r\phi(x^{(r)})` are not necessarily a
    good guide to the true variances". If :math:`Q` is small where
    :math:`|\phi P^*|` is large, no sample may ever land in that
    region -- the estimate is then "drastically wrong, and there
    would be no indication in the empirical variance". The reported
    ``effective_sample_size`` and ``max_weight_share`` are diagnostics
    for exactly that, not guarantees: they can only see regions that
    were actually sampled.

    The book's practical conclusion is that an importance sampler
    should have HEAVY TAILS. Its worked example has a Gaussian
    sampler still wrong after a million draws while a Cauchy sampler
    converges after about five thousand.

    Parameters
    ----------
    f : callable
        The function :math:`\phi` whose mean under :math:`P` is
        wanted.
    p : callable
        :math:`P^*`, the target up to a constant.
    q : callable
        :math:`Q^*`, the sampler density up to a constant.
    n : int, optional
        Number of draws, when ``samples`` is not supplied.
    samples : array-like, optional
        Draws from :math:`Q`. Required unless ``q`` also exposes a
        ``rvs`` attribute.
    normalised : bool, default False
        Use the unnormalised estimator instead of (29.22).
    seed : int, default 0
        Seed, used only when drawing via ``q.rvs``.

    Returns
    -------
    RichResult
        keys: ``estimate``, ``weights``, ``self_normalised``,
        ``effective_sample_size``, ``ess_fraction``,
        ``max_weight_share``, ``n``, ``diagnostics_are_not_guarantees``,
        ``heavy_tail_advice``, ``method``.

    References
    ----------
    MacKay, D. J. C. (2003), *Information Theory, Inference, and
    Learning Algorithms*, Cambridge University Press, Sec. 29.2,
    Eqs. (29.20)-(29.22) and the cautionary illustration of
    Fig. 29.6. Read from the PDF. Kahn and Marshall (1953).
    """
    if samples is None:
        if not hasattr(q, "rvs"):
            raise ValueError(
                "supply samples drawn from q, or a q with an rvs method; "
                "importance sampling cannot invent its own draws.")
        if n is None:
            raise ValueError("supply n when drawing samples from q.rvs.")
        samples = q.rvs(size=int(n), random_state=int(seed))
    xs = np.asarray(samples, dtype=float)
    R = xs.shape[0]
    if R < 2:
        raise ValueError(f"need at least 2 draws, got {R}.")
    pv = np.asarray(p(xs), dtype=float).ravel()
    qv = np.asarray(q(xs), dtype=float).ravel()
    if pv.size != R or qv.size != R:
        raise ValueError("p and q must return one value per draw.")
    if np.any(qv <= 0):
        raise ValueError(
            "the sampler density is zero or negative at a drawn point; "
            "Q must be positive wherever P is (MacKay's Exercise 29.1).")
    w = pv / qv                                          # (29.21)
    phi = np.asarray(f(xs), dtype=float).ravel()
    tot = float(w.sum())
    if tot <= 0:
        raise ValueError("every importance weight is zero; the sampler and "
                         "the target have no overlap in the drawn region.")
    est = float(np.mean(w * phi)) if normalised else float((w * phi).sum() / tot)
    ess = float(tot ** 2 / np.sum(w ** 2))
    return RichResult(payload={
        "estimate": est, "weights": w,
        "self_normalised": not normalised,
        "effective_sample_size": ess, "ess_fraction": ess / R,
        "max_weight_share": float(w.max() / tot),
        "n": int(R),
        "diagnostics_are_not_guarantees":
            "the effective sample size can only see regions that were "
            "actually sampled; if Q is small where |phi P*| is large the "
            "estimate is wrong with no empirical sign of it",
        "heavy_tail_advice":
            "an importance sampler should have HEAVY TAILS: MacKay's "
            "Fig. 29.6 has a Gaussian sampler still wrong after 10^6 draws "
            "where a Cauchy sampler converges after about 5000",
        "method": "Importance sampling, MacKay (29.21) weights and (29.22) "
                  "self-normalised estimator"})


def cheatsheet():
    return "wsmiis: (29.22) is self-normalised, so P* and Q* need only be known up to a constant"
