# morie.fn -- function file (rootcoder007/morie)
"""Plug-in estimator of a statistical functional."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_plug_in_estimator"]


def wasserman_plug_in_estimator(data, T, B=1000, seed=0, se=True):
    r"""The plug-in estimator :math:`\hat\theta = T(\hat F_n)` of a
    statistical functional :math:`\theta = T(F)`.

    Evaluate the functional at the empirical distribution instead of
    the unknown ``F``. Nothing about ``T`` is assumed here beyond
    being callable on a sample, so this works for the mean, the
    median, a quantile, the variance, a correlation, or anything else
    expressible as a functional.

    The standard error is where the content is. A functional's
    plug-in estimator is asymptotically normal when ``T`` is
    Hadamard-differentiable at ``F`` tangentially to the right
    subspace, which is the functional delta method: if
    :math:`\sqrt n(\hat F_n - F) \rightsquigarrow \mathbb G` then
    :math:`\sqrt n(T(\hat F_n) - T(F)) \rightsquigarrow
    \dot T_F(\mathbb G)`. That is a statement about ``T``, not about
    the data, and this function cannot check it -- ``T`` is an opaque
    callable. What it CAN do is report a nonparametric bootstrap
    standard error, which Kosorok's Ch. 10 shows is consistent for
    exactly the same class of functionals, and say plainly that its
    validity rests on a differentiability condition the caller is
    responsible for.

    A functional that is NOT differentiable in that sense -- the
    supremum of a density, say -- will still return a number here.
    It will just be the wrong number, and no diagnostic in this
    output will say so.

    Parameters
    ----------
    data : array-like
        Sample; rows are observations.
    T : callable
        The functional, applied to a sample.
    B : int, default 1000
        Bootstrap replicates for the standard error.
    seed : int, default 0
        Resampling seed.
    se : bool, default True
        Compute the bootstrap standard error.

    Returns
    -------
    RichResult
        keys: ``estimate``, ``se``, ``bootstrap_bias``,
        ``replicates``, ``ci_percentile``, ``n``, ``B``,
        ``validity_condition``, ``method``.

    References
    ----------
    Kosorok, M. R. (2008), *Introduction to Empirical Processes and
    Semiparametric Inference*, Springer. Ch. 12 (the functional delta
    method) and Sec. 2.2.4; Ch. 10 for the bootstrap of Donsker
    classes. von Mises (1947).
    """
    from ._wsm import bootstrap_replicates

    d = np.asarray(data, dtype=float)
    n = d.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    if not callable(T):
        raise ValueError("T must be callable on a sample.")
    est = float(T(d))
    if not se:
        return RichResult(payload={
            "estimate": est, "se": None, "n": int(n), "B": 0,
            "method": "Plug-in estimator T(F_n), no standard error requested"})
    reps = bootstrap_replicates(d, T, B=B, seed=seed)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return RichResult(payload={
        "estimate": est,
        "se": float(np.std(reps, ddof=1)),
        "bootstrap_bias": float(np.mean(reps) - est),
        "replicates": reps,
        "ci_percentile": (float(lo), float(hi)),
        "n": int(n), "B": int(len(reps)),
        "validity_condition":
            "asymptotic normality needs T to be Hadamard-differentiable at F "
            "tangentially to the relevant subspace (functional delta method); "
            "this function cannot verify that for an opaque callable, and a "
            "non-differentiable T returns a number that is simply wrong",
        "method": "Plug-in estimator theta_hat = T(F_n) with a nonparametric bootstrap SE"})


def cheatsheet():
    return "wsmpst: T(F_n) is always computable; its SE is only meaningful if T is Hadamard-differentiable"
