# morie.fn -- function file (rootcoder007/morie)
"""Batch-normalisation running statistics and inference transform."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bnrunstat", "alphazero_batch_norm"]


def bnrunstat(x, runmean=0.0, runvar=1.0, momentum=0.1, eps=1e-5,
              gamma=1.0, beta=0.0):
    """Update batch-norm moving statistics and normalise for inference.

    During training a batch-normalised activation is centred and scaled by
    the mini-batch moments; at inference time the normalisation must be
    deterministic, so the population moments are tracked by a moving
    average of the mini-batch moments.  Ioffe and Szegedy use the unbiased
    variance estimate Var[x] = (m/(m-1)) E_B[sigma_B^2], and the inference
    transform

        xhat = (x - E[x]) / sqrt(Var[x] + eps),   y = gamma xhat + beta.

    The moving-average update applied here is

        runmean <- (1 - momentum) runmean + momentum * mean(x)
        runvar  <- (1 - momentum) runvar
                   + momentum * (m/(m-1)) * biased_var(x)

    where biased_var divides by m and the m/(m-1) factor is exactly the
    unbiased correction the paper prescribes.

    Parameters
    ----------
    x : array-like
        Mini-batch of activations, length m >= 2.
    runmean, runvar : float
        Running population moments carried in from the previous batch.
    momentum : float
        Weight given to the new batch, in [0, 1].
    eps : float
        Numerical floor added inside the square root.
    gamma, beta : float
        Learned scale and shift.

    Returns
    -------
    RichResult
        ``runmean``, ``runvar``, ``batchmean``, ``batchvar``,
        ``batchvarunb``, ``normalized``, ``trainnorm``, ``m``.

    References
    ----------
    Ioffe, S. and Szegedy, C. (2015), "Batch normalization: accelerating
    deep network training by reducing internal covariate shift",
    arXiv:1502.03167.  Section 3.1 and Algorithm 2 give xhat =
    (x - E[x])/sqrt(Var[x]+eps) with Var[x] = (m/(m-1)) E_B[sigma_B^2],
    and name moving averages as the tracking scheme used in practice.
    Read from the ar5iv rendering of the arXiv source; a copy of the same
    paper is also in the local corpus.
    """
    x = C.vec(x)
    m = len(x)
    if m < 2:
        raise ValueError("need at least two activations to form a variance")
    momentum = float(momentum)
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must lie in [0, 1]")
    mu = sum(x) / m
    vb = sum((v - mu) ** 2 for v in x) / m
    vu = vb * m / (m - 1.0)
    rm = (1.0 - momentum) * float(runmean) + momentum * mu
    rv = (1.0 - momentum) * float(runvar) + momentum * vu
    g, b, eps = float(gamma), float(beta), float(eps)
    inf = [g * (v - rm) / math.sqrt(rv + eps) + b for v in x]
    trn = [g * (v - mu) / math.sqrt(vb + eps) + b for v in x]
    return RichResult(payload={
        "runmean": rm, "runvar": rv, "batchmean": mu, "batchvar": vb,
        "batchvarunb": vu, "normalized": inf, "trainnorm": trn, "m": m,
        "method": "Batch-norm running statistics (Ioffe-Szegedy 2015 Sect. 3.1)"})


alphazero_batch_norm = bnrunstat


def cheatsheet():
    return "agbtnm: Batch-normalisation running statistics and inference transform."
