# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scaled ELU (SELU) for self-normalizing networks."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_selu"]

# Klambauer et al. (2017) fixed point: the pair that leaves activations at
# mean 0 / variance 1 under a standard-normal pre-activation.
SELU_LAMBDA = 1.0507009873554804934193349852946
SELU_ALPHA = 1.6732632423543772848170429916717


def geron_selu(z, lam=SELU_LAMBDA, alpha=SELU_ALPHA):
    """
    Scaled ELU (SELU) for self-normalizing networks.

    Formula: SELU(z) = lambda * ELU(z, alpha); lambda, alpha ~ (1.0507, 1.6733)

    ``ELU(z, alpha) = z`` for ``z > 0`` and ``alpha*(exp(z) - 1)``
    otherwise, so SELU is continuous at 0 and its derivative is
    ``lambda`` on the positive branch and ``lambda*alpha*exp(z)`` on the
    negative one. The self-normalising property is *checked*, not
    asserted: the mean and variance of the returned activations are in
    the payload, and for a standard-normal input they sit near (0, 1).

    Parameters
    ----------
    z : array-like
        Pre-activation values. Must be finite.
    lam, alpha : float
        Scale and negative-saturation constants; both must be positive.
        Defaults are the self-normalising fixed point.

    Returns
    -------
    result : RichResult
        Keys: a, grad, mean, var, estimate, n, method.

    Examples
    --------
    >>> r = geron_selu([-1.0, 0.0, 1.0])
    >>> float(r["a"][1])
    0.0
    >>> round(float(r["a"][2]), 6)
    1.050701
    >>> round(float(r["a"][0]), 6)
    -1.111331
    >>> round(float(r["grad"][2]), 6)
    1.050701

    References
    ----------
    Géron Ch 11
    """
    x = np.atleast_1d(np.asarray(z, dtype=float))
    if x.size == 0:
        raise ValueError("geron_selu: z is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("geron_selu: z contains non-finite values")
    lm, al = float(lam), float(alpha)
    if not (np.isfinite(lm) and lm > 0) or not (np.isfinite(al) and al > 0):
        raise ValueError(f"geron_selu: lam and alpha must be positive and finite, got lam={lm}, alpha={al}")

    neg = x <= 0
    a = np.where(neg, al * np.expm1(np.minimum(x, 0.0)), x) * lm
    grad = np.where(neg, lm * al * np.exp(np.minimum(x, 0.0)), lm)

    return RichResult(
        title="SELU activation",
        summary_lines=[
            ("Mean activation", float(np.mean(a))),
            ("Variance", float(np.var(a))),
            ("lambda", lm),
            ("alpha", al),
        ],
        interpretation=(
            "With standard-normal inputs and LeCun-normal weights the activation mean and variance "
            "stay near 0 and 1, which is the self-normalising property."
        ),
        payload={
            "a": a,
            "grad": grad,
            "mean": float(np.mean(a)),
            "var": float(np.var(a)),
            "lam": lm,
            "alpha": al,
            "estimate": float(np.mean(a)),
            "n": int(x.size),
            "method": "Scaled ELU with the Klambauer self-normalising constants",
        },
    )


def cheatsheet():
    return "hmselu: Scaled ELU (SELU) for self-normalizing networks"
