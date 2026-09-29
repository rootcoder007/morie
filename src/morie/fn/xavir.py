# morie.fn -- function file (rootcoder007/morie)
"""Generate Xavier/Glorot weight initialization matrix."""

from __future__ import annotations

from . import _winit as wi
from ._containers import DescriptiveResult


def xavier_init(
    fan_in: int,
    fan_out: int,
    seed: int = 42,
    uniform: bool = True,
) -> DescriptiveResult:
    r"""Generate Xavier/Glorot weight initialization matrix.

    ``W ~ U(-a, a)``, ``a = sqrt(6 / (fan_in + fan_out))`` (``uniform``), or
    ``W ~ N(0, 2 / (fan_in + fan_out))``, both with variance ``2 / (fan_in
    + fan_out)`` (Glorot and Bengio 2010), drawn from the Philox stream
    ``seed`` row by row (:func:`morie.fn.vctrs.weight_init`).

    :param fan_in: Number of input units.
    :param fan_out: Number of output units.
    :param seed: Philox seed.
    :param uniform: Uniform (True) or normal (False).
    :return: DescriptiveResult; ``value`` is the population standard
        deviation of the weights, ``extra["weights"]`` the ``fan_in x
        fan_out`` matrix.

    References
    ----------
    Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural
    networks. *AISTATS*, 249-256.

    Examples
    --------
    >>> round(xavier_init(4, 3, seed=2).extra["weights"][0][0], 12)
    -0.138044608242
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError(f"fan_in and fan_out must be > 0, got {fan_in}, {fan_out}.")
    method = "xavier_uniform" if uniform else "xavier_normal"
    W = wi.draw(fan_in, fan_out, method, 1.0, seed, fan_in, fan_out)
    m, var = wi.moments(W)
    return DescriptiveResult(
        name="Xavier Initialization",
        value=var**0.5,
        extra={
            "weights": W,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "mean": m,
            "std": var**0.5,
            "shape": (fan_in, fan_out),
            "method": "uniform" if uniform else "normal",
        },
    )


short = xavier_init


def cheatsheet() -> str:
    return "xavir() -> Generate Xavier/Glorot weight initialization matrix"


# compact alias per ledger/NAMING.md
xavierinit = xavier_init
