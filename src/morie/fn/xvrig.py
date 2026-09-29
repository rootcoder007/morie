"""Xavier (Glorot) weight initialization."""

from __future__ import annotations

from . import _winit as wi
from ._containers import DescriptiveResult


def xavier_init(
    fan_in: int,
    fan_out: int,
    distribution: str = "normal",
    seed: int | None = None,
) -> DescriptiveResult:
    r"""Xavier (Glorot) weight initialization.

    A ``fan_out x fan_in`` matrix (the layout ``y = W x``) with ``W ~ N(0,
    2 / (fan_in + fan_out))`` or ``U(-a, a)``, ``a = sqrt(6 / (fan_in +
    fan_out))`` (Glorot and Bengio 2010), from the Philox stream ``seed``
    (``None`` means 0) row by row.

    :param fan_in: Number of input units.
    :param fan_out: Number of output units.
    :param distribution: ``"normal"`` or ``"uniform"``.
    :param seed: Philox seed.
    :return: DescriptiveResult; ``value`` is the population standard
        deviation, ``extra["weights"]`` the matrix.

    References
    ----------
    Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural
    networks. *AISTATS*, 249-256.

    Examples
    --------
    >>> round(xavier_init(3, 2, seed=5).extra["weights"][1][2], 12)
    -0.639000509727
    """
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in and fan_out must be positive")
    if distribution not in ("normal", "uniform"):
        raise ValueError(f"distribution must be 'normal' or 'uniform', got {distribution}")
    method = "xavier_normal" if distribution == "normal" else "xavier_uniform"
    W = wi.draw(fan_out, fan_in, method, 1.0, 0 if seed is None else seed, fan_in, fan_out)
    _m, var = wi.moments(W)
    return DescriptiveResult(
        name="xavier_init",
        value=var**0.5,
        extra={"weights": W, "fan_in": fan_in, "fan_out": fan_out, "distribution": distribution},
    )


def cheatsheet() -> str:
    return "xavier_init(fan_in, fan_out, distribution, seed) -> Glorot weights (fan_out x fan_in)"


xvrig = xavier_init
