# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Monte Carlo dropout: leave dropout on at inference for uncertainty."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_mc_dropout"]

_METHOD = "Monte Carlo dropout uncertainty"


def geron_mc_dropout(model, x, K=100, p=0.5, seed=0):
    """
    Monte Carlo dropout: leave dropout on at inference for uncertainty.

    Formula: y_hat_mean, y_hat_var estimated from K forward passes with dropout

    Dropout is normally switched off at test time; leaving it on turns a
    deterministic network into a stochastic one whose spread across
    passes is a usable uncertainty estimate -- and it costs no retraining
    at all, only ``K`` forward passes.

    The mask is **inverted dropout**: kept units are divided by
    ``1 - p``, so the expected input to the model is unchanged and the
    mean over passes stays an unbiased prediction.  Scaling at training
    time only, and then not scaling here, would shift every MC-dropout
    prediction by a factor of ``1 - p`` -- a bias that looks exactly like
    a badly calibrated model.

    The standard error of the mean, ``sd/sqrt(K)``, is returned next to
    the standard deviation: the first shrinks with more passes, the
    second does not. Confusing them is how MC dropout gets reported as
    far more certain than it is.

    ``model`` is caller-supplied and its contract enforced:
    ``model(x_masked) -> array`` of a fixed shape, finite, once per pass.

    Parameters
    ----------
    model : callable
        ``model(x_with_dropout_applied) -> prediction``.
    x : array-like
        Single input; the mask has this shape.
    K : int
        Number of stochastic passes (at least 2).
    p : float
        Drop probability in [0, 1).
    seed : int
        Seed for the masks.

    Returns
    -------
    result : RichResult
        Keys: mean, var, std, sem, samples, predictive_entropy,
        estimate, n, method.

    Examples
    --------
    A linear model with p = 0: no units are dropped, so every pass is
    identical and the variance is exactly zero.

    >>> f = lambda z: np.asarray([float(np.sum(z))])
    >>> r = geron_mc_dropout(f, [1.0, 2.0, 3.0], K=10, p=0.0)
    >>> float(r["mean"][0]), float(r["var"][0])
    (6.0, 0.0)

    With dropout on, inverted scaling keeps the mean unbiased: the
    average over many passes stays near the deterministic sum of 6.

    >>> d = geron_mc_dropout(f, [1.0, 2.0, 3.0], K=4000, p=0.5, seed=0)
    >>> bool(abs(float(d["mean"][0]) - 6.0) < 0.2)
    True
    >>> bool(d["var"][0] > 0)
    True

    The standard error shrinks with K while the standard deviation does
    not:

    >>> a = geron_mc_dropout(f, [1.0, 2.0, 3.0], K=500, p=0.5, seed=1)
    >>> b = geron_mc_dropout(f, [1.0, 2.0, 3.0], K=8000, p=0.5, seed=1)
    >>> bool(b["sem"][0] < a["sem"][0] / 2)
    True
    >>> bool(abs(b["std"][0] - a["std"][0]) < 0.5)
    True

    A model whose output shape moves between passes is refused:

    >>> wobbly = lambda z: np.zeros(int(np.sum(z != 0)) + 1)
    >>> geron_mc_dropout(wobbly, [1.0, 2.0, 3.0], K=50, p=0.5, seed=3)
    Traceback (most recent call last):
        ...
    ValueError: geron_mc_dropout: model returned shape (3,) on pass 5 but (2,) on pass 1

    References
    ----------
    Géron Ch 11
    """
    if not callable(model):
        raise ValueError(f"geron_mc_dropout: model must be callable, got {type(model).__name__}")
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("geron_mc_dropout: x is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_mc_dropout: x contains non-finite values")
    passes = int(K)
    if passes < 2:
        raise ValueError(f"geron_mc_dropout: K must be at least 2 passes to have a spread, got {K!r}")
    rate = float(p)
    if not (0.0 <= rate < 1.0):
        raise ValueError(f"geron_mc_dropout: p must lie in [0, 1); p=1 drops everything, got {p!r}")

    rng = np.random.default_rng(int(seed))
    samples = []
    shape = None
    for i in range(passes):
        mask = (rng.random(a.shape) >= rate).astype(float) / (1.0 - rate)
        out = np.atleast_1d(np.asarray(model(a * mask), dtype=float))
        if shape is None:
            shape = out.shape
        elif out.shape != shape:
            raise ValueError(
                f"geron_mc_dropout: model returned shape {out.shape} on pass {i + 1} but {shape} on pass 1"
            )
        if not np.all(np.isfinite(out)):
            raise ValueError(f"geron_mc_dropout: model returned a non-finite prediction on pass {i + 1}")
        samples.append(out)

    S = np.asarray(samples)
    mean = S.mean(axis=0)
    var = S.var(axis=0, ddof=1)
    std = np.sqrt(var)
    sem = std / np.sqrt(passes)

    # Predictive entropy of the mean, when the output looks like a distribution.
    ent = None
    if mean.ndim == 1 and mean.size > 1 and np.all(mean >= 0) and abs(float(mean.sum()) - 1.0) < 1e-6:
        ent = float(-np.sum(mean * np.log(np.clip(mean, 1e-300, None))))

    return RichResult(
        title="MC dropout",
        summary_lines=[
            ("Passes", passes),
            ("Drop rate", rate),
            ("Mean prediction", float(np.mean(mean))),
            ("Mean sd across passes", float(np.mean(std))),
        ],
        interpretation=(
            "sd is the model's predictive spread and does not shrink with K; sem is the Monte Carlo "
            "error of the estimate and does. Report the first as uncertainty."
        ),
        payload={
            "mean": mean,
            "var": var,
            "std": std,
            "sem": sem,
            "samples": S,
            "predictive_entropy": ent,
            "K": passes,
            "p": rate,
            "estimate": float(np.mean(mean)),
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmcd: MC dropout -- K inverted-dropout passes; sd is uncertainty, sem is Monte Carlo error"
