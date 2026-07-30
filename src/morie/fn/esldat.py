# morie.fn -- function file (rootcoder007/morie)
"""Dropout -- Srivastava et al. (2014), ESL 2nd-ed. supplement."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_dropout"]


def esl_dropout(X, p=0.5, training=True, seed=0):
    r"""Apply inverted dropout to an activation matrix.

    During training each unit is kept independently with probability
    :math:`p` and the survivors are rescaled by :math:`1/p`:

    .. math::
        \tilde x = x \odot m / p, \qquad m_j \sim \text{Bernoulli}(p).

    The :math:`1/p` is the *inverted* form, and it is what makes
    :math:`E[\tilde x] = x`, so the expected activation is unchanged and test
    time needs no adjustment at all -- ``training=False`` is exactly the
    identity. The older formulation instead scales weights by :math:`p` at
    test time; mixing the two conventions scales every activation by
    :math:`p^2` and is a standard source of a model that trains well and
    predicts badly.

    ``p`` here is the **keep** probability, following Srivastava et al. Some
    libraries take the drop probability instead, so ``p=0.8`` means "keep
    80%" and not "drop 80%".

    Parameters
    ----------
    X : array-like
        Activations ``(n, d)``.
    p : float
        Keep probability, in (0, 1].
    training : bool
        When False the input is returned unchanged.
    seed : int
        Seed for the mask.

    Returns
    -------
    RichResult
        ``output``, ``mask``, ``kept_fraction``, ``scale``.

    References
    ----------
    Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., &
        Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural
        networks from overfitting. *JMLR*, 15, 1929-1958.

    Examples
    --------
    Inverted dropout preserves the expected activation, which is why test
    time needs no rescaling.

    >>> import numpy as np
    >>> X = np.ones((4000, 10))
    >>> out = esl_dropout(X, p=0.5, seed=1)["output"]
    >>> bool(abs(out.mean() - 1.0) < 0.02)
    True

    Kept units are scaled by exactly 1/p; dropped units are zero.

    >>> r = esl_dropout(np.ones((100, 5)), p=0.5, seed=2)
    >>> sorted(set(np.round(r["output"].ravel(), 6).tolist()))
    [0.0, 2.0]

    At test time it is the identity.

    >>> bool(np.array_equal(esl_dropout(X, p=0.5, training=False)["output"], X))
    True

    >>> esl_dropout(X, p=0.0)
    Traceback (most recent call last):
        ...
    ValueError: p is the KEEP probability and must be in (0, 1]
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p is the KEEP probability and must be in (0, 1]")
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if not training or p == 1.0:
        return RichResult(
            title="Dropout (inference)",
            summary_lines=[("p (keep)", float(p)), ("training", bool(training))],
            payload={
                "output": A, "mask": np.ones_like(A),
                "kept_fraction": 1.0, "scale": 1.0, "training": bool(training),
                "method": "esl_dropout",
            },
        )
    rng = np.random.default_rng(seed)
    mask = (rng.random(A.shape) < p).astype(float)
    return RichResult(
        title="Dropout (training)",
        summary_lines=[("p (keep)", float(p)), ("kept", float(mask.mean()))],
        payload={
            "output": A * mask / p, "mask": mask,
            "kept_fraction": float(mask.mean()), "scale": 1.0 / p,
            "training": True, "p": float(p),
            "method": "esl_dropout",
        },
    )


def cheatsheet():
    return "esldat: INVERTED dropout, p is the KEEP prob; E[out]=in so inference is the identity"
