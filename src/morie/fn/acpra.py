# morie.fn -- function file (rootcoder007/morie)
"""MCMC acceptance-rate diagnostic."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["acceptance_rate_diagnostic"]


def acceptance_rate_diagnostic(chains, target=None, kind="metropolis"):
    r"""Acceptance rate, with the target appropriate to the sampler.

    Computed from consecutive draws: a repeated value is a rejection. The
    optimal rate is **sampler-specific** and the numbers are not
    interchangeable:

    ========================  ========
    random-walk Metropolis    0.234
    Metropolis-adjusted Langevin  0.574
    HMC / NUTS                0.8
    ========================  ========

    The 0.234 figure is asymptotic in dimension for a random-walk proposal on
    a product target; applying it to HMC, where 0.8 is right, produces a badly
    tuned sampler. Judging any of them without also looking at ESS is the
    deeper error: acceptance rate measures whether proposals are being taken,
    not whether the chain is *moving*. A sampler with tiny steps accepts
    almost everything and explores almost nothing.

    Parameters
    ----------
    chains : array-like
        Draws ``(m, n)`` or ``(n,)``.
    target : float, optional
        Override the sampler's conventional target.
    kind : {"metropolis", "mala", "hmc"}
        Sampler type, which sets the default target.

    Returns
    -------
    RichResult
        ``acceptance_rate``, ``target``, ``deviation``, ``recommendation``,
        ``per_chain``.

    References
    ----------
    Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and
        optimal scaling of random walk Metropolis algorithms. *Annals of
        Applied Probability*, 7(1), 110-120.

    Examples
    --------
    A chain that never repeats a value accepted everything.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> float(acceptance_rate_diagnostic(rng.normal(size=(2, 500)))["acceptance_rate"])
    1.0

    A sticky chain shows a low rate and is told to shrink its step.

    >>> x = np.repeat(rng.normal(size=(2, 100)), 5, axis=1)
    >>> r = acceptance_rate_diagnostic(x)
    >>> bool(r["acceptance_rate"] < 0.3)
    True

    The target depends on the sampler; these numbers are not interchangeable.

    >>> [float(acceptance_rate_diagnostic(rng.normal(size=(2, 100)), kind=k)["target"])
    ...  for k in ("metropolis", "mala", "hmc")]
    [0.234, 0.574, 0.8]

    >>> acceptance_rate_diagnostic(rng.normal(size=(2, 100)), kind="gibbs")
    Traceback (most recent call last):
        ...
    ValueError: kind must be one of ('metropolis', 'mala', 'hmc')
    """
    targets = {"metropolis": 0.234, "mala": 0.574, "hmc": 0.8}
    if kind not in targets:
        raise ValueError(f"kind must be one of {tuple(targets)}")
    tgt = float(targets[kind] if target is None else target)
    C = np.atleast_2d(np.asarray(chains, dtype=float))
    if C.shape[1] < 2:
        raise ValueError("need at least 2 draws per chain")
    per = np.array([float(np.mean(np.diff(C[j]) != 0)) for j in range(C.shape[0])])
    rate = float(per.mean())
    if rate < tgt * 0.5:
        rec = "acceptance is far below target: shrink the proposal step"
    elif rate > min(tgt * 2.0, 0.98):
        rec = "acceptance is far above target: the steps are too small to explore; enlarge them"
    else:
        rec = "acceptance is in a reasonable range for this sampler"
    return RichResult(
        title=f"Acceptance rate ({kind})",
        summary_lines=[("rate", rate), ("target", tgt),
                       ("chains", int(C.shape[0]))],
        warnings=["acceptance rate alone says nothing about mixing; a sampler "
                  "with tiny steps accepts almost everything and explores "
                  "almost nothing -- read it with ESS"],
        payload={
            "acceptance_rate": rate, "target": tgt,
            "deviation": float(rate - tgt), "recommendation": rec,
            "per_chain": per, "kind": kind,
            "method": "acceptance_rate_diagnostic",
        },
    )


def cheatsheet():
    return "acpra: 0.234 RWM / 0.574 MALA / 0.8 HMC -- not interchangeable; always read with ESS"
