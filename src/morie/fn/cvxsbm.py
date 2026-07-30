# morie.fn -- function file (rootcoder007/morie)
"""Subgradient method -- Boyd & Vandenberghe Sec. 9.4 / EE364b."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["boyd_subgrad_method"]


def boyd_subgrad_method(f, subgrad, x0, t=None, max_iter=500, rule="sqrt"):
    r"""Iterate :math:`x^{k+1} = x^k - t_k g^k`, :math:`g^k \in \partial
    f(x^k)`.

    NOT a descent method, and that is the fact to internalise: a
    subgradient step can increase f, so the iterate you finish on need not
    be the best one seen. The method therefore tracks the running BEST
    value, which is what the convergence theory is about.

    Because there is no descent guarantee, the step size cannot be chosen
    by line search. Diminishing but non-summable steps --
    :math:`t_k = a/\sqrt k`, with :math:`\sum t_k = \infty` and
    :math:`\sum t_k^2 < \infty` -- are what buy convergence, at a rate of
    :math:`O(1/\sqrt k)` that is genuinely slow and cannot be improved
    for general nondifferentiable f.

    Parameters
    ----------
    f : callable
        Objective.
    subgrad : callable
        Returns any element of the subdifferential.
    x0 : array-like
        Start.
    t : float, optional
        Step scale; the rule multiplies it.
    max_iter : int
        Iterations.
    rule : {"sqrt", "constant", "inverse"}
        Step schedule.

    Returns
    -------
    RichResult
        ``x_best``, ``f_best``, ``x_last``, ``f_last``, ``increased``
        (steps that made f worse), ``f_path``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    Minimising :math:`|x|`, which has no gradient at its minimum.

    >>> import numpy as np
    >>> r = boyd_subgrad_method(lambda z: float(abs(z[0])),
    ...                         lambda z: np.sign(z), [3.0], t=0.5)
    >>> bool(r["f_best"] < 0.05)
    True

    The method is NOT monotone: some steps increase f, which is why the
    best-so-far iterate is what is returned.

    >>> bool(r["increased"] > 0)
    True

    The last iterate is not the best one, so returning it would be wrong.

    >>> bool(r["f_last"] >= r["f_best"])
    True
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).ravel().copy()
    scale = 1.0 if t is None else float(t)
    fx = float(f(x))
    best_x, best_f = x.copy(), fx
    path = [fx]
    inc = 0
    for k in range(1, int(max_iter) + 1):
        g = np.atleast_1d(np.asarray(subgrad(x), dtype=float)).ravel()
        if rule == "sqrt":
            step = scale / np.sqrt(k)
        elif rule == "inverse":
            step = scale / k
        elif rule == "constant":
            step = scale
        else:
            raise ValueError('rule must be "sqrt", "constant" or "inverse"')
        x = x - step * g
        fx = float(f(x))
        path.append(fx)
        if fx > path[-2] + 1e-15:
            inc += 1
        if fx < best_f:
            best_f, best_x = fx, x.copy()
    return RichResult(
        title="Subgradient method",
        summary_lines=[("iterations", int(max_iter)), ("f best", best_f),
                       ("f last", fx), ("steps that increased f", inc)],
        warnings=["the subgradient method is not a descent method; use "
                  "x_best, not x_last"],
        payload={
            "x_best": best_x, "f_best": best_f, "x_last": x,
            "f_last": fx, "increased": inc, "f_path": np.asarray(path),
            "rule": rule, "method": "boyd_subgrad_method",
        },
    )


def cheatsheet():
    return "cvxsbm: NOT a descent method -- track the best iterate; O(1/sqrt k) and that is optimal"
