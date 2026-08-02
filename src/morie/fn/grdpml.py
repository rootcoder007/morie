# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DDPM simplified training loss."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_ddpm_simple_loss"]

_METHOD = "DDPM simplified noise-prediction loss"


def geron_ddpm_simple_loss(eps, eps_pred, reduction="mean"):
    r"""Mean squared error on the *noise*, not the image.

    .. math::
        L_{\text{simple}} = \mathbb E_{t, x_0, \varepsilon}
        \bigl[\,\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\,\bigr]

    Predicting the noise instead of the clean image is the
    reparameterisation that makes DDPM work in practice: it drops the
    per-timestep weighting of the true variational bound (hence
    "simplified"), which happens to down-weight the very noisy steps
    where the bound is dominated by terms nothing can learn.

    ``per_sample`` is the summed squared error per instance -- the
    quantity the expectation is over -- while ``loss`` averages over
    everything, the convention an optimizer expects.

    Parameters
    ----------
    eps : array-like, shape (d,) or (m, d)
        The noise actually added by
        :func:`morie.fn.grdpmf.geron_ddpm_forward_process`.
    eps_pred : array-like, same shape
        The network's prediction.
    reduction : {"mean", "sum"}, optional
        What ``estimate`` reports.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``sum_squared_error``, ``per_sample``,
        ``residual``, ``rmse``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 18, DDPM training section (Ho et al. 2020).

    Examples
    --------
    One component off by 1 out of two components averages to 0.5:

    >>> r = geron_ddpm_simple_loss([1.0, 0.0], [0.0, 0.0])
    >>> r["loss"]
    0.5
    >>> r["sum_squared_error"]
    1.0

    A perfect prediction costs nothing, and the RMSE agrees:

    >>> r2 = geron_ddpm_simple_loss([0.3, -0.7], [0.3, -0.7])
    >>> r2["loss"], r2["rmse"]
    (0.0, 0.0)
    """
    E = np.atleast_2d(np.asarray(eps, dtype=float))
    P = np.atleast_2d(np.asarray(eps_pred, dtype=float))
    if E.shape != P.shape:
        raise ValueError(f"eps {E.shape} and eps_pred {P.shape} must have the same shape.")
    if E.size == 0:
        raise ValueError("eps is empty.")
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(P)):
        raise ValueError("eps and eps_pred must be finite.")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}.")

    res = E - P
    sq = res**2
    per = sq.sum(axis=1)
    loss = float(sq.mean())
    total = float(sq.sum())

    return RichResult(
        title="DDPM simple loss",
        summary_lines=[("Loss (mean)", loss), ("Sum sq error", total)],
        payload={
            "loss": loss,
            "sum_squared_error": total,
            "per_sample": per.tolist(),
            "residual": res.tolist(),
            "rmse": float(np.sqrt(loss)),
            "estimate": loss if reduction == "mean" else total,
            "n": int(E.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdpml: L_simple = ||eps - eps_theta||^2 -- predict the noise, drop the bound's weighting"
