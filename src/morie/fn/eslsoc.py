# morie.fn -- function file (rootcoder007/morie)
"""Self-organizing map -- Kohonen (1990), ESL Sec 14.4."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_self_organize"]


def esl_self_organize(X, grid=(5, 5), eta=0.5, n_epochs=50, sigma0=None, seed=0):
    r"""Fit a Kohonen self-organizing map.

    Prototypes live on a fixed rectangular lattice. For each observation the
    closest prototype wins, and it *and its lattice neighbours* move toward
    the observation:

    .. math::
        m_j \leftarrow m_j + \eta(t)\, h_{j,j^*}(t)\,(x_i - m_j), \qquad
        h_{j,j^*} = \exp\!\left(-\frac{\lVert \ell_j - \ell_{j^*}\rVert^2}
                                      {2\sigma(t)^2}\right),

    where :math:`\ell_j` is the *lattice* coordinate, not the data-space one.
    Dragging the neighbours along is what separates a SOM from k-means: it
    forces prototypes adjacent on the grid to end up adjacent in data space,
    which is what makes the fitted map usable as a 2-D display.

    Both :math:`\eta` and :math:`\sigma` decay across epochs. ESL notes that
    with :math:`\sigma \to 0` the neighbourhood collapses to the winner alone
    and the SOM degenerates into online k-means -- so a "SOM" that shows no
    topological ordering usually has its neighbourhood decaying too fast.

    Parameters
    ----------
    X : array-like
        Data ``(n, p)``.
    grid : tuple of int
        Lattice shape ``(rows, cols)``.
    eta : float
        Initial learning rate, in (0, 1].
    n_epochs : int
        Passes over the data.
    sigma0 : float, optional
        Initial neighbourhood width. Defaults to ``max(grid)/2``.
    seed : int
        Seed for initialisation and the presentation order.

    Returns
    -------
    RichResult
        ``prototypes`` ``(rows*cols, p)``, ``lattice`` coordinates,
        ``assignment``, ``quantization_error``, ``topographic_error``,
        ``counts``.

    References
    ----------
    Kohonen, T. (1990). The self-organizing map. *Proceedings of the IEEE*,
        78(9), 1464-1480.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Prototypes adjacent on the lattice end up close in data space -- the
    topology-preserving property, measured by a low topographic error.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0, 1, (600, 2))
    >>> r = esl_self_organize(X, grid=(6, 6), seed=1)
    >>> bool(r["topographic_error"] < 0.15)
    True

    Quantization error beats using the global mean as a single prototype.

    >>> bool(r["quantization_error"] < np.mean(np.linalg.norm(X - X.mean(0), axis=1)))
    True

    >>> esl_self_organize(X, eta=0.0)
    Traceback (most recent call last):
        ...
    ValueError: eta must be in (0, 1]
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError("eta must be in (0, 1]")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    rows, cols = int(grid[0]), int(grid[1])
    if rows < 1 or cols < 1:
        raise ValueError("grid dimensions must be positive")
    K = rows * cols
    if K > n:
        raise ValueError(f"grid has {K} nodes but there are only {n} observations")

    lattice = np.array([(i, j) for i in range(rows) for j in range(cols)], dtype=float)
    Dlat = ((lattice[:, None] - lattice[None]) ** 2).sum(-1)
    sigma0 = max(rows, cols) / 2.0 if sigma0 is None else float(sigma0)

    rng = np.random.default_rng(seed)
    M = X[rng.choice(n, K, replace=False)].astype(float)

    for ep in range(n_epochs):
        frac = ep / max(n_epochs - 1, 1)
        lr = eta * (0.01 / eta) ** frac
        sig = max(sigma0 * (0.5 / sigma0) ** frac, 1e-3)
        for i in rng.permutation(n):
            win = int(np.argmin(((M - X[i]) ** 2).sum(1)))
            h = np.exp(-Dlat[win] / (2 * sig**2))
            M += lr * h[:, None] * (X[i] - M)

    d2 = ((X[:, None] - M[None]) ** 2).sum(-1)
    order = np.argsort(d2, axis=1)
    assign = order[:, 0]
    qe = float(np.mean(np.sqrt(d2[np.arange(n), assign])))
    # Topographic error: fraction of points whose two closest prototypes are
    # not lattice neighbours -- the standard measure of a broken map.
    te = float(np.mean(Dlat[order[:, 0], order[:, 1]] > 2.0))
    return RichResult(
        title="Self-organizing map",
        summary_lines=[("n", n), ("grid", f"{rows}x{cols}"),
                       ("quantization error", qe), ("topographic error", te)],
        payload={
            "prototypes": M, "lattice": lattice, "assignment": assign,
            "quantization_error": qe, "topographic_error": te,
            "counts": np.bincount(assign, minlength=K),
            "grid": (rows, cols), "n_epochs": int(n_epochs),
            "method": "esl_self_organize",
        },
    )


def cheatsheet():
    return "eslsoc: SOM drags LATTICE neighbours too; if sigma decays too fast it degenerates into k-means"
