# morie.fn -- function file (rootcoder007/morie)
"""Independent component analysis (FastICA) -- ESL Sec 14.7."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_ica"]


def esl_ica(X, k=None, fun="logcosh", max_iter=500, tol=1e-8, seed=0):
    r"""Recover independent sources from mixtures by FastICA.

    Given :math:`X = SA^\top`, ICA recovers :math:`S` by finding a
    projection that is maximally *non-Gaussian*, since by the central limit
    theorem a mixture of independent sources is closer to Gaussian than any
    source is. FastICA maximises the negentropy surrogate
    :math:`J(w) \approx [E\,G(w^\top z) - E\,G(\nu)]^2` with the fixed-point
    update

    .. math::
        w^+ = E\!\left[z\, g(w^\top z)\right] - E\!\left[g'(w^\top z)\right] w,

    followed by deflationary orthogonalisation.

    Two indeterminacies are intrinsic and not defects: the **scale** of each
    source (absorbed into the mixing matrix, so components are returned with
    unit variance) and the **order** of the components. Any comparison to
    known sources must therefore be up to permutation and sign, which is how
    the doctest checks it.

    ICA cannot separate Gaussian sources at all -- a rotation of independent
    Gaussians is again independent Gaussian, so there is nothing to find.

    Parameters
    ----------
    X : array-like
        Observed mixtures ``(n, p)``.
    k : int, optional
        Number of components. Defaults to ``p``.
    fun : {"logcosh", "exp", "cube"}
        Non-quadratic contrast :math:`G`.
    max_iter, tol
        Fixed-point iteration controls.
    seed : int
        Seed for the random initialisation.

    Returns
    -------
    RichResult
        ``sources`` ``(n, k)`` with unit variance, ``unmixing`` ``(k, p)``,
        ``mixing`` ``(p, k)``, ``whitening``, ``mean``, ``n_iter``,
        ``converged``.

    References
    ----------
    Hyvarinen, A., & Oja, E. (2000). Independent component analysis:
        Algorithms and applications. *Neural Networks*, 13(4-5), 411-430.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    Two non-Gaussian sources, linearly mixed, are recovered up to
    permutation and sign.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0, 8 * np.pi, 2000)
    >>> S = np.column_stack([np.sin(t), np.sign(np.cos(2.7 * t))])
    >>> A = np.array([[1.0, 0.7], [-0.6, 1.2]])
    >>> r = esl_ica(S @ A.T, k=2, seed=1)
    >>> C = np.abs(np.corrcoef(r["sources"].T, S.T)[:2, 2:])
    >>> bool(C.max(axis=1).min() > 0.95)          # each source matched by some component
    True

    Components come back standardised, since scale is not identifiable.

    >>> bool(np.allclose(r["sources"].std(axis=0, ddof=0), 1.0, atol=1e-6))
    True

    >>> esl_ica(S @ A.T, k=5)
    Traceback (most recent call last):
        ...
    ValueError: k must be between 1 and p=2
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    k = p if k is None else int(k)
    if not 1 <= k <= p:
        raise ValueError(f"k must be between 1 and p={p}")
    if n < 2:
        raise ValueError("need at least 2 observations")

    mean = X.mean(axis=0)
    Xc = X - mean
    # Whiten: unit covariance is what makes the fixed point orthogonal.
    cov = np.cov(Xc, rowvar=False).reshape(p, p)
    d, E = np.linalg.eigh(cov)
    order = np.argsort(d)[::-1][:k]
    d, E = np.clip(d[order], 1e-12, None), E[:, order]
    K = (E / np.sqrt(d)).T
    Z = Xc @ K.T

    if fun == "logcosh":
        g, gp = lambda u: np.tanh(u), lambda u: 1.0 - np.tanh(u) ** 2  # noqa: E731
    elif fun == "exp":
        g = lambda u: u * np.exp(-(u**2) / 2)                          # noqa: E731
        gp = lambda u: (1 - u**2) * np.exp(-(u**2) / 2)                # noqa: E731
    elif fun == "cube":
        g, gp = lambda u: u**3, lambda u: 3 * u**2                     # noqa: E731
    else:
        raise ValueError('fun must be "logcosh", "exp" or "cube"')

    rng = np.random.default_rng(seed)
    W = np.zeros((k, k))
    iters, converged = [], True
    for j in range(k):
        w = rng.normal(size=k)
        w /= np.linalg.norm(w)
        it = 0
        for it in range(1, max_iter + 1):
            wx = Z @ w
            new = (Z * g(wx)[:, None]).mean(axis=0) - gp(wx).mean() * w
            new -= W[:j].T @ (W[:j] @ new)          # deflation
            nrm = np.linalg.norm(new)
            if nrm < 1e-12:
                break
            new /= nrm
            if abs(abs(float(new @ w)) - 1.0) < tol:
                w = new
                break
            w = new
        else:
            converged = False
        W[j] = w
        iters.append(it)

    S = Z @ W.T
    sd = S.std(axis=0, ddof=0)
    S = S / np.where(sd > 0, sd, 1.0)
    unmix = (W / np.where(sd > 0, sd, 1.0)[:, None]) @ K
    return RichResult(
        title="FastICA",
        summary_lines=[("n", n), ("p", p), ("k", k), ("contrast", fun)],
        warnings=[] if converged else [f"some components did not converge in {max_iter} iterations"],
        payload={
            "sources": S, "unmixing": unmix,
            "mixing": np.linalg.pinv(unmix), "whitening": K, "mean": mean,
            "n_iter": np.array(iters), "converged": bool(converged), "fun": fun,
            "method": "esl_ica",
        },
    )


def cheatsheet():
    return "eslica: FastICA; sign and order of components are NOT identifiable -- compare up to permutation"
