"""morie.fast — the opt-in JIT acceleration surface.

Public re-export of the kernels in :mod:`morie._jit`.  ``import morie.fast``
gives a stable, user-facing namespace whose functions are
numerically identical to their scipy / numpy equivalents but compile
to native code when ``numba`` is installed.

Usage
-----

    >>> import morie.fast as mf
    >>> mf.is_jit_available()
    True            # after `pip install morie[fast]`
    False           # otherwise (kernels still work, just no JIT)
    >>> x = np.random.normal(size=10_000)
    >>> mf.normal_pdf(x, 0.0, 1.0)
    array([...])

Functions
---------

``normal_pdf(x, mean, sd)``
    Drop-in for ``scipy.stats.norm.pdf``.  ~5-10× faster on small
    arrays once JIT-compiled.

``normal_logpdf(x, mean, sd)``
    Drop-in for ``scipy.stats.norm.logpdf``.

``mean_jit(arr)`` / ``std_jit(arr, ddof=1)`` / ``var_jit(arr, ddof=1)``
    Drop-ins for ``arr.mean()`` / ``np.std(arr, ddof=...)`` /
    ``np.var(arr, ddof=...)``.

``cor_pearson_jit(x, y)``
    Drop-in for ``scipy.stats.pearsonr(x, y)[0]`` (point estimate only).

``euclid_dist_jit(a, b)``
    Drop-in for ``np.linalg.norm(a - b)``.

``is_jit_available()``
    Returns ``True`` iff the compiled C++ core is loaded and
    ``MORIE_DISABLE_JIT`` is not set.

``jit_if_available(*args, **kwargs)``
    Identity decorator factory (kept for API stability; the
    compiled core does the accelerating). For users decorating their
    own kernels::

        from morie.fast import jit_if_available

        @jit_if_available(cache=True, nogil=True)
        def my_hot_loop(arr):
            ...

"""

from __future__ import annotations

from ._jit import (
    bootstrap_mean_jit,
    cor_pearson_jit,
    euclid_dist_jit,
    is_jit_available,
    mean_jit,
    normal_logpdf,
    normal_pdf,
    std_jit,
    trimmed_ipw_weights_jit,
    var_jit,
)


def jit_if_available(*args, **kwargs):
    """Conditional ``numba.njit`` decorator.

    Returns ``numba.njit(*args, **kwargs)`` when numba is importable
    and ``MORIE_DISABLE_JIT`` is unset; returns an identity decorator
    otherwise.

    The signature mirrors ``numba.njit`` so existing JIT-decorated
    code can swap to this drop-in without other changes.

    Use either form::

        @jit_if_available
        def f(x): ...

        @jit_if_available(cache=True, nogil=True)
        def f(x): ...
    """
    # The compiled C++ core (morie._core) is the acceleration layer;
    # the numba path was removed with the external-dependency purge.
    # Both decorator forms stay valid as identities.
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def _passthrough(fn):
        return fn

    return _passthrough


__all__ = [
    "normal_pdf",
    "normal_logpdf",
    "mean_jit",
    "std_jit",
    "var_jit",
    "cor_pearson_jit",
    "euclid_dist_jit",
    "bootstrap_mean_jit",
    "trimmed_ipw_weights_jit",
    "is_jit_available",
    "jit_if_available",
]
