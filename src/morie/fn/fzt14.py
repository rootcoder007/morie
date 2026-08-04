# morie.fn -- function file (rootcoder007/morie)
"""Covariance of the gamma-kernel functions at bandwidths h and 4h (Theorem 1.4)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["gkcov", "fauzi_thm1_4_asympnorm_mgkde"]


def gkcov(x, h, n, f=None, boundary=False, c=None, density=None, sample=None, upper=20.0, ngrid=2001):
    r"""Covariance of the gamma-kernel functions at bandwidths h and 4h (Theorem 1.4).

    Theorem 1.4. With :math:`R` the Stirling ratio (1.12) and
    :math:`s=\sqrt h`, the interior form (:math:`x/h\to\infty`) is

    .. math::
        \mathrm{Cov}[A_h(x), A_{4h}(x)] =
        \frac{R(s^{-1}-1)\,R((2s)^{-1}-1)\,
              (\tfrac32-2s)^{\frac3{2s}-\frac32}}
             {2\sqrt\pi\,R(\tfrac3{2s}-2)\,(3x+5s)\,
              (2-2s)^{\frac1s-\frac12}\,(1-2s)^{\frac1{2s}-\frac12}}
        \Big(\frac{x+s}{3x+5s}\Big)^{\frac1{2s}-1}
        \Big(\frac{2x+4s}{3x+5s}\Big)^{\frac1s-1}
        \frac{f_X(x)}{nh^{1/4}}\Big(1+O\big(\tfrac{h^{1/4}}n\big)\Big),

    and the boundary form (:math:`x/h\to c`) replaces :math:`3x+5s` by
    :math:`3cs+5`, :math:`x+s` by :math:`cs+1`, :math:`2x+4s` by
    :math:`2cs+4`, and :math:`h^{1/4}` by :math:`h^{3/4}`.

    The exponents are unreadable in the book's PDF text layer, which drops
    stacked fractions. They are taken instead from the primary source,
    where they are legible: Fauzi, R. R. (2020), *Bias Reduction of
    Kernel-Type Estimators without Boundary Problems*, doctoral thesis,
    Kyushu University (Kyushu University Institutional Repository,
    ``math0257``), Theorem 2.1.7 -- the same result the book reproduces as
    Theorem 1.4.

    Everything is evaluated through logarithms, because :math:`1/s` is of
    order 10 for :math:`h=0.01` and the individual factors overflow long
    before their product does. The bases :math:`2-2s` and :math:`1-2s`
    must be positive, so the formula requires :math:`h<1/4`; smaller
    bandwidths are the regime it is an asymptotic statement about anyway.

    Passing ``sample`` or ``density`` instead of ``f`` returns the EXACT
    finite-sample covariance
    :math:`n^{-1}(E[K(X;x,h)K(X;x,4h)] - J_hJ_{4h})` rather than its
    asymptotic evaluation -- useful for checking how far into the
    asymptotics a given ``h`` actually is.

    Parameters
    ----------
    x : float
        Evaluation point, ``x >= 0``.
    h : float
        Bandwidth of the first estimator; the second uses ``4h``.
    n : int
        Sample size.
    f : float, optional
        ``f_X(x)``; selects the Theorem 1.4 closed form.
    boundary : bool, default False
        Use the boundary branch; requires ``c``.
    c : float, optional
        The constant in ``x/h -> c``.
    density : callable, optional
        ``f_X`` on ``[0, upper]``; selects the exact plug-in form.
    sample : array-like, optional
        Observed data; selects the exact plug-in form.
    upper : float, default 20.0
        Upper limit of the quadrature grid.
    ngrid : int, default 2001
        Number of grid points; fixed, never adapted.

    Returns
    -------
    RichResult
        Keys ``covariance``, ``form``, ``cross``, ``jh``, ``j4h``, ``h``, ``n``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 1.4; Fauzi (2020), Kyushu University doctoral thesis, Theorem 2.1.7.
    """
    from . import _stats_core as stats
    from ._fauzi import rratio

    x = float(x)
    h = float(h)
    n = int(n)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if x < 0:
        raise ValueError("gamma kernels need x >= 0.")
    given = [f is not None, density is not None, sample is not None]
    if sum(given) != 1:
        raise ValueError("supply exactly one of f, density or sample.")

    if f is not None:
        s = np.sqrt(h)
        if h >= 0.25:
            raise ValueError(
                "Theorem 1.4 needs 1 - 2 sqrt(h) > 0, i.e. h < 1/4; "
                f"got h = {h}."
            )
        if boundary:
            if c is None:
                raise ValueError("the boundary branch of Theorem 1.4 needs c.")
            cc = float(c)
            den = 3.0 * cc * s + 5.0
            num1 = cc * s + 1.0
            num2 = 2.0 * cc * s + 4.0
            power = h ** 0.75
            form = "boundary"
        else:
            den = 3.0 * x + 5.0 * s
            num1 = x + s
            num2 = 2.0 * x + 4.0 * s
            power = h ** 0.25
            form = "interior"
        if den <= 0 or num1 <= 0 or num2 <= 0:
            raise ValueError("Theorem 1.4 needs positive bases; check x, c and h.")
        r1 = float(np.atleast_1d(rratio(1.0 / s - 1.0))[0])
        r2 = float(np.atleast_1d(rratio(1.0 / (2.0 * s) - 1.0))[0])
        r3 = float(np.atleast_1d(rratio(3.0 / (2.0 * s) - 2.0))[0])
        log_v = (
            np.log(r1)
            + np.log(r2)
            - np.log(r3)
            + (3.0 / (2.0 * s) - 1.5) * np.log(1.5 - 2.0 * s)
            - 0.5 * np.log(np.pi)
            - np.log(2.0)
            - np.log(den)
            - (1.0 / s - 0.5) * np.log(2.0 - 2.0 * s)
            - (1.0 / (2.0 * s) - 0.5) * np.log(1.0 - 2.0 * s)
            + (1.0 / (2.0 * s) - 1.0) * np.log(num1 / den)
            + (1.0 / s - 1.0) * np.log(num2 / den)
        )
        cov = float(np.exp(log_v)) * float(f) / (n * power)
        return RichResult(
            payload={
                "covariance": cov,
                "form": form,
                "cross": float("nan"),
                "jh": float("nan"),
                "j4h": float("nan"),
                "h": h,
                "n": n,
                "method": "Cov[A_h, A_4h], Theorem 1.4 closed form",
            }
        )

    s1 = 1.0 / np.sqrt(h)
    b1 = x * np.sqrt(h) + h
    s2 = 1.0 / np.sqrt(4.0 * h)
    b2 = x * np.sqrt(4.0 * h) + 4.0 * h
    if sample is not None:
        w = np.asarray(sample, dtype=float).ravel()
        if np.any(w < 0):
            raise ValueError("gamma kernels need data on [0, infinity).")
        k1 = stats.gamma.pdf(w, a=s1, scale=b1)
        k2 = stats.gamma.pdf(w, a=s2, scale=b2)
        jh = float(np.mean(k1))
        j4h = float(np.mean(k2))
        cross = float(np.mean(k1 * k2))
    else:
        grid = np.linspace(0.0, float(upper), int(ngrid))
        fv = np.asarray([float(density(float(g))) for g in grid], dtype=float)
        k1 = stats.gamma.pdf(grid, a=s1, scale=b1)
        k2 = stats.gamma.pdf(grid, a=s2, scale=b2)
        jh = float(np.trapezoid(k1 * fv, grid))
        j4h = float(np.trapezoid(k2 * fv, grid))
        cross = float(np.trapezoid(k1 * k2 * fv, grid))
    return RichResult(
        payload={
            "covariance": float((cross - jh * j4h) / n),
            "form": "plugin",
            "cross": cross,
            "jh": jh,
            "j4h": j4h,
            "h": h,
            "n": n,
            "method": "Cov[A_h, A_4h] from its definition (exact)",
        }
    )


fauzi_thm1_4_asympnorm_mgkde = gkcov


def cheatsheet():
    return "fzt14: Cov[A_h, A_4h] closed form (Thm 1.4), exponents from Fauzi's Kyushu thesis Thm 2.1.7"


# CANONICAL TEST
# >>> r = gkcov(x=1.0, h=0.01, n=100, f=0.3)
# >>> r['form'] == 'interior' and r['covariance'] > 0
# True
