# morie.fn -- shared helpers (rootcoder007/morie)
"""Quadrature and sample-statistic helpers shared by the Rangayyan shelf.

Kept in one place so that, for example, the four PDF-moment modules
(mean, MS, variance, skewness, kurtosis) integrate by exactly the same
rule rather than five slightly different ones.
"""

from math import fsum, inf, isfinite

from . import _sci_core as _sc

__all__: list = []


def aslist(x):
    """Coerce to a plain list of floats without going through an array type."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    try:
        return [float(v) for v in x]
    except TypeError:
        # Only fall back for a genuine scalar.  Falling back for ANY
        # TypeError also caught "iterable whose elements are not real",
        # and then float(x) on the whole sequence reported the container
        # as the offender instead of the element.
        try:
            iter(x)
        except TypeError:
            return [float(x)]
        raise


def aslistc(x):
    """Coerce to a plain list, PRESERVING complex values.

    aslist() forces float and so cannot carry a spectrum.  Anything doing
    complex-logarithm or conjugate arithmetic -- the complex cepstrum of
    eqs (4.63) and (4.68), for instance -- needs this instead.
    """
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    try:
        return [v if isinstance(v, complex) else complex(float(v))
                for v in x]
    except TypeError:
        try:
            iter(x)
        except TypeError:
            return [x if isinstance(x, complex) else complex(float(x))]
        raise


def gridint(y, x=None):
    """Integrate tabulated y over x.

    Composite Simpson when the panel count is even and the grid is
    uniform (its error is O(h^4)); the trapezoidal rule otherwise, which
    is what a non-uniform grid admits without interpolation.
    """
    y = aslist(y)
    n = len(y)
    if n < 2:
        raise ValueError("need at least two grid points")
    if x is None:
        x = [float(i) for i in range(n)]
    else:
        x = aslist(x)
    if len(x) != n:
        raise ValueError("x and y must have the same length")
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    if any(d <= 0 for d in h):
        raise ValueError("x must be strictly increasing")
    uniform = all(abs(d - h[0]) <= 1e-12 * max(1.0, abs(h[0])) for d in h)
    if uniform and (n - 1) % 2 == 0:
        s = [y[0], y[-1]]
        s += [4.0 * y[i] for i in range(1, n - 1, 2)]
        s += [2.0 * y[i] for i in range(2, n - 1, 2)]
        return fsum(s) * h[0] / 3.0
    return fsum(0.5 * (y[i] + y[i + 1]) * h[i] for i in range(n - 1))


def pdfint(f, pdf=None, x=None, lower=-inf, upper=inf):
    """Integrate f(eta) * p(eta) d eta.

    Two spellings of the same integral, because a PDF reaches us either
    in closed form (callable, integrated adaptively) or tabulated on a
    grid (integrated by gridint).  f is applied to the abscissa.
    """
    if x is not None:
        xs = aslist(x)
        if callable(pdf):
            ps = [float(pdf(v)) for v in xs]
        else:
            ps = aslist(pdf)
        if len(ps) != len(xs):
            raise ValueError("pdf and x must have the same length")
        return gridint([f(v) * p for v, p in zip(xs, ps)], xs)
    if not callable(pdf):
        raise ValueError("give either a grid (x=) or a callable pdf")
    if not (isfinite(lower) and isfinite(upper)):
        lo = -40.0 if not isfinite(lower) else lower
        hi = 40.0 if not isfinite(upper) else upper
    else:
        lo, hi = float(lower), float(upper)
    return float(_sc.quad(lambda v: f(v) * float(pdf(v)), lo, hi)[0])


def checkpdf(mass, tol=1e-6):
    """Report how far a density integrates from unit mass."""
    return {"pdf_mass": float(mass), "pdf_mass_ok": abs(mass - 1.0) <= tol}
