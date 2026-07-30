# morie.fn -- function file (rootcoder007/morie)
"""Croston for intermittent demand, alternative front-end."""

from __future__ import annotations

from ._richresult import RichResult
from .croston import croston

__all__ = ["joseph_croston_intermittent"]


def joseph_croston_intermittent(y, alpha=0.1, variant="sba"):
    r"""Intermittent-demand forecast, defaulting to the bias-corrected variant.

    Same estimator as :func:`~morie.fn.croston.croston` but with
    ``variant="sba"`` as the default, since the Syntetos-Boylan correction is
    the standard recommendation for inventory use and the uncorrected version
    systematically overstocks.

    Also classifies the series on the Syntetos-Boylan-Croston grid, which
    decides whether Croston is the right tool at all: it is appropriate for
    *intermittent* and *lumpy* demand, and unnecessary for smooth or
    erratic-but-frequent demand, where ordinary smoothing does better.

    Parameters
    ----------
    y : array-like
        Demand series.
    alpha : float
        Smoothing parameter.
    variant : {"sba", "croston"}
        Defaults to the corrected estimator.

    Returns
    -------
    RichResult
        ``forecast``, ``classification``, ``cv_squared``,
        ``average_interval``, plus the fields of
        :func:`~morie.fn.croston.croston`.

    References
    ----------
    Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). On the
        categorization of demand patterns. *JORS*, 56(5), 495-503.

    Examples
    --------
    Regular intermittent demand is classified as intermittent.

    >>> y = [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]
    >>> str(joseph_croston_intermittent(y)["classification"])
    'intermittent'

    Highly variable intermittent demand is lumpy, where forecasts of any kind
    are least reliable.

    >>> lumpy = [0, 0, 1, 0, 0, 0, 50, 0, 0, 2, 0, 0, 0, 80, 0, 0]
    >>> str(joseph_croston_intermittent(lumpy)["classification"])
    'lumpy'

    The default is bias-corrected, so it forecasts below the plain estimator.

    >>> from morie.fn.croston import croston
    >>> a = joseph_croston_intermittent(y)["forecast"]
    >>> b = croston(y, variant="croston")["forecast"]
    >>> bool(a < b)
    True
    """
    import numpy as np

    r = croston(y, alpha=alpha, variant=variant)
    v = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    nz = v[v > 0]
    cv2 = float((np.std(nz, ddof=1) / np.mean(nz)) ** 2) if nz.size > 1 else 0.0
    p = float(r["interval"])
    # Syntetos-Boylan-Croston cutoffs.
    if p >= 1.32 and cv2 >= 0.49:
        cls = "lumpy"
    elif p >= 1.32:
        cls = "intermittent"
    elif cv2 >= 0.49:
        cls = "erratic"
    else:
        cls = "smooth"
    return RichResult(
        title=f"Intermittent demand ({variant})",
        summary_lines=[("forecast", float(r["forecast"])),
                       ("classification", cls), ("CV^2", cv2)],
        warnings=(list(r.warnings)
                  + (["demand is smooth; ordinary exponential smoothing is "
                      "more appropriate than Croston here"] if cls == "smooth" else [])),
        payload={
            "forecast": r["forecast"], "rate": r["rate"],
            "demand_size": r["demand_size"], "interval": r["interval"],
            "bias_factor": r["bias_factor"], "n_nonzero": r["n_nonzero"],
            "intermittency": r["intermittency"],
            "classification": cls, "cv_squared": cv2,
            "average_interval": p, "alpha": r["alpha"], "variant": variant,
            "method": "joseph_croston_intermittent",
        },
    )


def cheatsheet():
    return "jocros: SBA by default; classifies smooth/erratic/intermittent/lumpy so you know if Croston applies"
