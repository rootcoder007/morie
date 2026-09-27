"""Surface urban heat island (SUHI) intensity from land surface temperature.

Peng, S. et al. (2012). Surface urban heat island across 419 global big cities.
Environmental Science and Technology 46, 696-703. Voogt, J. A. and Oke, T. R. (2003).
Thermal remote sensing of urban climates. Remote Sensing of Environment 86, 370-384.
"""

from ._richresult import RichResult

__all__ = ["uhi_intensity"]


def uhi_intensity(lst, urban, rural=None):
    r"""SUHI intensity = mean urban LST - mean rural LST (Peng et al. 2012), and the per-pixel
    anomaly LST - mean rural LST mapping the heat island; rural pixels default to every
    non-urban pixel.

    Parameters
    ----------
    lst : sequence (flattened raster) of land surface temperature
    urban : sequence of bool
    rural : sequence of bool, optional
        Reference rural pixels (e.g. an exurban buffer excluding water).

    Returns
    -------
    RichResult
        Keys: intensity, urban_mean, rural_mean, anomaly.

    References
    ----------
    Peng, S. et al. (2012). Environmental Science and Technology 46, 696-703.
    Voogt, J. A. and Oke, T. R. (2003). Remote Sensing of Environment 86, 370-384.

    Examples
    --------
    >>> uhi_intensity([30, 32, 26, 24], [True, True, False, False])["intensity"]
    6.0
    """
    T = [float(v) for v in lst]
    U = [bool(v) for v in urban]
    R = [not u for u in U] if rural is None else [bool(v) for v in rural]
    if not any(U) or not any(R):
        raise ValueError("need at least one urban and one rural pixel")

    def mean(mask):
        s, c = 0.0, 0
        for t, m in zip(T, mask):
            if m:
                s += t
                c += 1
        return s / c

    um, rm = mean(U), mean(R)
    return RichResult(
        title="Surface urban heat island intensity",
        summary_lines=[("intensity", um - rm)],
        payload={"intensity": um - rm, "urban_mean": um, "rural_mean": rm, "anomaly": [t - rm for t in T]},
    )


def cheatsheet():
    return "uhiint: surface urban heat island intensity (urban minus rural LST)"
