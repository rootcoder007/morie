"""Impervious-surface indices NDBI, MNDWI and NDISI from multispectral bands.

Zha, Y., Gao, J. and Ni, S. (2003). Use of normalized difference built-up index in
automatically mapping urban areas from TM imagery. International Journal of Remote Sensing 24,
583-594. Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance
open water features in remotely sensed imagery. International Journal of Remote Sensing 27,
3025-3033. Xu, H. (2010). Analysis of impervious surface and its impact on urban heat
environment using the normalized difference impervious surface index (NDISI).
Photogrammetric Engineering and Remote Sensing 76, 557-565.
"""

from ._richresult import RichResult

__all__ = ["impervious_indices"]


def impervious_indices(green, nir, swir1, tir=None, threshold=0.0):
    r"""Per-pixel NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) and MNDWI = (G - SWIR1) / (G + SWIR1);
    with a thermal band, NDISI = (TIR - (MNDWI + NIR + SWIR1)/3) / (TIR + (MNDWI + NIR + SWIR1)/3)
    (Xu 2010, reflectances and a thermal band rescaled to 0-1). Pixels whose NDISI (or NDBI
    without a thermal band) exceeds ``threshold`` are mapped impervious.

    Parameters
    ----------
    green, nir, swir1 : sequences (flattened rasters) of reflectance
    tir : sequence, optional
        Thermal band rescaled to [0, 1].
    threshold : float

    Returns
    -------
    RichResult
        Keys: ndbi, mndwi, ndisi (with ``tir``), impervious (bool), impervious_share.

    References
    ----------
    Xu, H. (2010). Photogrammetric Engineering and Remote Sensing 76, 557-565.
    Zha, Y., Gao, J. and Ni, S. (2003). International Journal of Remote Sensing 24, 583-594.

    Examples
    --------
    >>> round(impervious_indices([0.1], [0.2], [0.3])["ndbi"][0], 12)
    0.2
    """
    G = [float(v) for v in green]
    N = [float(v) for v in nir]
    S = [float(v) for v in swir1]

    def nd(a, b):
        return (a - b) / (a + b) if a + b != 0 else 0.0

    ndbi = [nd(s, n) for s, n in zip(S, N)]
    mndwi = [nd(g, s) for g, s in zip(G, S)]
    out = {"ndbi": ndbi, "mndwi": mndwi}
    score = ndbi
    if tir is not None:
        T = [float(v) for v in tir]
        vis = [(m + n + s) / 3 for m, n, s in zip(mndwi, N, S)]
        out["ndisi"] = [nd(t, v) for t, v in zip(T, vis)]
        score = out["ndisi"]
    imp = [v > threshold for v in score]
    out["impervious"] = imp
    out["impervious_share"] = sum(imp) / len(imp)
    return RichResult(
        title="Impervious surface indices", summary_lines=[("impervious share", out["impervious_share"])], payload=out
    )


def cheatsheet():
    return "imperv: NDBI, MNDWI and NDISI impervious-surface indices"
