"""Effects of nugget, sill and range on kriging prediction."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_krig import simple_kriging

__all__ = ["schabenberger_nugget_sill_range_effect"]


def schabenberger_nugget_sill_range_effect(nugget=0.0, sill=1.0, range=1.0,
                                           target_dist=None, model="exponential"):
    r"""
    How the nugget, sill and range change a kriging prediction.

    The book works through what each parameter does (Sec. 5.2.3). This
    evaluates that on a fixed one-dimensional layout so the effects are
    exhibited rather than described:

    * the NUGGET drives the weights toward equality and pulls the
      prediction toward the mean, because a larger share of the variance
      is uncorrelated. With a pure nugget the prediction IS the mean and
      the smoothing is total.
    * the SILL scales the kriging variance but leaves the weights, and
      therefore the prediction, unchanged -- it is a pure variance factor.
    * the RANGE controls how far influence extends; a short range makes
      the prediction local.

    Parameters
    ----------
    nugget, sill, range : float
        Covariance parameters.
    target_dist : float, optional
        Distance from the prediction location to the nearest datum.
        Defaults to 0.5.
    model : {'exponential', 'gaussian', 'spherical'}
        Correlogram family.

    Returns
    -------
    RichResult
        ``prediction``, ``variance``, ``weights``, ``weight_spread``
        (max minus min weight; 0 means the nugget has flattened them),
        and the echoed parameters.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 5.2.3, p. 228.
    """
    if nugget < 0 or sill < 0:
        raise ValueError("`nugget` and `sill` must be >= 0")
    if range <= 0:
        raise ValueError("`range` must be > 0")
    d = 0.5 if target_dist is None else float(target_dist)
    coords = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    z = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    cm = {"nugget": nugget, "sill": sill, "range": range, "model": model}
    p, v, lam = simple_kriging(coords, z, np.array([[d]]), cm)
    w = lam[:, 0]
    return RichResult(
        title="Effect of nugget, sill and range on kriging",
        summary_lines=[("nugget", nugget), ("sill", sill), ("range", range),
                       ("prediction", float(p[0]))],
        payload={"prediction": float(p[0]), "variance": float(v[0]),
                 "weights": w, "weight_spread": float(w.max() - w.min()),
                 "mean": float(np.mean(z)),
                 "nugget": float(nugget), "sill": float(sill),
                 "range": float(range), "model": model},
    )


def cheatsheet():
    return "spnsr: how nugget/sill/range move a kriging prediction."
