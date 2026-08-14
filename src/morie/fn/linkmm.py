"""IRT moment linking: mean/mean and mean/sigma (Marco 1977; Loyd & Hoover 1980)."""

import math

from ._richresult import RichResult

__all__ = ["linkmm", "irt_linking_mean_mean"]


def linkmm(a_from, b_from, a_to, b_to, method="mean/mean"):
    """
    Moment methods for IRT scale linking.

    Estimates the linear transformation theta_T = A theta_F + B
    (plink Eq. 7; Weeks 2010) from the common items' slope/difficulty
    parameters.  The mean/mean method (Loyd & Hoover 1980; plink
    Eqs. 13a-13b) uses

        A = mu(a_F) / mu(a_T),    B = mu(b_T) - A mu(b_F),

    and the mean/sigma method (Marco 1977; plink Eqs. 12a-12b) uses

        A = sigma(b_T) / sigma(b_F),    B = mu(b_T) - A mu(b_F).

    Under the transformation the from-scale item parameters map to
    a* = a_F / A, b* = A b_F + B (plink Eqs. 8a-8b); both maps are
    reported for convenience.

    Sources
    -------
    Weeks, J. P. (2010). plink: An R package for linking mixed-format
    tests using IRT-based methods. *Journal of Statistical Software*,
    35(12), Eqs. 7-13 (local copy
    fetched-wave3/weeks-2010-plink-JSS35.pdf).
    Marco, G. L. (1977). Item characteristic curve solutions to three
    intractable testing problems. *Journal of Educational
    Measurement*, 14, 139-160 (mean/sigma).
    Loyd, B. H. & Hoover, H. D. (1980). Vertical equating using the
    Rasch model. *Journal of Educational Measurement*, 17, 179-193
    (mean/mean).

    Parameters
    ----------
    a_from, b_from : sequences of float
        Common-item discriminations and difficulties on the from
        scale.
    a_to, b_to : sequences of float
        The same items' parameters on the to scale.
    method : str
        "mean/mean" (default) or "mean/sigma".

    Returns
    -------
    RichResult
        Keys: A, B, a_transformed, b_transformed, method.
    """
    af = [float(v) for v in a_from]
    bf = [float(v) for v in b_from]
    at = [float(v) for v in a_to]
    bt = [float(v) for v in b_to]
    s = len(af)
    if not (len(bf) == len(at) == len(bt) == s) or s < 2:
        raise ValueError("need >= 2 common items with matching lengths")
    meth = str(method).lower().replace("_", "/")
    if meth == "mean/mean":
        ma_f = sum(af) / s
        ma_t = sum(at) / s
        if ma_t == 0:
            raise ValueError("mean of a_to is zero")
        A = ma_f / ma_t
    elif meth == "mean/sigma":
        mb_f = sum(bf) / s
        mb_t = sum(bt) / s
        sd_f = math.sqrt(sum((x - mb_f) ** 2 for x in bf) / (s - 1))
        sd_t = math.sqrt(sum((x - mb_t) ** 2 for x in bt) / (s - 1))
        if sd_f == 0:
            raise ValueError("sd of b_from is zero")
        A = sd_t / sd_f
    else:
        raise ValueError("method must be 'mean/mean' or 'mean/sigma'")
    B = sum(bt) / s - A * (sum(bf) / s)
    return RichResult(payload={
        "A": A,
        "B": B,
        "a_transformed": [x / A for x in af],
        "b_transformed": [A * x + B for x in bf],
        "n_common": s,
        "method": "IRT moment linking (%s; plink Eqs. 12-13)" % meth,
    })


# long descriptive alias (stub-era name)
irt_linking_mean_mean = linkmm


def cheatsheet():
    return "linkmm: A=mu(aF)/mu(aT) (or sd ratio), B=mu(bT)-A mu(bF)"

# public names resolved by fn/_lazy_map.json
linking_meanmean = linkmm
