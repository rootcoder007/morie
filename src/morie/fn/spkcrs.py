# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross K-function for bivariate point patterns."""

import numpy as np

from ._richresult import RichResult
from ._schab_pp import (as_points, as_region, cross_k_combined,
                        cross_k_function, diggle_chetwynd_d)

__all__ = ["schabenberger_cross_k_function"]


def schabenberger_cross_k_function(points1, points2, lambda1=None, lambda2=None,
                                   r=None, region=None, correction="ripley",
                                   hypothesis="independence"):
    """Cross K-function, Sec. 3.4.4, eq (3.9).

    ``Khat_ij(h) = [lam_i lam_j nu(A)]^-1 sum_k sum_l w(s_k,u_l)^-1
    I(h_kl <= h)``, with ``w`` Ripley's isotropic weight -- the proportion of
    the circumference of a circle centred at ``s_k`` with radius ``h_kl``
    that lies inside the window.

    Because ``Khat_12`` and ``Khat_21`` are not equal even though the
    population functions are, the pooled estimator of Lotwick and Silverman
    (1982) is returned as ``K_star``, with ``L_star = sqrt(K_star/pi)``.

    ``hypothesis`` selects which null the output is aimed at, and the two are
    not interchangeable:

    ``'independence'``
        ``K_ij(h) = pi h^2`` regardless of either pattern, so ``L*-h`` is the
        diagnostic: positive indicates attraction, negative repulsion.
    ``'random_labelling'``
        eq (3.10) ``K_11 = K_22 = K_12``; the statistic is Diggle and
        Chetwynd's ``D(h) = K_ii(h) - K_jj(h)``. This conditions on all
        locations and randomises only the marks.

    ``lambda1`` and ``lambda2`` are accepted for signature compatibility;
    the intensities are estimated from the patterns and the window as
    ``n/nu(A)`` per eq (3.8), and any supplied values are reported alongside
    for comparison rather than substituted.

    Returns
    -------
    RichResult
        Keys: ``estimate`` (``K_star``), ``K_12``, ``K_21``, ``L_star``,
        ``L_minus_h``, ``K_independence``, ``r``, ``lambda_1``, ``lambda_2``,
        and for random labelling ``D``, ``K_11``, ``K_22``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005), Sec. 3.4.4, eqs (3.9)-(3.10),
    pp. 103-105. Lotwick, H. W. & Silverman, B. W. (1982), JRSS B 44:406-413.
    Diggle, P. J. (1983), Statistical Analysis of Spatial Point Patterns.
    Diggle, P. J. & Chetwynd, A. G. (1991), Biometrics 47:1155-1163.
    """
    p1 = as_points(points1)
    p2 = as_points(points2)
    region = as_region(region, np.vstack([p1, p2]))
    if r is None:
        xmin, ymin, xmax, ymax = region
        r = np.linspace(0.0, 0.25 * min(xmax - xmin, ymax - ymin), 11)[1:]
    r = np.atleast_1d(np.asarray(r, dtype=float))

    res = cross_k_combined(p1, p2, region, r, correction=correction)
    payload = dict(res)
    payload["estimate"] = res["K_star"]
    payload["correction"] = correction
    payload["hypothesis"] = hypothesis
    if lambda1 is not None:
        payload["lambda_1_supplied"] = float(lambda1)
    if lambda2 is not None:
        payload["lambda_2_supplied"] = float(lambda2)

    lines = [("n1, n2", (p1.shape[0], p2.shape[0])),
             ("correction", correction),
             ("hypothesis", hypothesis)]
    if hypothesis == "random_labelling":
        d = diggle_chetwynd_d(p1, p2, region, r)
        payload.update({"D": d["D"], "K_11": d["K_11"], "K_22": d["K_22"]})
        lines.append(("max |D(h)|", float(np.nanmax(np.abs(d["D"])))))
    elif hypothesis != "independence":
        raise ValueError("`hypothesis` must be 'independence' or 'random_labelling'")
    else:
        lines.append(("max |L*(h) - h|", float(np.abs(res["L_minus_h"]).max())))
    return RichResult(title="Cross K-function", summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spkcrs: cross K-function for bivariate point patterns with "
            "Ripley edge correction (Sec. 3.4.4, eq (3.9))")
