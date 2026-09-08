# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spatio-temporal point processes."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_st import (cstr_reference, cstr_test, st_intensity,
                        st_marginal_intensities)

__all__ = ["schabenberger_st_point_process"]

_TYPES = ("earthquake", "explosion", "birth_death", "sampled_in_time")


def schabenberger_st_point_process(points, region, time_interval, times=None,
                                   process_type=None, n_space_bins=3,
                                   n_time_bins=3):
    """Spatio-temporal point process intensity, Sec. 9.5.

    The first-order intensity is eq (9.20),

        lambda(s,t) = lim_{|ds|,|dt| -> 0} E[N(ds, dt)] / (|ds| |dt|),

    where N(ds, dt) counts events in an infinitesimal CYLINDER with base ds
    and height dt (Dorai-Raj, 2001). The cylinder, rather than a ball in R^3,
    is the same refusal to treat time as a third spatial coordinate that runs
    through the whole chapter. Under first-order stationarity in space and
    time the intensity does not depend on s or t and is estimated by
    N / (|A| |T|).

    The marginals of eqs (9.21) and (9.22) are also returned:

        lambda(s, .) = integral_T lambda(s,v) dv
        lambda(., t) = integral_D lambda(u,t) du

    estimated by binning -- the marginal spatial intensity of a cell is its
    count over the cell AREA, already integrated across all of T; the
    marginal temporal intensity of a bin is its count over the bin WIDTH,
    already integrated across all of D. The corollaries in Sec. 9.5.3 give
    the consistency checks: under first-order stationarity in time
    lambda(s,.) = |T| lambda**(s), and in space lambda(.,t) = |A| lambda*(t).

    The benchmark is the completely spatio-temporally random (CSTR) process:
    Poisson in BOTH space and time, so N(A,T) ~ Poisson(lambda |A x T|),
    lambda(s,t) = lambda and lambda_2 = lambda^2. The text's own assessment
    is worth carrying: "If the CSR process is an unattainable standard for
    spatial point processes, then the CSTR process is even more so." It is
    the initial benchmark to test against, not a model of anything. The test
    reported here is the index of dispersion over equal-volume space-time
    cells, the spatio-temporal analogue of the quadrat test, and the index
    itself is returned alongside the p-value because with few cells the test
    has little power and a non-rejection is not evidence of randomness.

    ``process_type`` records which of the Sec. 9.5.1 types the data are taken
    to be. It does not change the arithmetic -- it is carried into the result
    because the same numbers mean different things for each, and because the
    text notes two of them can be indistinguishable from the data alone: a
    birth-death process observed at fixed times can look exactly like a
    pattern sampled in time, since an event present at t_i and absent at
    t_{i+1} may be a death or a displacement.

    Parameters
    ----------
    points : array-like, shape (n, 2)
    region : (xmin, xmax, ymin, ymax)
    time_interval : (t0, t1)
    times : array-like, shape (n,)
        Event times. Required; the third positional slot is kept for the
        region so the signature reads (where, over what area, over what span).
    process_type : {"earthquake", "explosion", "birth_death", "sampled_in_time"}, optional
    n_space_bins, n_time_bins : int
        Grid for the marginals and the CSTR test.

    Returns
    -------
    RichResult
        Keys: ``intensity``, ``n``, ``area``, ``duration``,
        ``marginal_spatial``, ``marginal_temporal``, ``cstr``,
        ``index_of_dispersion``, ``p_value``, ``process_type``.

    References
    ----------
    Schabenberger & Gotway (2005), Sec. 9.5, eqs (9.20)-(9.23).
    """
    if times is None:
        raise ValueError(
            "`times` is required: a spatio-temporal point process needs an "
            "event time for every event")
    if process_type is not None and process_type not in _TYPES:
        raise ValueError(f"`process_type` must be one of {_TYPES}")

    lam = st_intensity(points, times, region, time_interval)
    marg = st_marginal_intensities(points, times, region, time_interval,
                                   n_space_bins=n_space_bins,
                                   n_time_bins=n_time_bins)
    ref = cstr_reference(lam["area"], lam["duration"], lam["intensity"])
    test = cstr_test(points, times, region, time_interval,
                     n_space_bins=n_space_bins, n_time_bins=n_time_bins)

    payload = {
        "intensity": lam["intensity"],
        "n": lam["n"],
        "area": lam["area"],
        "duration": lam["duration"],
        "volume": lam["volume"],
        "marginal_spatial": marg["marginal_spatial"],
        "marginal_temporal": marg["marginal_temporal"],
        "cell_area": marg["cell_area"],
        "time_bin_width": marg["bin_width"],
        "cstr": ref,
        "index_of_dispersion": test["index_of_dispersion"],
        "df": test["df"],
        "p_value": test["p_value"],
        "cell_counts": test["counts"],
        "process_type": process_type,
    }
    lines = [("events", lam["n"]),
             ("intensity lambda(s,t)", lam["intensity"]),
             ("space-time volume |A x T|", lam["volume"]),
             ("CSTR index of dispersion", test["index_of_dispersion"]),
             ("CSTR p-value", test["p_value"])]
    if process_type is not None:
        lines.insert(0, ("process type", process_type))

    n_cells = int(np.asarray(test["counts"]).size)
    if n_cells < 20:
        payload["power_note"] = (
            f"only {n_cells} space-time cells: the dispersion test has little "
            f"power here, and failing to reject CSTR is not evidence for it")
    if process_type in ("birth_death", "sampled_in_time"):
        payload["identifiability_note"] = (
            "a birth-death process observed at fixed times can be "
            "indistinguishable from a pattern sampled in time (Sec. 9.5.1); "
            "an event absent at the next time may be a death or a "
            "displacement")
    if process_type == "earthquake":
        payload["conditional_note"] = (
            "for an earthquake process the conditional intensities "
            "lambda(s|t) and lambda(t|s) are not meaningful and should be "
            "replaced by intensities on intervals in time or areas in space "
            "(Rathbun, 1996)")

    return RichResult(title="Spatio-temporal point process",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spstp: spatio-temporal point process (Sec. 9.5) -- cylinder "
            "intensity (9.20), marginals (9.21)-(9.22), CSTR benchmark")
