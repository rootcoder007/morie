"""Canadian Forest Fire Weather Index System (Van Wagner & Pickett 1985)."""

import math

from ._richresult import RichResult

__all__ = ["fwxF", "fire_weather_index"]

# Effective day lengths (DMC, EL) and day-length factors (DC, FL) by
# month, transcribed from the DATA statements of the standard FORTRAN
# program in Van Wagner & Pickett (1985), Forestry Technical Report 33
# (fetched-wave3/vanwagner-pickett-1985-ftr33.pdf, program listing):
#   DATA EL /6.5,7.5,9.0,12.8,13.9,13.9,12.4,10.9,9.4,8.0,7.0,6.0/
#   DATA FL /-1.6,-1.6,-1.6,0.9,3.8,5.8,6.4,5.0,2.4,0.4,-1.6,-1.6/
_EL = (6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0)
_FL = (-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6)


def _ffmc_day(f0, t, h, w, r):
    # Fine Fuel Moisture Code, statements 110-165 of the FTR-33 program.
    wmo = 147.2 * (101.0 - f0) / (59.5 + f0)
    if r > 0.5:
        ra = r - 0.5
        dm = 42.5 * ra * math.exp(-100.0 / (251.0 - wmo)) \
            * (1.0 - math.exp(-6.93 / ra))
        if wmo > 150.0:
            wmo = wmo + dm + 0.0015 * (wmo - 150.0) ** 2 * math.sqrt(ra)
        else:
            wmo = wmo + dm
        if wmo > 250.0:
            wmo = 250.0
    ed = 0.942 * h ** 0.679 + 11.0 * math.exp((h - 100.0) / 10.0) \
        + 0.18 * (21.1 - t) * (1.0 - 1.0 / math.exp(0.115 * h))
    if wmo < ed:
        ew = 0.618 * h ** 0.753 + 10.0 * math.exp((h - 100.0) / 10.0) \
            + 0.18 * (21.1 - t) * (1.0 - 1.0 / math.exp(0.115 * h))
        if wmo < ew:
            z = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) \
                + 0.0694 * math.sqrt(w) * (1.0 - ((100.0 - h) / 100.0) ** 8)
            x = z * 0.581 * math.exp(0.0365 * t)
            wm = ew - (ew - wmo) / 10.0 ** x
        else:
            wm = wmo
    elif wmo == ed:
        wm = wmo
    else:
        z = 0.424 * (1.0 - (h / 100.0) ** 1.7) \
            + 0.0694 * math.sqrt(w) * (1.0 - (h / 100.0) ** 8)
        x = z * 0.581 * math.exp(0.0365 * t)
        wm = ed + (wmo - ed) / 10.0 ** x
    ffm = 59.5 * (250.0 - wm) / (147.2 + wm)
    if ffm > 101.0:
        ffm = 101.0
    if ffm < 0.0:
        ffm = 0.0
    return ffm


def _dmc_day(p0, t, h, r, month):
    # Duff Moisture Code, statements 165-210.
    if t < -1.1:
        t = -1.1
    rk = 1.894 * (t + 1.1) * (100.0 - h) * _EL[month - 1] * 1e-4
    if r > 1.5:
        rw = 0.92 * r - 1.27
        wmi = 20.0 + 280.0 / math.exp(0.023 * p0)
        if p0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * p0)
        elif p0 <= 65.0:
            b = 14.0 - 1.3 * math.log(p0)
        else:
            b = 6.2 * math.log(p0) - 17.2
        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - math.log(wmr - 20.0))
        if pr < 0.0:
            pr = 0.0
    else:
        pr = p0
    return pr + rk


def _dc_day(d0, t, r, month):
    # Drought Code, statements 215-235.
    if t < -2.8:
        t = -2.8
    pe = (0.36 * (t + 2.8) + _FL[month - 1]) / 2.0
    if r > 2.8:
        rw = 0.83 * r - 1.27
        smi = 800.0 * math.exp(-d0 / 400.0)
        dr = d0 - 400.0 * math.log(1.0 + 3.937 * rw / smi)
        if dr < 0.0:
            dr = 0.0
    else:
        dr = d0
    dc = dr + pe
    if dc < 0.0:
        dc = 0.0
    return dc


def _isi_bui_fwi(ffm, dmc, dc, w):
    # ISI, BUI, FWI, DSR, statements 235-280.
    fm = 147.2 * (101.0 - ffm) / (59.5 + ffm)
    sf = 19.115 * math.exp(fm * -0.1386) * (1.0 + fm ** 5.31 / 4.93e7)
    isi = sf * math.exp(0.05039 * w)
    if dmc == 0.0 and dc == 0.0:
        bui = 0.0
    else:
        bui = 0.8 * dc * dmc / (dmc + 0.4 * dc)
        if bui < dmc:
            p = (dmc - bui) / dmc
            cc = 0.92 + (0.0114 * dmc) ** 1.7
            bui = dmc - cc * p
            if bui < 0.0:
                bui = 0.0
    if bui > 80.0:
        bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 / math.exp(0.023 * bui)))
    else:
        bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
    if bb <= 1.0:
        fwi = bb
    else:
        fwi = math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647)
    dsr = 0.0272 * fwi ** 1.77
    return isi, bui, fwi, dsr


def fwxF(temp, rh, wind, rain, month, ffmc_init=85.0, dmc_init=6.0,
         dc_init=15.0):
    """
    Canadian Forest Fire Weather Index (FWI) System, daily codes.

    Runs the six standard components day by day from noon weather
    observations: the three moisture codes FFMC (fine fuel), DMC
    (duff) and DC (drought), then ISI (initial spread), BUI (buildup),
    FWI and the daily severity rating DSR = 0.0272 FWI^1.77.  The
    implementation follows the standard FORTRAN program of Van Wagner
    & Pickett (1985) statement by statement (that program is declared
    in the report to be "a standard for processing the equations in
    their most accurate mathematical form").

    Sources
    -------
    Van Wagner, C. E. & Pickett, T. L. (1985). Equations and FORTRAN
    program for the Canadian Forest Fire Weather Index System.
    Canadian Forestry Service, Forestry Technical Report 33, Ottawa
    (local copy fetched-wave3/vanwagner-pickett-1985-ftr33.pdf; FFMC
    statements 110-165, DMC 165-210, DC 215-235, ISI/BUI/FWI/DSR
    235-280, day-length DATA EL/FL).
    Note: later implementations (e.g. Wang et al. 2015, NOR-X-424;
    CRAN cffdrs) revise the FFMC conversion constant 147.2 to
    147.27723; this module keeps 147.2 to match the FTR-33
    standard program and its printed sample output (cffdrs
    cross-check: DMC/DC/BUI agree exactly, FFMC within 0.13).
    Van Wagner, C. E. (1987). Development and structure of the
    Canadian Forest Fire Weather Index System. Canadian Forestry
    Service, Forestry Technical Report 35 (system description).

    Parameters
    ----------
    temp, rh, wind, rain : sequences of float
        Daily noon temperature (deg C), relative humidity (%), wind
        speed (km/h), 24-h rain (mm).
    month : int or sequence of int
        Calendar month (1-12) per day (scalar = same month all days).
    ffmc_init, dmc_init, dc_init : float
        Starting code values (FTR-33 defaults 85, 6, 15).

    Returns
    -------
    RichResult
        Keys: ffmc, dmc, dc, isi, bui, fwi, dsr (daily lists), n_days.
    """
    t = [float(v) for v in temp]
    h = [float(v) for v in rh]
    w = [float(v) for v in wind]
    r = [float(v) for v in rain]
    n = len(t)
    if not (len(h) == len(w) == len(r) == n):
        raise ValueError("temp, rh, wind, rain must have equal length")
    if isinstance(month, (int, float)):
        mo = [int(month)] * n
    else:
        mo = [int(m) for m in month]
        if len(mo) != n:
            raise ValueError("month must be scalar or match n_days")
    if any(m < 1 or m > 12 for m in mo):
        raise ValueError("month entries must be in 1..12")
    if any(v < 0 or v > 100 for v in h):
        raise ValueError("rh must be in [0, 100]")
    f0, p0, d0 = float(ffmc_init), float(dmc_init), float(dc_init)
    out = {k: [] for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr")}
    for i in range(n):
        f0 = _ffmc_day(f0, t[i], h[i], w[i], r[i])
        p0 = _dmc_day(p0, t[i], h[i], r[i], mo[i])
        d0 = _dc_day(d0, t[i], r[i], mo[i])
        isi, bui, fwi, dsr = _isi_bui_fwi(f0, p0, d0, w[i])
        out["ffmc"].append(f0)
        out["dmc"].append(p0)
        out["dc"].append(d0)
        out["isi"].append(isi)
        out["bui"].append(bui)
        out["fwi"].append(fwi)
        out["dsr"].append(dsr)
    out.update(n_days=n,
               method="Canadian FWI System (Van Wagner & Pickett 1985)")
    return RichResult(payload=out)


# long descriptive alias (stub-era name)
fire_weather_index = fwxF


def cheatsheet():
    return "fwxF: Canadian FWI System daily FFMC/DMC/DC/ISI/BUI/FWI/DSR (FTR-33)"
