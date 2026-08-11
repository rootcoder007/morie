"""HBV conceptual rainfall-runoff model (Bergstrom; Seibert & Vis 2012)."""

import math

from ._richresult import RichResult

__all__ = ["hbvMod", "hbv_hydrology"]


def _maxbas_weights(maxbas):
    # Eq. 6 of Seibert & Vis (2012): c(i) = int_{i-1}^{i} of
    # (2/M - |u - M/2| * 4/M^2) du over the triangle of base M.
    m = float(maxbas)
    nw = int(math.ceil(m))
    def _antider(u):
        # antiderivative of 2/m - |u - m/2| * 4/m^2, with F(0)=0, F(m)=1
        if u <= m / 2.0:
            return 2.0 * u * u / (m * m)
        return 4.0 * u / m - 2.0 * u * u / (m * m) - 1.0
    w = []
    for i in range(1, nw + 1):
        lo, hi = i - 1.0, min(float(i), m)
        w.append(_antider(hi) - _antider(lo))
    return w


def hbvMod(precip, temp, epot, params, init=None):
    """
    HBV conceptual rainfall-runoff model, daily time step.

    Standard HBV structure as formalized in Seibert & Vis (2012),
    Eqs. 1-6: degree-day snow routine with refreezing (Eqs. 1-2),
    soil-moisture accounting with recharge fraction
    (S_soil/FC)^BETA (Eq. 3) and actual evaporation
    E_act = E_pot * min(S_soil/(FC*LP), 1) (Eq. 4), two linear
    groundwater boxes with threshold outflow
    Q_GW = K2*SLZ + K1*SUZ + K0*max(SUZ-UZL, 0) (Eq. 5) and maximum
    percolation PERC from the upper to the lower box, and a
    triangular routing filter of base MAXBAS (Eq. 6).  Water is
    conserved to machine precision:
    sum(P) = sum(E_act) + sum(Q) + (final - initial storage)
    + water still in the routing queue.

    Sources
    -------
    Seibert, J. & Vis, M. J. P. (2012). Teaching hydrological
    modeling with a user-friendly catchment-runoff-model software
    package. *Hydrology and Earth System Sciences*, 16, 3315-3325,
    Eqs. 1-6 (local copy
    fetched-wave3/seibert-vis-2012-hbv-light-hess16-3315.pdf).
    Bergstrom, S. (1995). The HBV model. In V. P. Singh (ed.),
    *Computer Models of Watershed Hydrology*, Water Resources
    Publications, 443-476 (original model description).

    Parameters
    ----------
    precip, temp, epot : sequences of float
        Daily precipitation (mm), mean temperature (deg C), and
        potential evaporation (mm).
    params : dict
        Keys: tt (threshold temperature, deg C), cfmax (degree-day
        factor, mm/d/degC), cfr (refreezing coefficient, default
        0.05), fc (max soil storage, mm), lp (fraction of fc above
        which E_act = E_pot), beta (recharge shape), k0, k1, k2
        (outflow coefficients, 1/d), uzl (threshold, mm), perc (max
        percolation, mm/d), maxbas (routing base length, days).
    init : dict, optional
        Starting states: snow, swater (liquid water in snowpack),
        soil, suz, slz (all mm; default 0).

    Returns
    -------
    RichResult
        Keys: q (routed runoff), q_gw (unrouted), snow, soil, suz,
        slz, e_act (daily lists), mass_balance_error, params_used.
    """
    p = [float(v) for v in precip]
    t = [float(v) for v in temp]
    ep = [float(v) for v in epot]
    n = len(p)
    if not (len(t) == len(ep) == n):
        raise ValueError("precip, temp, epot must have equal length")
    req = ("tt", "cfmax", "fc", "lp", "beta", "k0", "k1", "k2",
           "uzl", "perc", "maxbas")
    miss = [k for k in req if k not in params]
    if miss:
        raise ValueError("params missing: %s" % ", ".join(miss))
    tt = float(params["tt"]); cfmax = float(params["cfmax"])
    cfr = float(params.get("cfr", 0.05))
    fc = float(params["fc"]); lp = float(params["lp"])
    beta = float(params["beta"])
    k0 = float(params["k0"]); k1 = float(params["k1"])
    k2 = float(params["k2"])
    uzl = float(params["uzl"]); perc = float(params["perc"])
    maxbas = float(params["maxbas"])
    if fc <= 0 or maxbas < 1:
        raise ValueError("fc must be positive and maxbas >= 1")
    init = init or {}
    snow = float(init.get("snow", 0.0))
    swater = float(init.get("swater", 0.0))
    soil = float(init.get("soil", 0.0))
    suz = float(init.get("suz", 0.0))
    slz = float(init.get("slz", 0.0))
    s0 = snow + swater + soil + suz + slz

    w = _maxbas_weights(maxbas)
    queue = [0.0] * len(w)
    out = {k: [] for k in ("q", "q_gw", "snow", "soil", "suz", "slz",
                           "e_act")}
    for i in range(n):
        # snow routine (Eqs. 1-2); precipitation phase by TT
        if t[i] <= tt:
            snow += p[i]
            rain = 0.0
        else:
            rain = p[i]
        melt = min(cfmax * (t[i] - tt), snow) if t[i] > tt else 0.0
        snow -= melt
        swater += melt
        refreeze = min(cfr * cfmax * (tt - t[i]), swater) \
            if t[i] < tt else 0.0
        swater -= refreeze
        snow += refreeze
        # liquid water above 10% of snowpack becomes soil input
        hold = 0.1 * snow
        insoil = rain + max(swater - hold, 0.0)
        swater = min(swater, hold)
        # soil routine (Eqs. 3-4)
        recharge = insoil * (soil / fc) ** beta
        soil += insoil - recharge
        if soil > fc:                       # overflow to recharge
            recharge += soil - fc
            soil = fc
        eact = ep[i] * min(soil / (fc * lp), 1.0)
        eact = min(eact, soil)
        soil -= eact
        # groundwater boxes (Eq. 5) with max percolation PERC
        suz += recharge
        pc = min(perc, suz)
        suz -= pc
        slz += pc
        q0 = k0 * max(suz - uzl, 0.0)
        q1 = k1 * suz
        q2 = k2 * slz
        qgw = q0 + q1 + q2
        suz -= q0 + q1
        slz -= q2
        # triangular routing (Eq. 6)
        for j in range(len(w)):
            queue[j] += w[j] * qgw
        q = queue.pop(0)
        queue.append(0.0)
        out["q"].append(q)
        out["q_gw"].append(qgw)
        out["snow"].append(snow + swater)
        out["soil"].append(soil)
        out["suz"].append(suz)
        out["slz"].append(slz)
        out["e_act"].append(eact)
    s1 = snow + swater + soil + suz + slz
    mbe = sum(p) - sum(out["e_act"]) - sum(out["q"]) - (s1 - s0) \
        - sum(queue)
    out.update(mass_balance_error=mbe, n_days=n,
               params_used=dict(tt=tt, cfmax=cfmax, cfr=cfr, fc=fc,
                                lp=lp, beta=beta, k0=k0, k1=k1, k2=k2,
                                uzl=uzl, perc=perc, maxbas=maxbas),
               method="HBV (Seibert & Vis 2012, Eqs. 1-6)")
    return RichResult(payload=out)


# long descriptive alias (stub-era name)
hbv_hydrology = hbvMod


def cheatsheet():
    return "hbvMod: HBV rainfall-runoff (snow/soil/2 GW boxes/MAXBAS routing)"
