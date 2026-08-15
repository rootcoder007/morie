# morie.fn -- function file (rootcoder007/morie)
r"""Palmer's Drought Severity Index from a monthly water balance.

Rainfall alone does not measure drought: the same 40 mm is ample in a cool
month and a deficit in a hot one. Palmer's construction is to ask what
precipitation would have been CLIMATICALLY APPROPRIATE for the month's
existing conditions -- the CAFEC precipitation

.. math:: \hat P = \alpha PE + \beta PR + \gamma PRO - \delta PL,

with the four coefficients estimated from the record itself as ratios of
mean actual to mean potential quantities. The departure :math:`d = P -
\hat P` is then weighted into the moisture anomaly :math:`Z = K d` and
accumulated:

.. math:: X_i = 0.897\,X_{i-1} + Z_i/3.

**The 0.897 and the /3 are not free parameters.** Palmer fitted them so
that the index would be comparable between climates, which is the entire
point of the index and the reason a locally re-tuned version is no longer
PDSI. The duration factors are returned so that is visible.

References
----------
Palmer, W. C. (1965) *Meteorological Drought*, Research Paper No. 45,
U.S. Weather Bureau, Washington DC. The water balance, the CAFEC
precipitation, the climatic characteristic K and the duration factors of
the accumulation.

Alley, W. M. (1984) "The Palmer Drought Severity Index: limitations and
assumptions", *Journal of Climate and Applied Meteorology* **23**(7),
1100-1109, doi:10.1175/1520-0450(1984)023<1100:TPDSIL>2.0.CO;2. What the
index does and does not measure.

Wells, N., Goddard, S. and Hayes, M. J. (2004) "A self-calibrating Palmer
Drought Severity Index", *Journal of Climate* **17**(12), 2335-2351,
doi:10.1175/1520-0442(2004)017<2335:ASPDSI>2.0.CO;2.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["palmer_pdsi"]

_EPS = 1e-12


def palmer_pdsi(precip, pet, awc=100.0, month=None):
    r"""Two-layer water balance, CAFEC precipitation, Z index and PDSI."""
    P = [float(v) for v in k.vec(precip)]
    PE = [float(v) for v in k.vec(pet)]
    n = len(P)
    if n == 0:
        raise ValueError("droPDSI: an empty series has no water balance")
    if len(PE) != n:
        raise ValueError("droPDSI: %d precipitation but %d PET values"
                         % (n, len(PE)))
    awc = float(awc)
    if awc <= 0.0:
        raise ValueError("droPDSI: the available water capacity must be "
                         "positive")
    su_cap = min(25.4, awc)            # surface layer, Palmer's 1 inch
    sl_cap = awc - su_cap              # underlying layer

    Ss, Su = su_cap, sl_cap            # start at field capacity
    ET, R, RO, L = [], [], [], []
    PR, PRO, PL = [], [], []
    for i in range(n):
        pr = (su_cap - Ss) + (sl_cap - Su)      # potential recharge
        pro = Ss + Su                            # potential runoff (Palmer)
        # potential loss: surface first, then the underlying layer
        pls = min(PE[i], Ss)
        plu = min((PE[i] - pls) * Su / awc if awc > _EPS else 0.0, Su)
        pl = pls + plu
        PR.append(pr)
        PRO.append(pro)
        PL.append(pl)

        if P[i] >= PE[i]:
            et = PE[i]
            excess = P[i] - PE[i]
            recharge_s = min(su_cap - Ss, excess)
            Ss += recharge_s
            excess -= recharge_s
            recharge_u = min(sl_cap - Su, excess)
            Su += recharge_u
            excess -= recharge_u
            ro = excess
            loss = 0.0
        else:
            need = PE[i] - P[i]
            loss_s = min(Ss, need)
            Ss -= loss_s
            need -= loss_s
            loss_u = min(Su, need * Su / awc if awc > _EPS else 0.0)
            Su -= loss_u
            et = P[i] + loss_s + loss_u
            ro = 0.0
            loss = loss_s + loss_u
            recharge_s = recharge_u = 0.0
        ET.append(et)
        R.append(recharge_s + recharge_u if P[i] >= PE[i] else 0.0)
        RO.append(ro)
        L.append(loss)

    def ratio(num, den):
        sn, sd = sum(num), sum(den)
        return sn / sd if sd > _EPS else 0.0

    alpha = ratio(ET, PE)
    beta = ratio(R, PR)
    gamma = ratio(RO, PRO)
    delta = ratio(L, PL)

    Phat = [alpha * PE[i] + beta * PR[i] + gamma * PRO[i] - delta * PL[i]
            for i in range(n)]
    d = [P[i] - Phat[i] for i in range(n)]

    # Palmer's climatic characteristic is computed PER CALENDAR MONTH and
    # then rescaled across months; a single record-wide K collapses to zero
    # on ordinary seasonal data and takes the whole index with it.
    if month is None:
        mon = [i % 12 for i in range(n)]
    else:
        mon = [int(v) % 12 for v in k.vec(month)]
        if len(mon) != n:
            raise ValueError("droPDSI: %d observations but %d month labels"
                             % (n, len(mon)))
    Kp_month = [0.0] * 12
    D_month = [0.0] * 12
    for j in range(12):
        idx = [i for i in range(n) if mon[i] == j]
        if not idx:
            continue
        cnt = float(len(idx))
        Dj = sum(abs(d[i]) for i in idx) / cnt
        mPE = sum(PE[i] for i in idx) / cnt
        mR = sum(R[i] for i in idx) / cnt
        mRO = sum(RO[i] for i in idx) / cnt
        mP = sum(P[i] for i in idx) / cnt
        mL = sum(L[i] for i in idx) / cnt
        ratio_j = (mPE + mR + mRO) / (mP + mL + _EPS) + 2.8
        arg = ratio_j / (Dj + _EPS)
        Kp_month[j] = 1.5 * math.log10(arg if arg > _EPS else _EPS) + 0.5
        D_month[j] = Dj
    denom = sum(D_month[j] * Kp_month[j] for j in range(12))
    if abs(denom) > _EPS:
        Kp_month = [17.67 * v / denom for v in Kp_month]
    Kp = sum(Kp_month) / 12.0
    Z = [Kp_month[mon[i]] * d[i] for i in range(n)]

    X = []
    prev = 0.0
    for i in range(n):
        cur = 0.897 * prev + Z[i] / 3.0
        X.append(cur)
        prev = cur

    return RichResult(payload={
        "estimate": X, "pdsi": X, "z_index": Z, "departure": d,
        "cafec_precip": Phat,
        "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
        "K": Kp, "K_month": Kp_month,
        "mean_abs_departure": D_month, "evapotranspiration": ET, "recharge": R, "runoff": RO,
        "loss": L, "soil_surface_capacity": su_cap,
        "soil_under_capacity": sl_cap, "n": n,
        "duration_factor": 0.897, "duration_divisor": 3.0,
        "method": "Palmer Drought Severity Index from a two-layer water "
                  "balance (Palmer 1965, Research Paper 45)",
        "note": "the 0.897 and the /3 are Palmer's fitted duration factors, "
                "chosen so the index is comparable BETWEEN climates -- a "
                "locally re-tuned version is no longer PDSI",
    })


def cheatsheet():
    return ("droPDSI: palmer_pdsi(precip, pet, awc) -> PDSI, Z index and the "
            "CAFEC water balance (Palmer 1965, Meteorological Drought, "
            "Research Paper No. 45, U.S. Weather Bureau)")
