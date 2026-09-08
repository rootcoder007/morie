r"""In vitro to in vivo prediction of hepatic intrinsic clearance.

Wood, F. L., Houston, J. B., & Hallifax, D. (2017) "Clearance Prediction
Methodology Needs Fundamental Improvement: Trends Common to Rat and Human
Hepatocytes/Microsomes and Implications for Experimental Methodology",
*Drug Metabolism and Disposition* 45(11), 1178-1188.

The ledger row for this module also carried a second citation, "Pirmohamed
(2019)". It does not exist: a PubMed search for Pirmohamed M as author in
2019 with clearance, hepatocyte or microsome returns nothing, and his
publication record is pharmacogenetics rather than clearance prediction. It
is one of the wave's fabricated references and is not cited here.

The paper's subject is a failure, not a method: scaling in vitro clearance to
the whole liver *systematically underpredicts*, and the underprediction gets
worse as in vivo clearance rises. This module implements the pipeline it
uses to establish that, so the bias can be measured on the caller's own data
rather than assumed away.

**Binding in the incubation** (equations 1 and 2), when not measured:

.. math::

   f_{u,mic} &= \frac{1}{1 + P \cdot 10^{\,0.072 (\log P/D)^2
                + 0.067 \log P/D - 1.126}} \\
   f_{u,heps} &= \frac{1}{1 + 125\, V_R \cdot 10^{\,0.072 (\log P/D)^2
                 + 0.067 \log P/D - 1.126}}

with :math:`P` the microsomal protein concentration, :math:`V_R` the volume
ratio of hepatocytes to medium (0.005 at :math:`10^6` cells/ml), and
:math:`\log P/D` "either the log P value for basic and neutral drugs or the
log D value for acidic drugs".

**Scaling to the whole liver** (equation 3):

.. math:: \mathrm{predicted\ } CL_{int,u} =
          \frac{CL_{int,\ in\ vitro} \times \mathrm{PBSF} \times L_W}
               {f_{u,mic}\ \mathrm{or}\ f_{u,heps}}

with the paper's own physiological constants: 40 mg microsomal protein per g
liver for human and 60 for rat; hepatocellularity :math:`120 \times 10^6`
cells per g liver for both; liver weight 21.4 g/kg for human and 40 g/kg for
rat.

**The observed value** (equation 4, well-stirred):

.. math:: CL_{int,u} = \frac{CL_h}{f_{u,b}\,[1 - CL_h/Q_h]},
          \qquad Q_h = 20.7\ \mathrm{ml/min/kg\ (human)},\ 100\ (\mathrm{rat}).

The parallel-tube model is the other extreme of hepatic dispersion and is
implemented too (``liver_model="parallel_tube"``,
:math:`CL_{int,u} = -Q_h \ln(1 - CL_h/Q_h)/f_{u,b}`); the paper reports the
well-stirred results "since the difference in bias between these two liver
models ... was found to be marginal", which the anchor confirms on the
module's own numbers.

**Accuracy and precision** are the paper's equations 5-8: average fold error
:math:`AFE = 10^{\sum \log(\mathrm{pred}/\mathrm{obs})/n}`, root mean square
error, the empirical scaling factor
:math:`ESF = \mathrm{observed}/\mathrm{predicted}` per compound and its log
average, plus the percentage within 2-fold. "Since underprediction yields an
AFE below 1, underprediction was also expressed as fold underprediction
(inverse of AFE)."

Blood-versus-plasma bookkeeping follows the paper as well: given plasma
clearance and :math:`f_{u,p}`, :math:`CL_b = CL_{plasma}/R_b` and
:math:`f_{u,b} = f_{u,p}/R_b`, and where :math:`R_b` is unavailable it "was
assumed to be equal to 1 for a basic or neutral compound and 0.55
(1-hematocrit) for an acidic compound".
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["clrnt", "hepatic_clearance_prediction", "clearance_intrinsic", "fu_microsomes",
           "fu_hepatocytes", "scale_to_liver", "observed_clint_u",
           "prediction_accuracy", "blood_from_plasma"]

#: The paper's physiological scaling constants, by species and system.
CONSTANTS = {
    "human": {"microsomes_pbsf": 40.0,      # mg microsomal protein / g liver
              "hepatocytes_pbsf": 120e6,    # cells / g liver
              "liver_weight": 21.4,         # g liver / kg body weight
              "qh": 20.7},                  # ml/min/kg
    "rat": {"microsomes_pbsf": 60.0,
            "hepatocytes_pbsf": 120e6,
            "liver_weight": 40.0,
            "qh": 100.0},
}


def _binding_term(log_pd):
    """The shared exponent of equations 1 and 2."""
    x = float(log_pd)
    return 10.0 ** (0.072 * x * x + 0.067 * x - 1.126)


def fu_microsomes(log_pd, protein=1.0):
    r"""Equation 1 (Hallifax & Houston 2006):
    :math:`f_{u,mic} = 1/(1 + P \cdot 10^{\ldots})`.

    ``protein`` is the microsomal protein concentration in mg/ml.
    """
    if protein <= 0:
        raise ValueError("clrnt: microsomal protein concentration must be "
                         "positive")
    return 1.0 / (1.0 + float(protein) * _binding_term(log_pd))


def fu_hepatocytes(log_pd, volume_ratio=0.005):
    r"""Equation 2 (Kilford et al. 2008):
    :math:`f_{u,heps} = 1/(1 + 125 V_R \cdot 10^{\ldots})`.

    ``volume_ratio`` is hepatocytes to medium, 0.005 at
    :math:`10^6` cells/ml.
    """
    if volume_ratio <= 0:
        raise ValueError("clrnt: the volume ratio must be positive")
    return 1.0 / (1.0 + 125.0 * float(volume_ratio) *
                  _binding_term(log_pd))


def blood_from_plasma(cl_plasma, fu_plasma, blood_plasma_ratio=None,
                      charge="neutral"):
    r"""Blood clearance and unbound fraction from the plasma versions.

    :math:`CL_b = CL_{plasma}/R_b`, :math:`f_{u,b} = f_{u,p}/R_b`, with
    :math:`R_b` defaulting to 1 for a basic or neutral compound and 0.55
    for an acidic one, as the paper specifies when it is unavailable.
    """
    if charge not in ("acidic", "basic", "neutral"):
        raise ValueError("clrnt: charge must be acidic, basic or neutral")
    rb = (float(blood_plasma_ratio) if blood_plasma_ratio is not None
          else (0.55 if charge == "acidic" else 1.0))
    if rb <= 0:
        raise ValueError("clrnt: the blood/plasma ratio must be positive")
    return float(cl_plasma) / rb, float(fu_plasma) / rb, rb


def scale_to_liver(clint_in_vitro, fu_incubation, system="hepatocytes",
                   species="human", pbsf=None, liver_weight=None):
    r"""Equation 3: scale in vitro :math:`CL_{int}` to predicted in vivo
    :math:`CL_{int,u}`.

    Units follow the system: per :math:`10^6` cells for hepatocytes, per mg
    microsomal protein for microsomes, so the product with the PBSF and the
    liver weight lands in ml/min/kg.
    """
    if species not in CONSTANTS:
        raise ValueError("clrnt: species must be 'human' or 'rat'")
    if system not in ("hepatocytes", "microsomes"):
        raise ValueError("clrnt: system must be 'hepatocytes' or "
                         "'microsomes'")
    if not 0.0 < float(fu_incubation) <= 1.0:
        raise ValueError("clrnt: the incubational unbound fraction must lie "
                         "in (0, 1]")
    c = CONSTANTS[species]
    p = (c["hepatocytes_pbsf"] if system == "hepatocytes"
         else c["microsomes_pbsf"]) if pbsf is None else float(pbsf)
    lw = c["liver_weight"] if liver_weight is None else float(liver_weight)
    if system == "hepatocytes":
        p = p / 1e6            # CLint is quoted per 10^6 cells
    return float(clint_in_vitro) * p * lw / float(fu_incubation)


def observed_clint_u(cl_h, fu_blood, species="human", qh=None,
                     liver_model="well_stirred"):
    r"""Equation 4 and its parallel-tube counterpart.

    Well-stirred: :math:`CL_{int,u} = CL_h/(f_{u,b}[1 - CL_h/Q_h])`.
    Parallel tube: :math:`CL_{int,u} = -Q_h \ln(1 - CL_h/Q_h)/f_{u,b}`.
    Both are extremes of hepatic dispersion; the paper presents the
    well-stirred results.
    """
    if species not in CONSTANTS:
        raise ValueError("clrnt: species must be 'human' or 'rat'")
    if liver_model not in ("well_stirred", "parallel_tube"):
        raise ValueError("clrnt: liver_model must be 'well_stirred' or "
                         "'parallel_tube'")
    q = CONSTANTS[species]["qh"] if qh is None else float(qh)
    cl = float(cl_h)
    fu = float(fu_blood)
    if not 0.0 < fu <= 1.0:
        raise ValueError("clrnt: the unbound fraction must lie in (0, 1]")
    if cl <= 0:
        raise ValueError("clrnt: hepatic clearance must be positive")
    if cl >= q:
        raise ValueError("clrnt: hepatic clearance cannot reach or exceed "
                         "hepatic blood flow (%.4g >= %.4g ml/min/kg)"
                         % (cl, q))
    if liver_model == "well_stirred":
        return cl / (fu * (1.0 - cl / q))
    return -q * math.log(1.0 - cl / q) / fu


def prediction_accuracy(predicted, observed, fold=2.0):
    r"""Equations 5-8 plus the 2-fold count.

    ``afe`` is :math:`10^{\sum\log(\mathrm{pred/obs})/n}`,
    ``fold_underprediction`` its inverse, ``rmse`` equation 6, ``esf`` the
    per-compound :math:`\mathrm{obs}/\mathrm{pred}` of equation 7 and
    ``average_esf`` the log average of equation 8.
    """
    p = [float(v) for v in predicted]
    o = [float(v) for v in observed]
    n = len(p)
    if n == 0 or n != len(o):
        raise ValueError("clrnt: need one observed value per prediction")
    if any(v <= 0 for v in p + o):
        raise ValueError("clrnt: clearances must be positive to take logs")
    afe = 10.0 ** (sum(math.log10(p[i] / o[i]) for i in range(n)) / n)
    rmse = math.sqrt(sum((p[i] - o[i]) ** 2 for i in range(n)) / n)
    esf = [o[i] / p[i] for i in range(n)]
    avg_esf = 10.0 ** (sum(math.log10(o[i] / p[i]) for i in range(n)) / n)
    within = sum(1 for i in range(n)
                 if 1.0 / fold <= p[i] / o[i] <= fold) / float(n)
    return {"afe": afe, "fold_underprediction": 1.0 / afe, "rmse": rmse,
            "esf": esf, "average_esf": avg_esf,
            "within_fold": within, "beyond_fold": 1.0 - within,
            "n": n, "fold": float(fold)}


def clrnt(clint_in_vitro, cl_h=None, fu_blood=None, log_pd=None,
          fu_incubation=None, system="hepatocytes", species="human",
          liver_model="well_stirred", protein=1.0, volume_ratio=0.005,
          cl_plasma=None, fu_plasma=None, blood_plasma_ratio=None,
          charge="neutral", fold=2.0):
    r"""Predict in vivo unbound intrinsic clearance and, given observations,
    score the prediction.

    Parameters
    ----------
    clint_in_vitro : float or sequence
        In vitro :math:`CL_{int}`: ml/min per :math:`10^6` cells for
        hepatocytes, ml/min per mg protein for microsomes.
    cl_h, fu_blood : float or sequence, optional
        Observed hepatic (blood) clearance in ml/min/kg and unbound
        fraction in blood. Supplied, the observed :math:`CL_{int,u}` and the
        accuracy summary come back too.
    log_pd : float or sequence, optional
        :math:`\log P` for basic and neutral drugs, :math:`\log D` for
        acidic ones. Used for equations 1-2 when ``fu_incubation`` is not
        measured.
    fu_incubation : float or sequence, optional
        Measured :math:`f_{u,mic}` or :math:`f_{u,heps}`, which the paper
        prefers "where experimentally determined".
    cl_plasma, fu_plasma, blood_plasma_ratio, charge
        The plasma-to-blood conversion, when the source reports plasma
        values.

    Returns
    -------
    RichResult
        ``estimate`` / ``predicted`` is predicted :math:`CL_{int,u}`;
        ``observed`` and ``accuracy`` appear when observations are given;
        ``fu_incubation``, ``constants`` and ``liver_model`` record what was
        used.

    Examples
    --------
    ::

        r = clrnt(clint_in_vitro=[5.0, 20.0], cl_h=[3.0, 12.0],
                  fu_blood=[0.1, 0.05], log_pd=[2.0, 3.5])
        r["accuracy"]["fold_underprediction"]

    References
    ----------
    Wood, Houston & Hallifax (2017) *Drug Metab Dispos* 45(11), 1178-1188,
    equations 1-8 and the physiological scaling constants of its Methods.
    """
    single = not isinstance(clint_in_vitro, (list, tuple))
    cl_in = [float(clint_in_vitro)] if single else \
        [float(v) for v in clint_in_vitro]
    n = len(cl_in)

    def spread(v, name):
        if v is None:
            return None
        if not isinstance(v, (list, tuple)):
            return [float(v)] * n
        if len(v) != n:
            raise ValueError("clrnt: %s must have one entry per compound"
                             % name)
        return [float(t) for t in v]

    lp = spread(log_pd, "log_pd")
    fu_inc = spread(fu_incubation, "fu_incubation")
    clh = spread(cl_h, "cl_h")
    fub = spread(fu_blood, "fu_blood")
    clp = spread(cl_plasma, "cl_plasma")
    fup = spread(fu_plasma, "fu_plasma")
    rbs = spread(blood_plasma_ratio, "blood_plasma_ratio")

    if clh is None and clp is not None:
        if fup is None:
            raise ValueError("clrnt: plasma clearance needs fu_plasma too")
        clh, fub, rb_used = [], [], []
        for i in range(n):
            a, b, r = blood_from_plasma(clp[i], fup[i],
                                        None if rbs is None else rbs[i],
                                        charge)
            clh.append(a)
            fub.append(b)
            rb_used.append(r)
    else:
        rb_used = None

    if fu_inc is None:
        if lp is None:
            raise ValueError("clrnt: give either fu_incubation or log_pd "
                             "so equations 1-2 can estimate it")
        fu_inc = [fu_microsomes(lp[i], protein) if system == "microsomes"
                  else fu_hepatocytes(lp[i], volume_ratio)
                  for i in range(n)]

    predicted = [scale_to_liver(cl_in[i], fu_inc[i], system, species)
                 for i in range(n)]

    payload = {
        "estimate": predicted[0] if single else predicted,
        "predicted": predicted[0] if single else predicted,
        "fu_incubation": fu_inc[0] if single else fu_inc,
        "system": system,
        "species": species,
        "liver_model": liver_model,
        "constants": dict(CONSTANTS[species]),
        "blood_plasma_ratio": rb_used,
        "note": "predictions of this kind are systematically LOW and the "
                "shortfall grows with in vivo clearance (Wood, Houston & "
                "Hallifax 2017); the accuracy block is how you measure it "
                "on your own data",
        "method": "in vitro to in vivo CLint,u prediction (Wood, Houston & "
                  "Hallifax 2017)",
    }
    if clh is not None and fub is not None:
        obs = [observed_clint_u(clh[i], fub[i], species, None, liver_model)
               for i in range(n)]
        payload["observed"] = obs[0] if single else obs
        payload["cl_h"] = clh[0] if single else clh
        payload["fu_blood"] = fub[0] if single else fub
        payload["accuracy"] = prediction_accuracy(predicted, obs, fold)
    return RichResult(payload=payload)


def cheatsheet():
    return ("clrnt: in vitro to in vivo CLint,u (Wood, Houston & Hallifax "
            "2017). fu in the incubation from eq.1 (microsomes) or eq.2 "
            "(hepatocytes) when unmeasured; scale by PBSF x liver weight "
            "over fu (eq.3) -- 40 mg/g human microsomes, 60 rat, 120e6 "
            "cells/g both, 21.4 g/kg human liver, 40 g/kg rat; observed "
            "from CLh/(fub[1 - CLh/Qh]) (eq.4, well-stirred; parallel tube "
            "also available), Qh 20.7 human, 100 rat. Accuracy by AFE "
            "(eq.5), RMSE (eq.6), ESF = observed/predicted (eq.7) and its "
            "log average (eq.8), plus the 2-fold count. The paper's "
            "finding is that this pipeline UNDERPREDICTS, worse as "
            "clearance rises. The ledger's second citation, Pirmohamed "
            "2019, does not exist.")


# compact aliases
hepatic_clearance_prediction = clrnt
hepatic_clearance = clrnt

# name carried over from the generated stub this replaced
clearance_intrinsic = clrnt
