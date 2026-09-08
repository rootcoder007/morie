# morie.fn -- function file (rootcoder007/morie)
"""Knee-joint sound generation model, patellofemoral crepitus (Rangayyan Sect. 7.7.3).

Book pages read from the typeset PDF: Rangayyan and Krishnan,
*Biomedical Signal Analysis*, 3rd ed., Wiley-IEEE Press, 2024,
Section 7.7.3 "Modeling sound generation in knee joints", pp. 402-404,
after Beverland, Kernohan, McCoy and Mollan, "What is physiological
patellofemoral crepitus?", Proc. XIV Int. Conf. on Medical and
Biological Engineering, pp. 1249-1250, Espoo, 1985.

  "Beverland et al. studied the PPC signals produced during very slow
  movement of the leg (at about 4 deg/s) ... Reproducible series of
  bursts of vibration were recorded in their experiments ... as the
  wheel in the model is slowly rotated clockwise (representing
  extension), it would initially stick to the overlying patella
  (hardboard) due to static friction ... A point would be reached where
  the static friction would be overcome, when the patella would slip and
  the rotation is suddenly reversed ... The mechanical model was shown
  to generate signals similar to those recorded from subjects, thereby
  confirming the stick-slip frictional model for the generation of PPC
  signals."

Section 7.7.3 is entirely descriptive: the book prints NO equation for
this model, and none is invented here.  What the section does specify is
the structure of the signal the model generates -- a reproducible train
of vibration bursts, one burst per slip event, separated by quiet stick
intervals -- and that is what this function measures.

The stick/slip decision is the model's own: the surface sticks while the
tangential force stays below the static-friction limit and slips once it
exceeds it.  With the short-time RMS envelope standing in for the
tangential force and its median standing in for the kinetic (sliding)
level, the sample is in slip when

    envelope(n) > (mu_s / mu_k) * median(envelope),

so the single parameter of the detector is the static-to-kinetic
friction ratio, which is the `force` argument.  A burst is a maximal run
of consecutive slip samples.  This reading is stated here explicitly
because it is a reading: the section names the mechanism, not a
threshold.

Ceiling of the median baseline: once slip occupies more than half the
record the median of the envelope falls INSIDE the bursts, the threshold
rises with them, and no slip is reported.  That is the correct answer for
this model -- a permanently sliding surface has no stick-slip structure
and generates no burst train -- but it does mean the detector is for the
sparse-burst regime of Figure 7.27, not for continuous vibration.  Pass a
`force` ratio below the burst-to-quiet amplitude ratio, or an explicit
`window`, if a record breaks that assumption.
"""

from __future__ import annotations

from math import sqrt

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["rangayyan_knee_joint_sound"]

# 10 ms, short against the burst durations Figure 7.27 shows and long
# enough to smooth one cycle of the vibration itself
_ENVELOPE_SECONDS = 0.010


def _aslist(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def _median(v):
    s = sorted(v)
    m = len(s)
    if m == 0:
        return float("nan")
    if m % 2 == 1:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def rangayyan_knee_joint_sound(vag, fs, force=1.5, window=None):
    """Detect the stick-slip burst train in a PPC/VAG record.

    Parameters
    ----------
    vag : array-like
        The vibration signal recorded over the patella.
    fs : float
        Sampling rate in Hz.
    force : float
        The static-to-kinetic friction ratio mu_s / mu_k of the
        Beverland stick-slip model; must be at least 1, since static
        friction is never below kinetic friction.
    window : int, optional
        Length in samples of the causal short-time RMS window; ten
        milliseconds by default.

    Returns
    -------
    estimate : the burst rate in bursts per second
    slip     : 0/1 per sample, 1 while the model is slipping
    n_bursts : number of slip events
    onsets, offsets : sample indices bounding each burst
    intervals : inter-onset intervals in seconds
    mean_interval : their mean, the stick-slip period
    duty : the fraction of the record spent slipping
    envelope, baseline, threshold : the detector's internals
    """
    x = _aslist(vag)
    N = len(x)
    if N < 3:
        raise ValueError("rangayyan_knee_joint_sound: need at least three samples")
    fs = float(fs)
    if fs <= 0.0:
        raise ValueError("rangayyan_knee_joint_sound: fs must be positive")
    ratio = float(force)
    if not ratio >= 1.0:
        raise ValueError("rangayyan_knee_joint_sound: the friction ratio must be at least 1")
    if window is None:
        w = int(round(_ENVELOPE_SECONDS * fs))
        if w < 1:
            w = 1
    else:
        w = int(window)
        if w < 1:
            raise ValueError("rangayyan_knee_joint_sound: window must be at least one sample")
    if w > N:
        w = N

    # causal short-time RMS envelope, Section 5.6.1
    env = []
    for i in range(N):
        lo = i - w + 1
        if lo < 0:
            lo = 0
        s = 0.0
        for j in range(lo, i + 1):
            s += x[j] * x[j]
        env.append(sqrt(s / (i - lo + 1)))

    baseline = _median(env)
    thr = ratio * baseline
    slip = [1 if env[i] > thr else 0 for i in range(N)]

    onsets = []
    offsets = []
    i = 0
    while i < N:
        if slip[i]:
            j = i
            while j < N and slip[j]:
                j += 1
            onsets.append(i)
            offsets.append(j)
            i = j
        else:
            i += 1
    gaps = [(onsets[i + 1] - onsets[i]) / fs for i in range(len(onsets) - 1)]
    mean_gap = (sum(gaps) / len(gaps)) if gaps else float("nan")
    duty = sum(slip) / float(N)
    rate = len(onsets) / (N / fs)
    return RichResult(
        title="Patellofemoral crepitus stick-slip burst train",
        summary_lines=[("samples", N), ("bursts", len(onsets)), ("bursts/s", rate)],
        payload={
            "estimate": rate,
            "burst_rate": rate,
            "slip": slip,
            "n_bursts": len(onsets),
            "onsets": onsets,
            "offsets": offsets,
            "intervals": gaps,
            "mean_interval": mean_gap,
            "duty": duty,
            "envelope": env,
            "baseline": baseline,
            "threshold": thr,
            "friction_ratio": ratio,
            "window": w,
            "n": N,
            "fs": fs,
            "method": "Rangayyan (2024) Sect. 7.7.3 pp.402-404, Beverland et al. (1985) stick-slip model; burst = a maximal run of the RMS envelope above mu_s/mu_k times its median",
        },
    )


def cheatsheet():
    return "rgkneejt: stick-slip burst train of patellofemoral crepitus, Rangayyan Sect. 7.7.3"
