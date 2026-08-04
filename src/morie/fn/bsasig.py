# morie.fn -- bsasig (rootcoder007/morie)
"""Elementary signals and systems: delta and step functions, convolution, LSI interconnections, signal models.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 27
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from math import cos, exp, fsum, pi, sin
from math import cos, fsum, log, pi, sin, sqrt
from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._rgcore import aslist, gridint
from ._rgcore import gridint
from ._richresult import RichResult

__all__ = [
    'amsig',
    'rangayyan_am_signal',
    'linconv',
    'rangayyan_linear_convolution',
    'fmsig',
    'rangayyan_fm_signal',
    'tvlsi',
    'rangayyan_tvlsi',
    'diracdelta',
    'rangayyan_ch3_dirac_delta_definition',
    'deltaarea',
    'rangayyan_ch3_dirac_delta_unit_area',
    'deltalim',
    'rangayyan_ch3_dirac_delta_limit_form',
    'ustep',
    'rangayyan_ch3_unit_step_continuous',
    'sifting',
    'rangayyan_ch3_sifting_property',
    'contconv',
    'rangayyan_ch3_continuous_convolution',
    'contconvalt',
    'rangayyan_ch3_continuous_convolution_alt',
    'rangayyan_ch3_causal_convolution',
    'rangayyan_ch3_causal_convolution_alt',
    'kdelta',
    'rangayyan_ch3_discrete_delta',
    'stepseq',
    'rangayyan_ch3_discrete_unit_step',
    'rangayyan_ch3_discrete_convolution_causal',
    'rangayyan_ch3_discrete_convolution_causal_alt',
    'sincostest',
    'rangayyan_ch3_test_signal_sin_cos',
    'lsiser',
    'rangayyan_ch3_lsi_series_intermediate',
    'lsisery',
    'rangayyan_ch3_lsi_series_total',
    'lsipar',
    'rangayyan_ch3_lsi_parallel_branch_1',
    'lsipar2',
    'rangayyan_ch3_lsi_parallel_branch_2',
    'lsipary',
    'rangayyan_ch3_lsi_parallel_total',
    'ltiprod',
    'rangayyan_ch3_lti_convolution_property',
    'perconv',
    'rangayyan_ch3_periodic_convolution',
    'rangayyan_ch4_test_signal_three_events',
    'compsig',
    'rangayyan_ch4_composite_signal_in_terms_of_g',
]



# -- rgam: Amplitude-modulated (AM) signal model.
def amsig(x, fc, fs, conventional=False, depth=1.0):
    """Amplitude modulation and synchronous demodulation.

    Rangayyan (2024) Section 5.5.1 gives the AM signal as
        y(t) = x(t) cos(wc t),
    that is, double-sideband SUPPRESSED-carrier modulation, and the
    synchronous demodulator as
        x_d(t) = y(t) cos(wc t) = 0.5 x(t) + 0.5 x(t) cos(2 wc t),
    so that lowpass filtering x_d recovers x/2.

    The placeholder this replaces stated y = A(1 + m(t)) cos(wc t), the
    conventional large-carrier form used in broadcast radio.  That is a
    different model -- it is what allows envelope detection without the
    carrier -- and the book does not use it here.  It is available under
    ``conventional=True`` so the distinction is a choice rather than a
    silent substitution.

    Parameters
    ----------
    x : array-like
        Modulating signal, sampled at ``fs``.
    fc : float
        Carrier frequency in Hz.  Must be below the Nyquist rate.
    fs : float
        Sampling rate in Hz.
    conventional : bool
        Use y = (1 + depth * x) cos(wc t) instead of the book's form.
    depth : float
        Modulation index for the conventional form.
    """
    xs = aslist(x)
    if not xs:
        raise ValueError("need at least one sample")
    fsv, fcv = float(fs), float(fc)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if not 0 < fcv < fsv / 2.0:
        raise ValueError("the carrier must satisfy 0 < fc < fs/2, got "
                         "fc=%g with fs=%g" % (fcv, fsv))
    w = 2.0 * pi * fcv / fsv
    carrier = [cos(w * n) for n in range(len(xs))]
    if conventional:
        y = [(1.0 + depth * v) * c for v, c in zip(xs, carrier)]
    else:
        y = [v * c for v, c in zip(xs, carrier)]
    demod = [v * c for v, c in zip(y, carrier)]
    return RichResult(payload={
        "y": y, "carrier": carrier, "demodulated": demod,
        "fc": fcv, "fs": fsv, "suppressed_carrier": not conventional,
        "baseband_gain": 0.5,
        "image_frequency": 2.0 * fcv,
        "method": "Rangayyan (2024) Section 5.5.1"})


rangayyan_am_signal = amsig  # pre-policy spelling


# -- rgconv: Linear convolution of two finite-length sequences.
def linconv(x, h, causal=True):
    """Linear convolution of two finite-length sequences.

    Rangayyan (2024) eqs. (3.36)-(3.37):
        y(n) = sum_{k=0}^{n} x(k) h(n - k) = sum_{k=0}^{n} h(k) x(n - k),

    with causality assumed, as the book states directly under eq. (3.37).
    Eq. (3.39) reads the same sum the other way -- as a sum of delayed and
    weighted copies of the impulse response, the weights being the input
    samples -- and those copies are returned as ``contributions`` so the
    overlap the book describes in Figure 3.19 is visible.

    The result has Nx + Nh - 1 samples.  Note this is LINEAR convolution:
    the DFT product of eq. (3.87) gives the circular one unless both
    sequences are zero-padded to that length first.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both sequences need at least one sample")
    n, m = len(xs), len(hs)
    y = []
    for k in range(n + m - 1):
        lo, hi = max(0, k - m + 1), min(k, n - 1)
        y.append(fsum(xs[i] * hs[k - i] for i in range(lo, hi + 1)))
    contributions = [[0.0] * (n + m - 1) for _ in range(n)]
    for i, xv in enumerate(xs):
        for j, hv in enumerate(hs):
            contributions[i][i + j] = xv * hv
    swapped = []
    for k in range(n + m - 1):
        lo, hi = max(0, k - n + 1), min(k, m - 1)
        swapped.append(fsum(hs[i] * xs[k - i] for i in range(lo, hi + 1)))
    return RichResult(payload={
        "y": y, "n": n + m - 1, "n_x": n, "n_h": m,
        "contributions": contributions,
        "commutes": max(abs(a - b) for a, b in zip(y, swapped)) <= 1e-12
        * (1 + max(abs(v) for v in y)),
        "causal": bool(causal),
        "method": "Rangayyan (2024) eqs. (3.36)-(3.39)"})


rangayyan_linear_convolution = linconv  # pre-policy spelling


# -- rgfm: Frequency-modulated (FM) signal model for respiratory sounds.
def fmsig(m, fc, fs, kf=1.0, amplitude=1.0):
    """Frequency-modulated signal, and the instantaneous frequency in it.

    Rangayyan (2024) names frequency modulation as a signal model but,
    unlike amplitude modulation in Section 5.5.1, does not print an
    equation for it; the standard definition is used and said so:
        y(t) = A cos(2 pi fc t + 2 pi kf integral_0^t m(tau) d tau),
        f_inst(t) = fc + kf m(t).

    The phase is accumulated by the trapezoidal rule, which is the
    discrete form of the integral above; accumulating m(n) by a plain
    running sum instead biases the phase by half a sample of m at each
    end, which shows up as a slow drift over a long record.

    ``max_instantaneous_frequency`` is reported so a caller can see
    whether the deviation pushes the signal past the Nyquist rate --
    where the model is still well defined but its sampling is not.
    """
    ms = aslist(m)
    if not ms:
        raise ValueError("need at least one sample")
    fsv, fcv = float(fs), float(fc)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    if not 0 < fcv < fsv / 2.0:
        raise ValueError("the carrier must satisfy 0 < fc < fs/2")
    dt = 1.0 / fsv
    phase, acc = [], 0.0
    for i, v in enumerate(ms):
        if i:
            acc += 0.5 * (ms[i] + ms[i - 1]) * dt
        phase.append(2.0 * pi * fcv * i * dt + 2.0 * pi * kf * acc)
    y = [amplitude * cos(p) for p in phase]
    finst = [fcv + kf * v for v in ms]
    return RichResult(payload={
        "y": y, "phase": phase, "instantaneous_frequency": finst,
        "fc": fcv, "fs": fsv, "kf": float(kf),
        "max_instantaneous_frequency": max(finst),
        "min_instantaneous_frequency": min(finst),
        "aliases": max(abs(v) for v in finst) >= fsv / 2.0,
        "method": "standard FM model; Rangayyan (2024) names FM as a "
                  "signal model without printing this equation"})


rangayyan_fm_signal = fmsig  # pre-policy spelling


# -- rgstvar: Time-variant linear system (TV-LSI) characterization.
def tvlsi(x, h):
    """Time-variant linear system: a different impulse response per instant.

    An LSI system is characterized by one impulse response h(n); a
    time-variant one needs h(n, m), the response at output instant n to
    an impulse applied m samples earlier:
        y(n) = sum_m h(n, m) x(n - m).

    Rangayyan (2024) uses time-variant modelling for nonstationary
    signals -- Chapter 8 tracks a system whose parameters change with
    time -- and the point of this function is the contrast with eq.
    (3.36): the convolution sum still holds instant by instant, but the
    kernel is no longer shift-invariant, so no single transfer function
    describes it.  ``shift_invariant`` records whether the supplied
    kernel is in fact constant in n, in which case the system reduces to
    an ordinary LSI filter.

    Parameters
    ----------
    x : array-like
        Input, N samples.
    h : sequence of sequences, or array-like
        Either N rows h(n, .), one per output instant, or a single
        impulse response, which is then used at every instant.
    """
    xs = aslist(x)
    n = len(xs)
    if not n:
        raise ValueError("need at least one sample")
    rows = [aslist(r) for r in h] if h and hasattr(h[0], "__len__") \
        else [aslist(h)] * n
    if len(rows) == 1:
        rows = rows * n
    if len(rows) != n:
        raise ValueError("give one impulse response per sample (%d), or "
                         "one response for all; got %d" % (n, len(rows)))
    y = []
    for i in range(n):
        row = rows[i]
        y.append(fsum(row[mm] * xs[i - mm] for mm in range(len(row))
                      if 0 <= i - mm < n))
    first = rows[0]
    invariant = all(len(r) == len(first)
                    and all(abs(a - b) < 1e-12 for a, b in zip(r, first))
                    for r in rows)
    return RichResult(payload={
        "y": y, "n": n, "kernel_lengths": [len(r) for r in rows],
        "shift_invariant": invariant,
        "method": "time-variant convolution; contrast Rangayyan (2024) "
                  "eq. (3.36)"})


rangayyan_tvlsi = tvlsi  # pre-policy spelling


# -- rng024: Continuous-time Dirac delta function (Rangayyan eq. 3.24).
def diracdelta(t, width=None):
    """Dirac delta evaluated on a time grid.

    Rangayyan (2024) eq. (3.24):
        delta(t) = undefined at t = 0, 0 otherwise.

    A generalized function has no pointwise value at the origin, so the
    honest return is the definition itself: 0 everywhere and None at
    t = 0.  Passing ``width`` instead returns the unit-area rectangular
    pulse of that duration -- the approximating family of Figure 3.10,
    whose limit is the delta -- which is what a numerical caller actually
    needs.  The two are kept in one function so that no caller silently
    treats the rectangle as if it were the delta.
    """
    ts = aslist(t)
    if width is None:
        vals = [None if v == 0.0 else 0.0 for v in ts]
        return RichResult(payload={
            "delta": vals, "t": ts, "undefined_at_zero": True,
            "method": "Rangayyan (2024) eq. (3.24)"})
    w = float(width)
    if w <= 0:
        raise ValueError("width must be positive")
    h = 1.0 / w
    vals = [h if abs(v) <= w / 2.0 else 0.0 for v in ts]
    return RichResult(payload={
        "delta": vals, "t": ts, "width": w, "height": h,
        "undefined_at_zero": False,
        "method": "Rangayyan (2024) eq. (3.24), rectangular approximation"})


rangayyan_ch3_dirac_delta_definition = diracdelta  # pre-policy spelling


# -- rng025: Unit-area property of the Dirac delta (Rangayyan eq. 3.25).
def deltaarea(t=None, values=None, width=None):
    """Verify the unit-area property of a delta approximation.

    Rangayyan (2024) eq. (3.25):
        integral_{-inf}^{inf} delta(t) dt = 1.

    The property is what defines the delta, so the useful computation is
    the check: integrate a candidate approximation and report how far its
    mass is from 1.  With no arguments the exact area 1.0 is returned;
    with ``width`` the rectangular pulse of Figure 3.10 is integrated on
    a fine grid, which must give 1 at any width -- that invariance under
    compression is the point of the figure.
    """
    if values is not None:
        if t is None:
            raise ValueError("give the grid t alongside values")
        area = gridint(values, t)
        return RichResult(payload={
            "area": float(area), "unit_area": abs(area - 1.0) <= 1e-6,
            "method": "Rangayyan (2024) eq. (3.25)"})
    if width is not None:
        w = float(width)
        if w <= 0:
            raise ValueError("width must be positive")
        # panel edges placed on +/- w/2 so no panel straddles the jump;
        # the midpoint rule is then exact for a piecewise-constant pulse.
        n_panels = 800
        span = 2.0 * w
        h = 2.0 * span / n_panels
        edges = [-span + i * h for i in range(n_panels + 1)]
        edges = sorted(set(edges + [-w / 2.0, w / 2.0]))
        area = 0.0
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            mid = 0.5 * (lo_e + hi_e)
            area += (hi_e - lo_e) * ((1.0 / w) if abs(mid) <= w / 2.0 else 0.0)
        return RichResult(payload={
            "area": float(area), "width": w,
            "unit_area": abs(area - 1.0) <= 1e-9,
            "method": "Rangayyan (2024) eq. (3.25)"})
    return RichResult(payload={
        "area": 1.0, "unit_area": True,
        "method": "Rangayyan (2024) eq. (3.25)"})


rangayyan_ch3_dirac_delta_unit_area = deltaarea  # pre-policy spelling


# -- rng026: Dirac delta as a limit of a power function (Rangayyan eq. 3.26).
def deltalim(t, a):
    """The power-function family whose limit is the delta.

    Rangayyan (2024) eq. (3.26):
        delta(t) = 0.5 * lim_{a->0} a |t|^(a-1).

    Figure 3.11 plots this for a = 0.8, 0.4, 0.2.  The exponent a - 1 is
    negative for every a in (0, 1), so the function diverges at t = 0 --
    returned as None there, not as a large finite number.  Its integral
    over any symmetric interval [-L, L] is L^a, which tends to 1 as
    a -> 0 for any fixed L: that is why the limit is the unit-area delta,
    and it is reported so the caller can see the convergence.
    """
    ts = aslist(t)
    av = float(a)
    if av <= 0:
        raise ValueError("a must be positive")
    vals = [None if v == 0.0 else 0.5 * av * abs(v) ** (av - 1.0)
            for v in ts]
    lim = max((abs(v) for v in ts), default=1.0) or 1.0
    return RichResult(payload={
        "values": vals, "t": ts, "a": av,
        "area_symmetric": lim ** av,
        "half_width": lim,
        "method": "Rangayyan (2024) eq. (3.26)"})


rangayyan_ch3_dirac_delta_limit_form = deltalim  # pre-policy spelling


# -- rng027: Continuous-time unit step function (Rangayyan eq. 3.27).
def ustep(t, shift=0.0):
    """Continuous-time unit step u(t).

    Rangayyan (2024) eq. (3.27):
        u(t) = 1 for t > 0, 0 otherwise.

    Note the strict inequality: u(0) = 0 in this book, not 0.5 and not 1.
    The discrete step of eq. (3.35) uses n >= 0 instead, so the two
    disagree at the origin -- they are separate definitions, not one
    sampled from the other.  The book also notes the delta is the
    derivative of u.
    """
    ts = aslist(t)
    s = float(shift)
    return RichResult(payload={
        "u": [1.0 if v - s > 0.0 else 0.0 for v in ts], "t": ts,
        "shift": s, "value_at_origin": 0.0,
        "method": "Rangayyan (2024) eq. (3.27)"})


rangayyan_ch3_unit_step_continuous = ustep  # pre-policy spelling


# -- rng028: Sifting property of the Dirac delta (Rangayyan eq. 3.28).
def sifting(x, t0, lower, upper):
    """Sift the value of x at t0 out of an interval.

    Rangayyan (2024) eq. (3.28):
        integral_{T1}^{T2} x(t) delta(t - to) dt
            = x(to)  if T1 < to < T2,
            = 0      otherwise.

    The inequalities are strict at both ends: an impulse sitting exactly
    on a limit of integration contributes nothing under this definition,
    which is why ``inside`` is reported alongside the value.

    Parameters
    ----------
    x : callable
        The function being sifted; must be continuous at t0.
    t0 : float
        Location of the impulse.
    lower, upper : float
        Interval of integration.
    """
    if not callable(x):
        raise TypeError("x must be a callable continuous at t0")
    lo, hi, t = float(lower), float(upper), float(t0)
    if hi <= lo:
        raise ValueError("upper must exceed lower")
    inside = lo < t < hi
    return RichResult(payload={
        "value": float(x(t)) if inside else 0.0, "inside": inside,
        "t0": t, "lower": lo, "upper": hi,
        "method": "Rangayyan (2024) eq. (3.28)"})


rangayyan_ch3_sifting_property = sifting  # pre-policy spelling


# -- rng030: Continuous-time convolution (Rangayyan eq. 3.30).
def contconv(x, h, dt=1.0, t=None):
    """Convolution of an input with an impulse response, tabulated form.

    Rangayyan (2024) eq. (3.30):
        y(t) = integral x(tau) h(t - tau) d tau.

    Tabulated on a uniform grid of spacing dt this becomes the discrete
    convolution scaled by dt -- the dt is what makes it an integral
    rather than eq. (3.36)'s sum, and dropping it is the usual way a
    continuous-time convolution comes out wrong by a factor of the
    sampling interval.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both signals need at least one sample")
    step = float(dt)
    if t is not None:
        ts = aslist(t)
        if len(ts) != len(xs):
            raise ValueError("t must match x in length")
        if len(ts) > 1:
            step = ts[1] - ts[0]
    if step <= 0:
        raise ValueError("dt must be positive")
    n, m = len(xs), len(hs)
    y = []
    for k in range(n + m - 1):
        lo = max(0, k - m + 1)
        hi = min(k, n - 1)
        y.append(sum(xs[i] * hs[k - i] for i in range(lo, hi + 1)) * step)
    t_out = [i * step for i in range(len(y))]
    if t is not None and len(ts):
        t_out = [ts[0] + i * step for i in range(len(y))]
    return RichResult(payload={
        "y": y, "t": t_out, "dt": step, "n": n, "m": m,
        "integral": gridint(y, t_out) if len(y) > 1 else 0.0,
        "method": "Rangayyan (2024) eq. (3.30)"})


rangayyan_ch3_continuous_convolution = contconv  # pre-policy spelling


# -- rng031: Commuted form of continuous-time convolution (Rangayyan eq. 3.31).
def contconvalt(x, h, dt=1.0, t=None):
    """Convolution with the roles of x and h swapped.

    Rangayyan (2024) eq. (3.31):
        y(t) = integral h(tau) x(t - tau) d tau,

    which the book gives as an equivalent result to eq. (3.30).  It is
    computed here the other way round and compared against eq. (3.30)
    rather than merely asserted equivalent: ``max_difference`` is the
    largest discrepancy between the two orders, and is zero up to
    rounding for any pair of finite sequences.
    """
    xs, hs = aslist(x), aslist(h)
    swapped = contconv(hs, xs, dt=dt, t=t)
    direct = contconv(xs, hs, dt=dt, t=t)
    a, b = swapped["y"], direct["y"]
    diff = max((abs(p - q) for p, q in zip(a, b)), default=0.0)
    out = dict(swapped)
    out["max_difference"] = diff
    out["commutes"] = diff <= 1e-12 * max(
        1.0, max((abs(v) for v in b), default=1.0))
    out["method"] = "Rangayyan (2024) eq. (3.31)"
    return RichResult(payload=out)


rangayyan_ch3_continuous_convolution_alt = contconvalt  # pre-policy spelling


# -- rng032: Causal continuous-time convolution form (lower limit 0, upper limit t).
def rangayyan_ch3_causal_convolution(x, h, dt=1.0):
    r"""Causal continuous convolution
    :math:`y(t) = \int_0^t x(\tau) h(t-\tau)\, d\tau`.

    Evaluated on the sample grid by the trapezoidal rule for every
    upper limit t, so ``y[i]`` approximates the integral up to
    ``t = i * dt``. Both signals are treated as zero for negative
    time, which is what makes the limits 0 and t rather than
    :math:`\pm\infty`.

    Parameters
    ----------
    x, h : array-like, shape (m,)
        Samples of the input and the impulse response on a uniform
        grid starting at t = 0.
    dt : float, default 1.0
        Sampling interval.

    Returns
    -------
    RichResult
        keys: ``y`` (m,), ``t`` (m,), ``dt``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3 (convolution for causal LTI systems).
    """
    x = np.asarray(x, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    if x.size != h.size:
        raise ValueError("x and h must be sampled on the same grid.")
    if x.size < 2:
        raise ValueError("need at least 2 samples.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")

    m = x.size
    y = np.empty(m)
    for i in range(m):
        integrand = x[: i + 1] * h[i::-1]
        y[i] = np.trapezoid(integrand, dx=dt) if i > 0 else 0.0

    return RichResult(
        payload={
            "y": y,
            "t": np.arange(m) * dt,
            "dt": dt,
            "method": "Causal convolution integral y(t) = int_0^t x(tau) h(t-tau) dtau (trapezoid)",
        }
    )


# -- rng033: Equivalent causal continuous-time convolution with swapped arguments.
def rangayyan_ch3_causal_convolution_alt(x, h, dt=1.0):
    r"""Commuted form :math:`y(t) = \int_0^t h(\tau) x(t-\tau)\, d\tau`.

    Identical to :func:`rangayyan_ch3_causal_convolution` with the
    arguments swapped; the pair exists in the text to make the
    commutativity of convolution explicit.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    out = rangayyan_ch3_causal_convolution(h, x, dt=dt)
    payload = dict(out)
    payload["method"] = "Causal convolution integral, commuted: y(t) = int_0^t h(tau) x(t-tau) dtau"
    return RichResult(payload=payload)


# -- rng034: Discrete-time unit impulse function (Rangayyan eq. 3.34).
def kdelta(n, shift=0, amplitude=1.0):
    """Discrete-time unit impulse delta(n).

    Rangayyan (2024) eq. (3.34):
        delta(n) = 1 if n = 0, 0 otherwise.

    Unlike the continuous delta of eq. (3.24) this is an ordinary
    sequence with a finite value at the origin, so it can be evaluated
    rather than only approximated.  Figure 3.13 shows the shifted and
    scaled versions ``shift`` and ``amplitude`` produce.

    Parameters
    ----------
    n : int or sequence of int
        Sample indices to evaluate at; an integer N is read as the range
        0, 1, ..., N-1.
    shift : int
        Location of the impulse (the n0 of delta(n - n0)).
    amplitude : float
        Scale factor.
    """
    if isinstance(n, int):
        idx = list(range(n))
    else:
        idx = [int(v) for v in n]
    s, a = int(shift), float(amplitude)
    return RichResult(payload={
        "delta": [a if i == s else 0.0 for i in idx], "n": idx,
        "shift": s, "amplitude": a,
        "method": "Rangayyan (2024) eq. (3.34)"})


rangayyan_ch3_discrete_delta = kdelta  # pre-policy spelling


# -- rng035: Discrete-time unit step function (Rangayyan eq. 3.35).
def stepseq(n, shift=0):
    """Discrete-time unit step u(n).

    Rangayyan (2024) eq. (3.35):
        u(n) = 1 for n >= 0, 0 otherwise.

    The inequality is non-strict here, so u(0) = 1 -- the opposite of the
    continuous step of eq. (3.27), where u(0) = 0.  The first difference
    of this sequence is the discrete impulse of eq. (3.34), which is
    returned as a cross-check.
    """
    if isinstance(n, int):
        idx = list(range(n))
    else:
        idx = [int(v) for v in n]
    s = int(shift)
    u = [1.0 if i - s >= 0 else 0.0 for i in idx]
    diff = [u[0]] + [u[i] - u[i - 1] for i in range(1, len(u))]
    return RichResult(payload={
        "u": u, "n": idx, "shift": s, "first_difference": diff,
        "value_at_origin": 1.0,
        "method": "Rangayyan (2024) eq. (3.35)"})


rangayyan_ch3_discrete_unit_step = stepseq  # pre-policy spelling


# -- rng036: Discrete-time causal convolution sum.
def rangayyan_ch3_discrete_convolution_causal(x, h, n=None):
    r"""Causal discrete convolution :math:`y(n) = \sum_{k=0}^{n} x(k) h(n-k)`.

    The sum runs only over past and present inputs, so ``y[n]`` never
    depends on ``x[n+1]`` and beyond -- the definition of a causal
    system.

    Parameters
    ----------
    x, h : array-like
        Input signal and impulse response, both taken as zero for
        negative indices.
    n : int, optional
        Return only ``y(n)`` (a float) instead of the whole sequence.

    Returns
    -------
    RichResult
        keys: ``y`` (full sequence of length len(x) + len(h) - 1),
        ``value`` (y(n) when n given, else None), ``n``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3 (linear time-invariant systems and the
    causal convolution sum).
    """
    x = np.asarray(x, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    if x.size == 0 or h.size == 0:
        raise ValueError("x and h must be non-empty.")
    y = np.convolve(x, h)

    value = None
    if n is not None:
        n = int(n)
        if not 0 <= n < y.size:
            raise ValueError(f"n must lie in [0, {y.size - 1}], got {n}.")
        value = float(y[n])

    return RichResult(
        payload={
            "y": y,
            "value": value,
            "n": n,
            "method": "Causal discrete convolution sum y(n) = sum_k x(k) h(n-k)",
        }
    )


# -- rng037: Equivalent discrete-time causal convolution with swapped arguments.
def rangayyan_ch3_discrete_convolution_causal_alt(x, h, n=None):
    r"""Commuted form :math:`y(n) = \sum_{k=0}^{n} h(k) x(n-k)`.

    Convolution is commutative, so this returns exactly the same
    sequence as :func:`rangayyan_ch3_discrete_convolution_causal` with
    the arguments swapped -- the identity is the point of the pair in
    the text, and the test asserts it.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    out = rangayyan_ch3_discrete_convolution_causal(h, x, n=n)
    payload = dict(out)
    payload["method"] = "Causal discrete convolution, commuted: y(n) = sum_k h(k) x(n-k)"
    return RichResult(payload=payload)


# -- rng038: Synthetic test signal: sum of a sine and a cosine..
def sincostest(n=None, f1=1.0, f2=2.0, a1=1.0, a2=1.0, fs=100.0,
               duration=1.0):
    """Synthetic test signal, a sine plus a cosine.

        x(t) = a1 sin(2 pi f1 t) + a2 cos(2 pi f2 t)

    The book's standard exercise signal: two known components at known
    amplitudes, so any transform, filter or spectral estimate applied to
    it has an answer that can be checked by hand rather than eyeballed.

    Returned with the time base, because a test signal without its
    sampling rate cannot be checked against anything.
    """
    fsv = float(fs)
    if fsv <= 0:
        raise ValueError("fs must be positive")
    for f in (f1, f2):
        if abs(float(f)) >= fsv / 2.0:
            raise ValueError("component at %g Hz is at or above the "
                             "Nyquist frequency %g Hz" % (f, fsv / 2.0))
    if n is not None:
        N = int(n)
    else:
        d = float(duration)
        if d <= 0:
            raise ValueError("duration must be positive")
        N = int(round(d * fsv))
    if N < 2:
        raise ValueError("need at least two samples")
    t = [i / fsv for i in range(N)]
    x = [float(a1) * sin(2.0 * pi * float(f1) * v)
         + float(a2) * cos(2.0 * pi * float(f2) * v) for v in t]
    return RichResult(payload={
        "x": x, "t": t, "n": N, "fs": fsv,
        "f1": float(f1), "f2": float(f2),
        "a1": float(a1), "a2": float(a2),
        "components_are_known_by_construction": True,
        "method": "Rangayyan (2024) Ch. 3 (synthetic test signal)"})


rangayyan_ch3_test_signal_sin_cos = sincostest  # pre-policy spelling


# -- rng041: Intermediate output of the first LSI system in a series cascade..
def lsiser(x, h1, h2):
    """Two LSI systems in series, and the single system equivalent to them.

    Rangayyan (2024) eqs. (3.43)-(3.45):
        s(n) = x(n) * h1(n)                                      (3.43)
        y(n) = s(n) * h2(n) = x(n) * h1(n) * h2(n) = x(n) * h(n)  (3.44)
        h(n) = h1(n) * h2(n)                                      (3.45)

    The three equations are one method, so they are one function: the
    intermediate s(n) of eq. (3.43), the output y(n), and the combined
    impulse response h(n) all come back together.  The equivalence in eq.
    (3.44) is checked rather than asserted -- ``max_difference`` compares
    filtering twice against filtering once with h(n).
    """
    xs = aslist(x)
    a, b = aslist(h1), aslist(h2)
    if not xs or not a or not b:
        raise ValueError("input and both impulse responses need samples")

    def conv(p, q):
        out = []
        for k in range(len(p) + len(q) - 1):
            lo, hi = max(0, k - len(q) + 1), min(k, len(p) - 1)
            out.append(fsum(p[i] * q[k - i] for i in range(lo, hi + 1)))
        return out

    s = conv(xs, a)
    y = conv(s, b)
    h = conv(a, b)
    direct = conv(xs, h)
    gap = max(abs(u - v) for u, v in zip(y, direct))
    return RichResult(payload={
        "s": s, "y": y, "h": h, "y_via_combined": direct,
        "max_difference": gap,
        "equivalent": gap <= 1e-9 * (1 + max(abs(v) for v in y)),
        "method": "Rangayyan (2024) eqs. (3.43)-(3.45)"})


rangayyan_ch3_lsi_series_intermediate = lsiser  # pre-policy spelling


# -- rng042: Output of two LSI systems in series equals input convolved with combined response..
def lsisery(x, h1, h2):
    """Output of two LSI systems in series.

    Rangayyan (2024) eq. (3.44):
        y(n) = s(n) * h2(n) = x(n) * h1(n) * h2(n) = x(n) * h(n),

    with h(n) = h1(n) * h2(n) from eq. (3.45).  The three spellings are
    one computation, so this reads the output off :func:`lsiser` rather
    than convolving a second time -- and returns the combined h as well,
    since the content of eq. (3.44) is precisely that the cascade IS a
    single filter.
    """
    r = lsiser(x, h1, h2)
    return RichResult(payload={
        "y": r["y"], "h": r["h"], "s": r["s"],
        "equivalent": r["equivalent"], "max_difference": r["max_difference"],
        "method": "Rangayyan (2024) eqs. (3.44)-(3.45)"})


rangayyan_ch3_lsi_series_total = lsisery  # pre-policy spelling


# -- rng044: Output of the first branch in a parallel LSI configuration..
def lsipar(x, h1, h2):
    """Two LSI systems in parallel, and the single system equivalent.

    Rangayyan (2024) eqs. (3.46)-(3.49):
        s1(n) = x(n) * h1(n)                                     (3.46)
        s2(n) = x(n) * h2(n)                                     (3.47)
        y(n)  = s1(n) + s2(n) = x(n) * [h1(n) + h2(n)]           (3.48)
        h(n)  = h1(n) + h2(n)                                    (3.49)

    Four equations describing one structure, so one function returns all
    of them.  The branch responses are zero-padded to a common length
    before being added: adding sequences of different lengths by
    truncation would silently drop the tail of the longer filter.
    """
    xs = aslist(x)
    a, b = aslist(h1), aslist(h2)
    if not xs or not a or not b:
        raise ValueError("input and both impulse responses need samples")

    def conv(p, q):
        out = []
        for k in range(len(p) + len(q) - 1):
            lo, hi = max(0, k - len(q) + 1), min(k, len(p) - 1)
            out.append(fsum(p[i] * q[k - i] for i in range(lo, hi + 1)))
        return out

    m = max(len(a), len(b))
    h = [(a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0)
         for i in range(m)]
    s1, s2 = conv(xs, a), conv(xs, b)
    ny = max(len(s1), len(s2))
    y = [(s1[i] if i < len(s1) else 0.0) + (s2[i] if i < len(s2) else 0.0)
         for i in range(ny)]
    direct = conv(xs, h)
    gap = max(abs(u - v) for u, v in zip(y, direct))
    return RichResult(payload={
        "s1": s1, "s2": s2, "y": y, "h": h, "y_via_combined": direct,
        "max_difference": gap,
        "equivalent": gap <= 1e-9 * (1 + max(abs(v) for v in y)),
        "method": "Rangayyan (2024) eqs. (3.46)-(3.49)"})


rangayyan_ch3_lsi_parallel_branch_1 = lsipar  # pre-policy spelling


# -- rng045: Output of the second branch in a parallel LSI configuration..
def lsipar2(x, h2):
    """Output of the second branch of a parallel LSI pair.

    Rangayyan (2024) eq. (3.47):  s2(n) = x(n) * h2(n).

    Identical in form to eq. (3.46) for the first branch -- that is the
    point of a parallel structure, both branches see the same input --
    so this is the same convolution applied to the other filter.
    """
    xs, b = aslist(x), aslist(h2)
    if not xs or not b:
        raise ValueError("input and impulse response need samples")
    out = []
    for k in range(len(xs) + len(b) - 1):
        lo, hi = max(0, k - len(b) + 1), min(k, len(xs) - 1)
        out.append(fsum(xs[i] * b[k - i] for i in range(lo, hi + 1)))
    return RichResult(payload={
        "s2": out, "n": len(out),
        "method": "Rangayyan (2024) eq. (3.47)"})


rangayyan_ch3_lsi_parallel_branch_2 = lsipar2  # pre-policy spelling


# -- rng046: Output of two LSI systems in parallel equals input convolved with sum of responses..
def lsipary(x, h1, h2):
    """Output of two LSI systems in parallel.

    Rangayyan (2024) eqs. (3.48)-(3.49):
        y(n) = s1(n) + s2(n) = x(n) * [h1(n) + h2(n)] = x(n) * h(n),
        h(n) = h1(n) + h2(n).

    The parallel counterpart of eq. (3.44): here the impulse responses
    ADD where a cascade convolves them.  Both routes to y are computed
    inside :func:`lsipar` and compared, so the equivalence is measured.
    """
    r = lsipar(x, h1, h2)
    return RichResult(payload={
        "y": r["y"], "h": r["h"], "s1": r["s1"], "s2": r["s2"],
        "equivalent": r["equivalent"], "max_difference": r["max_difference"],
        "method": "Rangayyan (2024) eqs. (3.48)-(3.49)"})


rangayyan_ch3_lsi_parallel_total = lsipary  # pre-policy spelling


# -- rng051: LTI convolution maps to multiplication in s-domain and frequency domain..
def ltiprod(x, h, s=None, omega=None, dt=1.0):
    """Convolution in time is multiplication in the s and omega domains.

    Rangayyan (2024) eq. (3.53):
        if y(t) = x(t) * h(t),
        then Y(s) = X(s) H(s)  and  Y(omega) = X(omega) H(omega),

    with the Laplace transform of eq. (3.50) evaluated on a
    finite-duration causal signal.  Setting s = j omega recovers the
    frequency-domain statement, which is why one function covers both:
    the omega form is the s form on the imaginary axis.

    Parameters
    ----------
    x, h : array-like
        Sampled signals, uniform spacing ``dt``.
    s : complex or sequence, optional
        Laplace variable.
    omega : float or sequence, optional
        Frequency in rad/s; equivalent to s = j omega.
    dt : float
        Sampling interval; the transforms are integrals, so each carries
        a factor dt.
    """
    xs, hs = aslist(x), aslist(h)
    if not xs or not hs:
        raise ValueError("both signals need at least one sample")
    if (s is None) == (omega is None):
        raise ValueError("give exactly one of s, omega")
    step = float(dt)
    if step <= 0:
        raise ValueError("dt must be positive")
    if s is not None:
        scalar = isinstance(s, (int, float, complex))
        pts = [complex(s)] if scalar else [complex(v) for v in s]
    else:
        scalar = isinstance(omega, (int, float))
        ws = [float(omega)] if scalar else [float(v) for v in omega]
        pts = [complex(0.0, w) for w in ws]
    y = []
    for k in range(len(xs) + len(hs) - 1):
        lo, hi = max(0, k - len(hs) + 1), min(k, len(xs) - 1)
        y.append(fsum(xs[i] * hs[k - i] for i in range(lo, hi + 1)) * step)

    def lap(sig, sv):
        acc = 0j
        for i, v in enumerate(sig):
            t = i * step
            e = complex(cos(-sv.imag * t), sin(-sv.imag * t))
            acc += v * exp(-sv.real * t) * e
        return acc * step

    Y = [lap(y, p) for p in pts]
    X = [lap(xs, p) for p in pts]
    H = [lap(hs, p) for p in pts]
    prod = [a * b for a, b in zip(X, H)]
    gap = max(abs(a - b) for a, b in zip(Y, prod))
    return RichResult(payload={
        "y": y, "Y": Y[0] if scalar else Y, "X": X[0] if scalar else X,
        "H": H[0] if scalar else H, "XH": prod[0] if scalar else prod,
        "s": (pts[0] if scalar else pts),
        "max_difference": gap,
        "holds": gap <= 1e-8 * (1 + max(abs(v) for v in prod)),
        "method": "Rangayyan (2024) eqs. (3.50), (3.53)"})


rangayyan_ch3_lti_convolution_property = ltiprod  # pre-policy spelling


# -- rng079: Circular (periodic) convolution of two N-periodic discrete signals..
def perconv(x, h, npoints=None):
    """Periodic (circular) convolution of two N-periodic signals.

    Rangayyan (2024) eq. (3.90):
        y_p(n) = sum_{k=0}^{N-1} x_p(k) h_p[(n - k) mod N].

    This is the same equation :func:`circconv` implements, so it
    delegates rather than carrying a second copy -- two implementations
    of one equation is how the two drift apart.  The book defines it only
    for signals of equal period, and Figures 3.40-3.42 show what goes
    wrong when it is used where linear convolution was meant.
    """
    from .bsaxfrm import circconv

    r = circconv(x, h, npoints=npoints)
    out = dict(r)
    out["method"] = "Rangayyan (2024) eq. (3.90)"
    return RichResult(payload=out)


rangayyan_ch3_periodic_convolution = perconv  # pre-policy spelling


# -- rng223: Rangayyan Ch. 4 synthetic three-event test signal (Eq. 4.51).
def rangayyan_ch4_test_signal_three_events(n=36):
    r"""Synthetic signal of three scaled repetitions of one basic pattern.

    .. math::

        x(n) = 3\delta(n-5) + 2\delta(n-6) + \delta(n-7)
             + 1.5\delta(n-16) + \delta(n-17) + 0.5\delta(n-18)
             + 0.75\delta(n-26) + 0.5\delta(n-27) + 0.25\delta(n-28)

    (Eq. 4.51), equivalently :math:`x(n) = g(n-5) + 0.5\,g(n-16) +
    0.25\,g(n-26)` with the basic pattern :math:`g(n) = 3\delta(n) +
    2\delta(n-1) + \delta(n-2)` (Eqs. 4.52-4.53). The book uses it to
    illustrate matched filtering: the matched filter's output peaks at
    the three event locations with amplitudes in the 1 : 0.5 : 0.25
    ratio. This replaces a placeholder that computed a KS statistic on
    the length argument.

    Parameters
    ----------
    n : int, default 36
        Signal length; must cover the last event sample (28).

    Returns
    -------
    RichResult
        keys: ``signal`` (length n), ``pattern`` (g), ``onsets``,
        ``amplitudes``, ``n``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis*, 3rd edn.
    Wiley-IEEE. Ch. 4, Eqs. (4.51)-(4.53), p. 240.
    """
    n = int(n)
    if n < 29:
        raise ValueError(f"n must be at least 29 to hold the third event, got {n}.")
    g = np.array([3.0, 2.0, 1.0])
    onsets = [5, 16, 26]
    amps = [1.0, 0.5, 0.25]
    x = np.zeros(n)
    for o, a in zip(onsets, amps):
        x[o : o + 3] += a * g
    return RichResult(
        payload={
            "signal": x,
            "pattern": g,
            "onsets": onsets,
            "amplitudes": amps,
            "n": n,
            "method": "Rangayyan Ch.4 three-event test signal (Eq. 4.51)",
        }
    )


# -- rng225: Composite test signal expressed in terms of three delayed scaled copies of g(n)..
def compsig(g, shifts, scales=None, n=None):
    """Composite signal built from delayed, scaled copies of a pattern.

        x(n) = sum_k a_k g(n - d_k)

    Chapter 4's construction for testing a matched filter: the filter
    matched to g should produce a peak at each d_k with height
    proportional to a_k, so the ground truth is known exactly.

    Copies that overlap ADD, which is the point of the exercise -- a
    matched filter resolves overlapping instances only while they stay
    further apart than the pattern is long, and the overlap count is
    returned so a test can tell whether it is exercising that limit.
    """
    gs = aslist(g)
    ds = [int(v) for v in aslist(shifts)]
    if not gs:
        raise ValueError("the pattern needs at least one sample")
    if not ds:
        raise ValueError("give at least one shift")
    if any(v < 0 for v in ds):
        raise ValueError("shifts cannot be negative")
    a = [1.0] * len(ds) if scales is None else aslist(scales)
    if len(a) != len(ds):
        raise ValueError("give one scale per shift, or none")
    N = int(n) if n is not None else max(ds) + len(gs)
    if N < max(ds) + len(gs):
        raise ValueError("n is too short to hold every shifted copy")
    x = [0.0] * N
    for amp, d in zip(a, ds):
        for i, v in enumerate(gs):
            x[d + i] += amp * v
    overlaps = sum(1 for i in range(len(ds)) for j in range(i + 1, len(ds))
                   if abs(ds[i] - ds[j]) < len(gs))
    return RichResult(payload={
        "x": x, "n": N, "pattern": list(gs), "shifts": ds, "scales": a,
        "pattern_length": len(gs), "n_copies": len(ds),
        "overlapping_pairs": overlaps, "copies_add": True,
        "peaks_expected_at": [d + len(gs) - 1 for d in ds],
        "method": "Rangayyan (2024) Ch. 4 (composite test signal)"})


rangayyan_ch4_composite_signal_in_terms_of_g = compsig  # pre-policy spelling


_CHEATSHEET = [
    'rgam: Amplitude-modulated (AM) signal model.',
    'rgconv: Linear convolution of two finite-length sequences.',
    'rgfm: Frequency-modulated (FM) signal model for respiratory sounds.',
    'rgstvar: Time-variant linear system (TV-LSI) characterization.',
    'rng024: Continuous-time Dirac delta function (Rangayyan eq. 3.24).',
    'rng025: Unit-area property of the Dirac delta (Rangayyan eq. 3.25).',
    'rng026: Dirac delta as a limit of a power function (Rangayyan eq. 3.26).',
    'rng027: Continuous-time unit step function (Rangayyan eq. 3.27).',
    'rng028: Sifting property of the Dirac delta (Rangayyan eq. 3.28).',
    'rng030: Continuous-time convolution (Rangayyan eq. 3.30).',
    'rng031: Commuted form of continuous-time convolution (Rangayyan eq. 3.31).',
    'rng032: Causal continuous-time convolution form (lower limit 0, upper limit t).',
    'rng033: Equivalent causal continuous-time convolution with swapped arguments.',
    'rng034: Discrete-time unit impulse function (Rangayyan eq. 3.34).',
    'rng035: Discrete-time unit step function (Rangayyan eq. 3.35).',
    'rng036: Discrete-time causal convolution sum.',
    'rng037: Equivalent discrete-time causal convolution with swapped arguments.',
    'synthetic sine-plus-cosine test signal',
    'rng041: Intermediate output of the first LSI system in a series cascade..',
    'rng042: Output of two LSI systems in series equals input convolved with combined response..',
    'rng044: Output of the first branch in a parallel LSI configuration..',
    'rng045: Output of the second branch in a parallel LSI configuration..',
    'rng046: Output of two LSI systems in parallel equals input convolved with sum of responses..',
    'rng051: LTI convolution maps to multiplication in s-domain and frequency domain..',
    'rng079: Circular (periodic) convolution of two N-periodic discrete signals..',
    'rng223: Rangayyan Ch. 4 synthetic three-event test signal (Eq. 4.51).',
    'composite signal of delayed scaled patterns',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
