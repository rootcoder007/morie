# morie.fn -- bsasig (rootcoder007/morie)
"""Elementary signals and systems: delta and step functions, convolution, LSI interconnections, signal models.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 27
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from . import _array_core as np
from . import _stats_core as stats
from ._rgcore import aslist
from ._rgcore import aslist, gridint
from ._rgcore import gridint
from ._richresult import RichResult

__all__ = [
    'rangayyan_am_signal',
    'rangayyan_linear_convolution',
    'rangayyan_fm_signal',
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
    'rangayyan_ch3_test_signal_sin_cos',
    'rangayyan_ch3_lsi_series_intermediate',
    'rangayyan_ch3_lsi_series_total',
    'rangayyan_ch3_lsi_parallel_branch_1',
    'rangayyan_ch3_lsi_parallel_branch_2',
    'rangayyan_ch3_lsi_parallel_total',
    'rangayyan_ch3_lti_convolution_property',
    'rangayyan_ch3_periodic_convolution',
    'rangayyan_ch4_test_signal_three_events',
    'rangayyan_ch4_composite_signal_in_terms_of_g',
]


# -- rgam: Amplitude-modulated (AM) signal model.
def rangayyan_am_signal(t, fc, m_t, Ac):
    """
    Amplitude-modulated (AM) signal model

    Formula: t(t) = A_c*(1 + m(t))*cos(2*pi*f_c*t)

    Parameters
    ----------
    t : array-like
        Input data.
    fc : array-like
        Input data.
    m_t : array-like
        Input data.
    Ac : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: am_signal

    References
    ----------
    Rangayyan Ch 5.5.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Amplitude-modulated (AM) signal model"})


# -- rgconv: Linear convolution of two finite-length sequences.
def rangayyan_linear_convolution(x, h):
    """
    Linear convolution of two finite-length sequences

    Formula: y[n] = sum_{k=-inf}^{inf} x[k] * h[n-k]

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Rangayyan Ch 3.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Linear convolution of two finite-length sequences"}
    )


# -- rgfm: Frequency-modulated (FM) signal model for respiratory sounds.
def rangayyan_fm_signal(t, f0, m_t, kf):
    """
    Frequency-modulated (FM) signal model for respiratory sounds

    Formula: t(t) = A*cos(2*pi*f0*t + k_f*integral m(tau)d(tau))

    Parameters
    ----------
    t : array-like
        Input data.
    f0 : array-like
        Input data.
    m_t : array-like
        Input data.
    kf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fm_signal

    References
    ----------
    Rangayyan Ch 7.7.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Frequency-modulated (FM) signal model for respiratory sounds",
        }
    )


# -- rgstvar: Time-variant linear system (TV-LSI) characterization.
def rangayyan_tvlsi(x, y, fs, window):
    """
    Time-variant linear system (TV-LSI) characterization

    Formula: TV impulse response h(t,tau); spectrogram = |H_tv(t,f)|^2

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    fs : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tv_spectrum, t, freqs

    References
    ----------
    Rangayyan Ch 8.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Time-variant linear system (TV-LSI) characterization"}
    )


# compact alias per ledger/NAMING.md
rangayyantvlsi = rangayyan_tvlsi


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
def rangayyan_ch3_test_signal_sin_cos(t, cdf=None):
    """
    Synthetic test signal: sum of a sine and a cosine.

    Formula: x(t) = 5 sin(2*pi*2*t) + 2 cos(2*pi*3*t)

    Parameters
    ----------
    t : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.40, p. 112
    """
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n < 2:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Synthetic test signal: sum of a sine and a cosine.",
            }
        )
    x_sorted = np.sort(t)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(t), scale=np.std(t, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(
        payload={
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n": n,
            "method": "Synthetic test signal: sum of a sine and a cosine.",
        }
    )


# -- rng041: Intermediate output of the first LSI system in a series cascade..
def rangayyan_ch3_lsi_series_intermediate(x, h_1, n):
    """
    Intermediate output of the first LSI system in a series cascade.

    Formula: s(n) = x(n) * h_1(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_1 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.43, p. 115
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Intermediate output of the first LSI system in a series cascade.",
        }
    )


# -- rng042: Output of two LSI systems in series equals input convolved with combined response..
def rangayyan_ch3_lsi_series_total(x, h_1, h_2, n):
    """
    Output of two LSI systems in series equals input convolved with combined response.

    Formula: y(n) = s(n) * h_2(n) = x(n) * h_1(n) * h_2(n) = x(n) * h(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_1 : array-like
        Input data.
    h_2 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.44, p. 115
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Output of two LSI systems in series equals input convolved with combined response.",
        }
    )


# -- rng044: Output of the first branch in a parallel LSI configuration..
def rangayyan_ch3_lsi_parallel_branch_1(x, h_1, n):
    """
    Output of the first branch in a parallel LSI configuration.

    Formula: s_1(n) = x(n) * h_1(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_1 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.46, p. 116
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Output of the first branch in a parallel LSI configuration.",
        }
    )


# -- rng045: Output of the second branch in a parallel LSI configuration..
def rangayyan_ch3_lsi_parallel_branch_2(x, h_2, n):
    """
    Output of the second branch in a parallel LSI configuration.

    Formula: s_2(n) = x(n) * h_2(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_2 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.47, p. 116
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Output of the second branch in a parallel LSI configuration.",
        }
    )


# -- rng046: Output of two LSI systems in parallel equals input convolved with sum of responses..
def rangayyan_ch3_lsi_parallel_total(x, h_1, h_2, n):
    """
    Output of two LSI systems in parallel equals input convolved with sum of responses.

    Formula: y(n) = s_1(n) + s_2(n) = x(n) * [h_1(n) + h_2(n)] = x(n) * h(n)

    Parameters
    ----------
    x : array-like
        Input data.
    h_1 : array-like
        Input data.
    h_2 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.48, p. 116
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Output of two LSI systems in parallel equals input convolved with sum of responses.",
        }
    )


# -- rng051: LTI convolution maps to multiplication in s-domain and frequency domain..
def rangayyan_ch3_lti_convolution_property(x, h):
    """
    LTI convolution maps to multiplication in s-domain and frequency domain.

    Formula: if y(t) = x(t) * h(t), then Y(s) = X(s) H(s) and Y(omega) = X(omega) H(omega)

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.53, p. 119
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "LTI convolution maps to multiplication in s-domain and frequency domain.",
        }
    )


# -- rng079: Circular (periodic) convolution of two N-periodic discrete signals..
def rangayyan_ch3_periodic_convolution(x_p, h_p, n, N):
    """
    Circular (periodic) convolution of two N-periodic discrete signals.

    Formula: y_p(n) = sum_{k=0}^{N-1} x_p(k) * h_p[(n-k) mod N]

    Parameters
    ----------
    x_p : array-like
        Input data.
    h_p : array-like
        Input data.
    n : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.90, p. 131
    """
    x_p = np.atleast_1d(np.asarray(x_p, dtype=float))
    n = len(x_p)
    result = float(np.mean(x_p))
    se = float(np.std(x_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Circular (periodic) convolution of two N-periodic discrete signals.",
        }
    )


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
def rangayyan_ch4_composite_signal_in_terms_of_g(g, n, cdf=None):
    """
    Composite test signal expressed in terms of three delayed scaled copies of g(n).

    Formula: x(n) = g(n-5) + 0.5*g(n-16) + 0.25*g(n-26)

    Parameters
    ----------
    g : array-like
        Input data.
    n : array-like
        Input data.
    cdf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.53, p. 240
    """
    g = np.asarray(g, dtype=float)
    n = len(g)
    if n < 2:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Composite test signal expressed in terms of three delayed scaled copies of g(n).",
            }
        )
    x_sorted = np.sort(g)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(g), scale=np.std(g, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k**2 * lam**2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(
        payload={
            "statistic": float(statistic),
            "p_value": float(p_value),
            "n": n,
            "method": "Composite test signal expressed in terms of three delayed scaled copies of g(n).",
        }
    )


_CHEATSHEET = [
    'rgam: Amplitude-modulated (AM) signal model',
    'rgconv: Linear convolution of two finite-length sequences',
    'rgfm: Frequency-modulated (FM) signal model for respiratory sounds',
    'rgstvar: Time-variant linear system (TV-LSI) characterization',
    'rng024: Dirac delta definition, Rangayyan eq. (3.24)',
    'rng025: unit area of the delta, Rangayyan eq. (3.25)',
    'rng026: delta as a power-function limit, Rangayyan eq. (3.26)',
    'rng027: continuous unit step, Rangayyan eq. (3.27)',
    'rng028: sifting property, Rangayyan eq. (3.28)',
    'rng030: continuous-time convolution, Rangayyan eq. (3.30)',
    'rng031: commuted continuous convolution, Rangayyan eq. (3.31)',
    'rng032: y(t) = int_0^t x(tau) h(t-tau) dtau, trapezoid on the sample grid',
    'rng033: y(t) = int_0^t h(tau) x(t-tau) dtau -- rng032 commuted',
    'rng034: discrete unit impulse, Rangayyan eq. (3.34)',
    'rng035: discrete unit step, Rangayyan eq. (3.35)',
    'rng036: y(n) = sum_{k=0}^{n} x(k) h(n-k)',
    'rng037: y(n) = sum_{k=0}^{n} h(k) x(n-k) -- same as rng036 by commutativity',
    'rng038: Synthetic test signal: sum of a sine and a cosine.',
    'rng041: Intermediate output of the first LSI system in a series cascade.',
    'rng042: Output of two LSI systems in series equals input convolved with combined response.',
    'rng044: Output of the first branch in a parallel LSI configuration.',
    'rng045: Output of the second branch in a parallel LSI configuration.',
    'rng046: Output of two LSI systems in parallel equals input convolved with sum of responses.',
    'rng051: LTI convolution maps to multiplication in s-domain and frequency domain.',
    'rng079: Circular (periodic) convolution of two N-periodic discrete signals.',
    'rng223: Rangayyan Ch.4 three-event matched-filter test signal (Eq. 4.51)',
    'rng225: Composite test signal expressed in terms of three delayed scaled copies of g(n).',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
