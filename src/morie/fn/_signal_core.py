"""morie signal core: scipy.signal subset (butter / lfilter / filtfilt /
sosfilt / sosfiltfilt).

Native Butterworth design: analog prototype poles, frequency pre-warp,
lp2lp/lp2hp/lp2bp transforms, bilinear transform, zpk -> ba or sos.
Zero-phase filtering reproduces scipy's odd-extension padding and
steady-state initial conditions (lfilter_zi / sosfilt_zi), so outputs
match scipy.signal to ~1e-9 (equivalence-tested in
tests/fn/test_signal_core.py).  Pure Python; complex arithmetic uses the
builtin complex type internally, all public outputs are real.
"""

from __future__ import annotations

import cmath as _cmath
import math as _math

from . import _array_core as _ac


# ------------------------------------------------------------ helpers

def _poly(roots):
    """Monic polynomial coefficients from roots (complex ok, real out)."""
    c = [complex(1.0)]
    for r in roots:
        c = [c[0]] + [c[i + 1] - r * c[i] for i in range(len(c) - 1)] \
            + [-r * c[-1]]
        # rebuild properly: convolve c with (1, -r)
    return c


def _polymulroot(c, r):
    out = [complex(0.0)] * (len(c) + 1)
    for i, v in enumerate(c):
        out[i] += v
        out[i + 1] -= v * r
    return out


def _poly_from_roots(roots):
    c = [complex(1.0)]
    for r in roots:
        c = _polymulroot(c, r)
    return c


def _real(coeffs):
    return [v.real for v in coeffs]


# ------------------------------------------------------------ design

def _butter_analog_poles(n):
    return [_cmath.exp(1j * _math.pi * (2.0 * k + n + 1.0) / (2.0 * n))
            for k in range(n)]


def _bilinear_zpk(z, p, k, fs):
    fs2 = 2.0 * fs
    zd = [(fs2 + zi) / (fs2 - zi) for zi in z]
    pd = [(fs2 + pi) / (fs2 - pi) for pi in p]
    zd += [-1.0 + 0j] * (len(p) - len(z))
    num = complex(1.0)
    den = complex(1.0)
    for zi in z:
        num *= (fs2 - zi)
    for pi in p:
        den *= (fs2 - pi)
    kd = (k * num / den).real
    return zd, pd, kd


def _butter_zpk(n, wn, btype):
    """Digital Butterworth zpk; wn normalized (Nyquist = 1)."""
    p = _butter_analog_poles(n)
    z = []
    k = 1.0
    fs = 2.0
    if btype in ("low", "lowpass"):
        warped = 2.0 * fs * _math.tan(_math.pi * float(wn) / fs)
        p = [pi * warped for pi in p]
        k *= warped ** n
    elif btype in ("high", "highpass"):
        warped = 2.0 * fs * _math.tan(_math.pi * float(wn) / fs)
        prod = complex(1.0)
        for pi in p:
            prod *= -pi
        k /= prod.real if abs(prod.imag) < 1e-12 * abs(prod.real) \
            else prod.real
        z = [0j] * n
        p = [warped / pi for pi in p]
    elif btype in ("band", "bandpass"):
        lo, hi = float(wn[0]), float(wn[1])
        w1 = 2.0 * fs * _math.tan(_math.pi * lo / fs)
        w2 = 2.0 * fs * _math.tan(_math.pi * hi / fs)
        bw = w2 - w1
        w0 = _math.sqrt(w1 * w2)
        pn = []
        for pi in p:
            pb = pi * bw / 2.0
            disc = _cmath.sqrt(pb * pb - w0 * w0)
            pn.append(pb + disc)
            pn.append(pb - disc)
        z = [0j] * n
        p = pn
        k *= bw ** n
    else:
        raise ValueError("unsupported btype %r" % btype)
    return _bilinear_zpk(z, p, k, fs)


def _zpk2tf(z, p, k):
    b = [k * c for c in _poly_from_roots(z)]
    a = _poly_from_roots(p)
    return _real(b), _real(a)


def _zpk2sos(z, p, k):
    """Pair conjugate poles/zeros into second-order sections.

    ponytail: simple conjugate pairing (Butterworth-only inputs here) —
    scipy's nearest-neighbour pairing if other filters ever need it.
    """
    def split(vals):
        cplx, real = [], []
        used = [False] * len(vals)
        for i, v in enumerate(vals):
            if used[i]:
                continue
            if abs(v.imag) > 1e-12:
                for j in range(i + 1, len(vals)):
                    if not used[j] and abs(vals[j] - v.conjugate()) < 1e-8:
                        used[j] = True
                        break
                used[i] = True
                cplx.append(v)
            else:
                used[i] = True
                real.append(v.real)
        return cplx, real

    pc, pr = split(p)
    zc, zr = split(z)
    sections = []
    # complex pole pairs first (each with a zero pair if available)
    for pp in pc:
        a = _real(_poly_from_roots([pp, pp.conjugate()]))
        if zc:
            zz = zc.pop()
            b = _real(_poly_from_roots([zz, zz.conjugate()]))
        elif len(zr) >= 2:
            b = _real(_poly_from_roots([zr.pop(), zr.pop()]))
        elif zr:
            b = _real(_poly_from_roots([zr.pop()])) + [0.0]
        else:
            b = [1.0, 0.0, 0.0]
        sections.append(b + a)
    # leftover real poles in pairs
    while pr:
        if len(pr) >= 2:
            a = _real(_poly_from_roots([pr.pop(), pr.pop()]))
        else:
            a = _real(_poly_from_roots([pr.pop()])) + [0.0]
        if len(zr) >= 2:
            b = _real(_poly_from_roots([zr.pop(), zr.pop()]))
        elif zr:
            b = _real(_poly_from_roots([zr.pop()])) + [0.0]
        else:
            b = [1.0, 0.0, 0.0]
        sections.append(b + a)
    if not sections:
        sections.append([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    sections[0][0] *= k
    sections[0][1] *= k
    sections[0][2] *= k
    return sections


def butter(N, Wn, btype="low", output="ba", fs=None):
    if fs is not None:
        if isinstance(Wn, (list, tuple)):
            Wn = [2.0 * w / fs for w in Wn]
        else:
            Wn = 2.0 * float(Wn) / fs
    if hasattr(Wn, "tolist"):
        Wn = Wn.tolist()
    z, p, k = _butter_zpk(int(N), Wn, btype)
    if output == "ba":
        return _zpk2tf(z, p, k)
    if output == "sos":
        return _ac.marr(_zpk2sos(z, p, k))
    raise ValueError("unsupported output %r" % output)


# ------------------------------------------------------------ filtering

def lfilter(b, a, x, zi=None):
    b = list(_ac.asarray(b)._flat())
    a = list(_ac.asarray(a)._flat())
    xs = list(_ac.asarray(x)._flat())
    if a[0] != 1.0:
        b = [v / a[0] for v in b]
        a = [v / a[0] for v in a]
    n = max(len(a), len(b))
    b = b + [0.0] * (n - len(b))
    a = a + [0.0] * (n - len(a))
    z = list(zi) if zi is not None else [0.0] * (n - 1)
    y = []
    for xv in xs:
        yv = b[0] * xv + (z[0] if n > 1 else 0.0)
        for i in range(n - 2):
            z[i] = b[i + 1] * xv + z[i + 1] - a[i + 1] * yv
        if n > 1:
            z[n - 2] = b[n - 1] * xv - a[n - 1] * yv
        y.append(yv)
    if zi is None:
        return _ac.marr(y)
    return _ac.marr(y), z


def lfilter_zi(b, a):
    b = list(_ac.asarray(b)._flat())
    a = list(_ac.asarray(a)._flat())
    if a[0] != 1.0:
        b = [v / a[0] for v in b]
        a = [v / a[0] for v in a]
    n = max(len(a), len(b))
    b = b + [0.0] * (n - len(b))
    a = a + [0.0] * (n - len(a))
    m = n - 1
    # (I - A^T) zi = B,  A = companion(a)^T convention of scipy.lfilter_zi
    ImA = [[(1.0 if i == j else 0.0) for j in range(m)] for i in range(m)]
    for i in range(m):
        ImA[i][0] += a[i + 1]
        if i + 1 < m:
            ImA[i][i + 1] -= 1.0
    B = [b[i + 1] - a[i + 1] * b[0] for i in range(m)]
    sol = _ac.linalg.solve(_ac.marr(ImA), _ac.marr(B))
    return list(sol._flat())


def _odd_ext(xs, n):
    left = [2.0 * xs[0] - xs[i] for i in range(n, 0, -1)]
    right = [2.0 * xs[-1] - xs[-2 - i] for i in range(n)]
    return left + xs + right


def filtfilt(b, a, x):
    bs = list(_ac.asarray(b)._flat())
    as_ = list(_ac.asarray(a)._flat())
    xs = list(_ac.asarray(x)._flat())
    edge = 3 * max(len(as_), len(bs))
    if len(xs) <= edge:
        raise ValueError("input too short for padlen %d" % edge)
    ext = _odd_ext(xs, edge)
    zi = lfilter_zi(bs, as_)
    y, _ = lfilter(bs, as_, ext, zi=[z * ext[0] for z in zi])
    y = list(y._flat())[::-1]
    y2, _ = lfilter(bs, as_, y, zi=[z * y[0] for z in zi])
    y2 = list(y2._flat())[::-1]
    return _ac.marr(y2[edge:len(y2) - edge])


def _sos_rows(sos):
    a = _ac.asarray(sos)
    return [list(map(float, r)) for r in a.data]


def sosfilt(sos, x, zi=None):
    rows = _sos_rows(sos)
    xs = list(_ac.asarray(x)._flat())
    z = [list(zs) for zs in zi] if zi is not None \
        else [[0.0, 0.0] for _ in rows]
    for s, r in enumerate(rows):
        b0, b1, b2, a0, a1, a2 = r
        if a0 != 1.0:
            b0, b1, b2, a1, a2 = (b0 / a0, b1 / a0, b2 / a0,
                                  a1 / a0, a2 / a0)
        z0, z1 = z[s]
        out = []
        for xv in xs:
            yv = b0 * xv + z0
            z0 = b1 * xv + z1 - a1 * yv
            z1 = b2 * xv - a2 * yv
            out.append(yv)
        z[s] = [z0, z1]
        xs = out
    if zi is None:
        return _ac.marr(xs)
    return _ac.marr(xs), z


def sosfilt_zi(sos):
    rows = _sos_rows(sos)
    scale = 1.0
    zi = []
    for r in rows:
        b = r[:3]
        a = r[3:]
        zi.append([scale * v for v in lfilter_zi(b, a)])
        scale *= sum(b) / sum(a)
    return zi


def sosfiltfilt(sos, x):
    rows = _sos_rows(sos)
    xs = list(_ac.asarray(x)._flat())
    n_sections = len(rows)
    ntaps = 2 * n_sections + 1
    ntaps -= min(sum(1 for r in rows if r[2] == 0.0),
                 sum(1 for r in rows if r[5] == 0.0))
    edge = ntaps * 3
    if len(xs) <= edge:
        raise ValueError("input too short for padlen %d" % edge)
    ext = _odd_ext(xs, edge)
    zi = sosfilt_zi(rows)
    y, _ = sosfilt(rows, ext,
                   zi=[[v * ext[0] for v in zs] for zs in zi])
    y = list(y._flat())[::-1]
    y2, _ = sosfilt(rows, y,
                    zi=[[v * y[0] for v in zs] for zs in zi])
    y2 = list(y2._flat())[::-1]
    return _ac.marr(y2[edge:len(y2) - edge])


class signal:  # namespace mirror for `from scipy import signal`
    butter = staticmethod(butter)
    lfilter = staticmethod(lfilter)
    lfilter_zi = staticmethod(lfilter_zi)
    filtfilt = staticmethod(filtfilt)
    sosfilt = staticmethod(sosfilt)
    sosfilt_zi = staticmethod(sosfilt_zi)
    sosfiltfilt = staticmethod(sosfiltfilt)


# ------------------------------------------------------- spectral tail

def _hann(n):
    if n == 1:
        return [1.0]
    return [0.5 - 0.5 * _math.cos(2.0 * _math.pi * i / n)
            for i in range(n)]


def get_window(window, nperseg):
    if window in ("hann", "hanning"):
        return _hann(nperseg)
    if window == "hamming":
        return [0.54 - 0.46 * _math.cos(2.0 * _math.pi * i / nperseg)
                for i in range(nperseg)]
    if window in ("boxcar", "rectangular", None):
        return [1.0] * nperseg
    if isinstance(window, tuple) and window[0] == "tukey":
        alpha = float(window[1])
        if alpha <= 0:
            return [1.0] * nperseg
        # periodic (fftbins=True) like scipy's spectral helpers:
        # symmetric window of length n+1 with the last sample dropped
        n = nperseg + 1
        w = []
        width = int(alpha * (n - 1) / 2.0)
        for i in range(n):
            if i <= width:
                w.append(0.5 * (1.0 + _math.cos(_math.pi * (
                    -1.0 + 2.0 * i / (alpha * (n - 1))))))
            elif i >= n - width - 1:
                w.append(0.5 * (1.0 + _math.cos(_math.pi * (
                    -2.0 / alpha + 1.0 + 2.0 * i / (alpha * (n - 1))))))
            else:
                w.append(1.0)
        return w[:-1]
    if window == "blackman":
        return [0.42 - 0.5 * _math.cos(2.0 * _math.pi * i / nperseg)
                + 0.08 * _math.cos(4.0 * _math.pi * i / nperseg)
                for i in range(nperseg)]
    raise ValueError("unsupported window %r" % (window,))


def _csd_core(x, y, fs, window, nperseg, noverlap, detrend):
    xs = list(_ac.asarray(x)._flat())
    ys = list(_ac.asarray(y)._flat())
    n = len(xs)
    nperseg = int(min(nperseg, n))
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    win = get_window(window, nperseg)
    scale = 1.0 / (fs * _math.fsum(w * w for w in win))
    nfreq = nperseg // 2 + 1
    acc = [0j] * nfreq
    nseg = 0
    start = 0
    while start + nperseg <= n:
        segx = xs[start:start + nperseg]
        segy = ys[start:start + nperseg]
        if detrend in ("constant", True):
            mx = _math.fsum(segx) / nperseg
            my = _math.fsum(segy) / nperseg
            segx = [v - mx for v in segx]
            segy = [v - my for v in segy]
        fx = _ac.fft.rfft([v * w for v, w in zip(segx, win)]).tolist()
        fy = _ac.fft.rfft([v * w for v, w in zip(segy, win)]).tolist()
        for k in range(nfreq):
            acc[k] += fx[k].conjugate() * fy[k]
        nseg += 1
        start += step
    if nseg == 0:
        raise ValueError("signal shorter than nperseg")
    pxy = [v * scale / nseg for v in acc]
    # one-sided: double everything but DC (and Nyquist when even)
    for k in range(1, nfreq - (1 if nperseg % 2 == 0 else 0)):
        pxy[k] *= 2.0
    freqs = [k * fs / nperseg for k in range(nfreq)]
    return _ac.marr(freqs), pxy


def welch(x, fs=1.0, window="hann", nperseg=256, noverlap=None,
          detrend="constant", **kw):
    del kw
    freqs, pxy = _csd_core(x, x, fs, window, nperseg, noverlap,
                           detrend)
    return freqs, _ac.marr([v.real for v in pxy])


def csd(x, y, fs=1.0, window="hann", nperseg=256, noverlap=None,
        detrend="constant", **kw):
    del kw
    freqs, pxy = _csd_core(x, y, fs, window, nperseg, noverlap,
                           detrend)
    from . import _array_core as _ac2
    return freqs, _ac2.carr(pxy)


def coherence(x, y, fs=1.0, window="hann", nperseg=256,
              noverlap=None, **kw):
    del kw
    f1, pxx = welch(x, fs, window, nperseg, noverlap)
    _, pyy = welch(y, fs, window, nperseg, noverlap)
    _, pxy = csd(x, y, fs, window, nperseg, noverlap)
    cxy = [abs(pxy.tolist()[k]) ** 2
           / max(pxx.tolist()[k] * pyy.tolist()[k], 1e-300)
           for k in range(len(f1))]
    return f1, _ac.marr(cxy)


def periodogram(x, fs=1.0, window="boxcar", **kw):
    del kw
    xs = list(_ac.asarray(x)._flat())
    return welch(xs, fs=fs, window=window, nperseg=len(xs),
                 noverlap=0)


def stft(x, fs=1.0, window="hann", nperseg=256, noverlap=None, **kw):
    del kw
    xs = list(_ac.asarray(x)._flat())
    n = len(xs)
    nperseg = int(min(nperseg, n))
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    win = get_window(window, nperseg)
    scale = 1.0 / _math.fsum(win)
    nfreq = nperseg // 2 + 1
    cols = []
    times = []
    start = 0
    while start + nperseg <= n:
        seg = [v * w for v, w in
               zip(xs[start:start + nperseg], win)]
        cols.append([v * scale for v in _ac.fft.rfft(seg).tolist()])
        times.append((start + nperseg / 2.0) / fs)
        start += step
    freqs = _ac.marr([k * fs / nperseg for k in range(nfreq)])
    z = [[cols[t][k] for t in range(len(cols))]
         for k in range(nfreq)]
    return freqs, _ac.marr(times), z


def spectrogram(x, fs=1.0, window=("tukey", 0.25), nperseg=256,
                noverlap=None, **kw):
    del kw
    xs = list(_ac.asarray(x)._flat())
    n = len(xs)
    nperseg = int(min(nperseg, n))
    if noverlap is None:
        noverlap = nperseg // 8
    step = nperseg - noverlap
    win = get_window(window, nperseg)
    scale = 1.0 / (fs * _math.fsum(w * w for w in win))
    nfreq = nperseg // 2 + 1
    cols = []
    times = []
    start = 0
    while start + nperseg <= n:
        seg = xs[start:start + nperseg]
        m = _math.fsum(seg) / nperseg
        seg = [(v - m) * w for v, w in zip(seg, win)]
        fx = _ac.fft.rfft(seg).tolist()
        p = [abs(v) ** 2 * scale for v in fx]
        for k in range(1, nfreq - (1 if nperseg % 2 == 0 else 0)):
            p[k] *= 2.0
        cols.append(p)
        times.append((start + nperseg / 2.0) / fs)
        start += step
    freqs = _ac.marr([k * fs / nperseg for k in range(nfreq)])
    sxx = [[cols[t][k] for t in range(len(cols))]
           for k in range(nfreq)]
    return freqs, _ac.marr(times), _ac.marr(sxx)


def hilbert(x):
    """Analytic signal via FFT (matches scipy.signal.hilbert)."""
    xs = list(_ac.asarray(x)._flat())
    n = len(xs)
    X = _ac.fft.fft(xs).tolist()
    h = [0.0] * n
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        for k in range(1, n // 2):
            h[k] = 2.0
    else:
        h[0] = 1.0
        for k in range(1, (n + 1) // 2):
            h[k] = 2.0
    Y = [X[k] * h[k] for k in range(n)]
    from . import _array_core as _ac2
    return _ac2.carr(_ac.fft.ifft(Y).tolist())


def fftconvolve(a, b, mode="full"):
    av = list(_ac.asarray(a)._flat())
    bv = list(_ac.asarray(b)._flat())
    n = len(av) + len(bv) - 1
    m = 1
    while m < n:
        m <<= 1
    fa = _ac.fft.fft(av + [0.0] * (m - len(av))).tolist()
    fb = _ac.fft.fft(bv + [0.0] * (m - len(bv))).tolist()
    out = _ac.fft.ifft([fa[k] * fb[k] for k in range(m)]).tolist()
    full = [v.real for v in out[:n]]
    if mode == "full":
        return _ac.marr(full)
    if mode == "same":
        start = (len(bv) - 1) // 2
        return _ac.marr(full[start:start + len(av)])
    if mode == "valid":
        lo = min(len(av), len(bv)) - 1
        hi = max(len(av), len(bv))
        return _ac.marr(full[lo:hi])
    raise ValueError("unsupported mode %r" % mode)


def find_peaks(x, height=None, distance=None, prominence=None, **kw):
    del kw
    xs = list(_ac.asarray(x)._flat())
    n = len(xs)
    peaks = []
    i = 1
    while i < n - 1:
        if xs[i - 1] < xs[i]:
            # find right edge of any plateau
            j = i
            while j < n - 1 and xs[j + 1] == xs[j]:
                j += 1
            if j < n - 1 and xs[j + 1] < xs[j]:
                peaks.append((i + j) // 2)
                i = j + 1
                continue
        i += 1
    props = {}
    if height is not None:
        hmin = height[0] if isinstance(height, (tuple, list)) \
            else float(height)
        peaks = [p for p in peaks if xs[p] >= hmin]
    if prominence is not None:
        pmin = prominence[0] if isinstance(prominence, (tuple, list)) \
            else float(prominence)
        kept = []
        proms = []
        for p in peaks:
            lo = p
            left_min = xs[p]
            for k in range(p - 1, -1, -1):
                if xs[k] > xs[p]:
                    break
                left_min = min(left_min, xs[k])
            right_min = xs[p]
            for k in range(p + 1, n):
                if xs[k] > xs[p]:
                    break
                right_min = min(right_min, xs[k])
            prom = xs[p] - max(left_min, right_min)
            if prom >= pmin:
                kept.append(p)
                proms.append(prom)
        peaks = kept
        props["prominences"] = _ac.marr(proms)
    if distance is not None:
        dmin = int(distance)
        order = sorted(range(len(peaks)),
                       key=lambda k: -xs[peaks[k]])
        keep = [True] * len(peaks)
        for oi in order:
            if not keep[oi]:
                continue
            for oj in range(len(peaks)):
                if oj != oi and keep[oj] \
                        and abs(peaks[oj] - peaks[oi]) < dmin:
                    keep[oj] = False
        peaks = [p for p, k in zip(peaks, keep) if k]
    props["peak_heights"] = _ac.marr([xs[p] for p in peaks])
    return _ac.marr([float(p) for p in peaks]), props


def savgol_filter(x, window_length, polyorder, **kw):
    del kw
    xs = list(_ac.asarray(x)._flat())
    wl = int(window_length)
    if wl % 2 == 0:
        raise ValueError("window_length must be odd")
    half = wl // 2
    # least-squares smoothing coefficients via normal equations
    # design matrix on offsets -half..half
    off = list(range(-half, half + 1))
    A = [[float(o) ** j for j in range(polyorder + 1)] for o in off]
    AtA = [[_math.fsum(A[i][a] * A[i][b] for i in range(wl))
            for b in range(polyorder + 1)]
           for a in range(polyorder + 1)]
    # solve AtA c = At e_row for the smoothing (0th derivative) weights
    At0 = [[A[i][a] for i in range(wl)]
           for a in range(polyorder + 1)]
    inv = _ac.linalg.inv(_ac.marr(AtA)).tolist()
    w = [_math.fsum(inv[0][a] * At0[a][i]
                    for a in range(polyorder + 1))
         for i in range(wl)]
    n = len(xs)
    out = []
    for i in range(n):
        # mirror-pad edges (scipy default mode="interp" differs at the
        # boundary; interior values match exactly)
        acc = 0.0
        for k, o in enumerate(off):
            idx = i + o
            if idx < 0:
                idx = -idx
            elif idx >= n:
                idx = 2 * (n - 1) - idx
            acc += w[k] * xs[idx]
        out.append(acc)
    return _ac.marr(out)


def medfilt(x, kernel_size=3):
    xs = list(_ac.asarray(x)._flat())
    k = int(kernel_size)
    half = k // 2
    n = len(xs)
    out = []
    for i in range(n):
        window = []
        for o in range(-half, half + 1):
            idx = i + o
            window.append(xs[idx] if 0 <= idx < n else 0.0)
        window.sort()
        out.append(window[k // 2])
    return _ac.marr(out)


def detrend(x, type="linear"):
    xs = list(_ac.asarray(x)._flat())
    n = len(xs)
    if type in ("constant", "c"):
        m = _math.fsum(xs) / n
        return _ac.marr([v - m for v in xs])
    tbar = (n - 1) / 2.0
    xbar = _math.fsum(xs) / n
    stt = _math.fsum((i - tbar) ** 2 for i in range(n))
    sxt = _math.fsum((i - tbar) * (xs[i] - xbar) for i in range(n))
    slope = sxt / stt
    return _ac.marr([xs[i] - (xbar + slope * (i - tbar))
                     for i in range(n)])


for _n in ("welch", "csd", "coherence", "periodogram", "stft",
           "spectrogram", "hilbert", "fftconvolve", "find_peaks",
           "savgol_filter", "medfilt", "detrend", "get_window"):
    setattr(signal, _n, staticmethod(globals()[_n]))
