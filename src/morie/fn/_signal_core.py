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


def filtfilt(b, a, x, axis=-1):
    xa = _ac.asarray(x)
    if len(xa.shape) == 2:
        if axis in (-1, 1):
            return _ac.marr([list(filtfilt(b, a, row)._flat())
                             for row in xa.data])
        cols = [list(filtfilt(b, a, [xa.data[i][j] for i in
                                     range(xa.shape[0])])._flat())
                for j in range(xa.shape[1])]
        return _ac.marr([[cols[j][i] for j in range(xa.shape[1])]
                         for i in range(xa.shape[0])])
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


# ------------------------------------------------------- design tail

def _cheb1_analog_zpk(n, rp):
    eps = _math.sqrt(10.0 ** (0.1 * rp) - 1.0)
    mu = _math.asinh(1.0 / eps) / n
    p = []
    for k in range(n):
        theta = _math.pi * (2.0 * k + 1.0) / (2.0 * n)
        p.append(complex(-_math.sinh(mu) * _math.sin(theta),
                         _math.cosh(mu) * _math.cos(theta)))
    kgain = 1.0
    prod = complex(1.0)
    for pi in p:
        prod *= -pi
    kgain = prod.real
    if n % 2 == 0:
        kgain /= _math.sqrt(1.0 + eps * eps)
    return [], p, kgain


def _cheb2_analog_zpk(n, rs):
    de = 1.0 / _math.sqrt(10.0 ** (0.1 * rs) - 1.0)
    mu = _math.asinh(1.0 / de) / n
    z = []
    p = []
    for k in range(n):
        theta = _math.pi * (2.0 * k + 1.0) / (2.0 * n)
        # chebyshev-1 pole, inverted
        p1 = complex(-_math.sinh(mu) * _math.sin(theta),
                     _math.cosh(mu) * _math.cos(theta))
        p.append(1.0 / p1)
        s = _math.sin(theta)
        if abs(_math.cos(theta)) > 1e-15:
            z.append(complex(0.0, 1.0 / _math.cos(theta)))
    # pure imaginary zeros come in conjugate pairs; odd n drops one
    kgain = 1.0
    num = complex(1.0)
    den = complex(1.0)
    for zi in z:
        num *= -zi
    for pi in p:
        den *= -pi
    kgain = (den / num).real
    return z, p, kgain


def _design_iir(n, wn, btype, atype, ripple, output):
    if atype == "cheby1":
        z, p, k = _cheb1_analog_zpk(n, ripple)
    elif atype == "cheby2":
        z, p, k = _cheb2_analog_zpk(n, ripple)
    else:
        raise ValueError(atype)
    fs = 2.0
    if btype in ("low", "lowpass"):
        warped = 2.0 * fs * _math.tan(_math.pi * float(wn) / fs)
        z = [zi * warped for zi in z]
        pn = [pi * warped for pi in p]
        k *= warped ** (len(p) - len(z))
        p = pn
    elif btype in ("high", "highpass"):
        warped = 2.0 * fs * _math.tan(_math.pi * float(wn) / fs)
        zn = [warped / zi for zi in z]
        pn = [warped / pi for pi in p]
        num = complex(1.0)
        den = complex(1.0)
        for zi in z:
            num *= -zi
        for pi in p:
            den *= -pi
        k *= (num / den).real
        z = zn + [0j] * (len(p) - len(z))
        p = pn
    else:
        raise NotImplementedError("cheby band: lowpass/highpass only")
    zd, pd, kd = _bilinear_zpk(z, p, k, fs)
    if output == "ba":
        return _zpk2tf(zd, pd, kd)
    return _ac.marr(_zpk2sos(zd, pd, kd))


def cheby1(N, rp, Wn, btype="low", output="ba", fs=None):
    if fs is not None:
        Wn = 2.0 * float(Wn) / fs
    return _design_iir(int(N), Wn, btype, "cheby1", float(rp), output)


def cheby2(N, rs, Wn, btype="low", output="ba", fs=None):
    if fs is not None:
        Wn = 2.0 * float(Wn) / fs
    return _design_iir(int(N), Wn, btype, "cheby2", float(rs), output)


def iirnotch(w0, Q, fs=2.0):
    w0n = 2.0 * float(w0) / fs if fs != 2.0 else float(w0)
    om = _math.pi * w0n
    bw = om / float(Q)
    gb = 1.0 / _math.sqrt(2.0)
    beta = _math.sqrt(1.0 - gb * gb) / gb * _math.tan(bw / 2.0)
    gain = 1.0 / (1.0 + beta)
    b = [gain, -2.0 * _math.cos(om) * gain, gain]
    a = [1.0, -2.0 * _math.cos(om) * gain, 2.0 * gain - 1.0]
    return b, a


def firwin(numtaps, cutoff, window="hamming", pass_zero=True, fs=None):
    if fs is not None:
        if isinstance(cutoff, (list, tuple)):
            cutoff = [2.0 * c / fs for c in cutoff]
        else:
            cutoff = 2.0 * float(cutoff) / fs
    n = int(numtaps)
    m = (n - 1) / 2.0
    if not isinstance(cutoff, (list, tuple)):
        cutoff = [float(cutoff)]

    def sinc(x):
        return 1.0 if x == 0.0 else _math.sin(_math.pi * x) / (
            _math.pi * x)
    # ideal impulse response (lowpass or bandpass sum)
    h = [0.0] * n
    bands = []
    if pass_zero:
        bands.append((0.0, cutoff[0]))
        for i in range(1, len(cutoff) - 1, 2):
            bands.append((cutoff[i], cutoff[i + 1]))
    else:
        cl = [0.0] + list(cutoff) + [1.0]
        for i in range(1, len(cl) - 1, 2):
            bands.append((cl[i], cl[i + 1]))
    for lo, hi in bands:
        for i in range(n):
            x = i - m
            h[i] += hi * sinc(hi * x) - lo * sinc(lo * x)
    win = get_window(window, n) if window != "hamming" else [
        0.54 - 0.46 * _math.cos(2.0 * _math.pi * i / (n - 1))
        for i in range(n)]
    h = [h[i] * win[i] for i in range(n)]
    # normalize DC gain to 1 for pass_zero
    if pass_zero:
        s = _math.fsum(h)
        h = [v / s for v in h]
    return _ac.marr(h)


def freqz(b, a=1, worN=512, fs=None):
    bv = [float(v) for v in _ac.asarray(b)._flat()]
    av = [float(v) for v in (_ac.asarray(a)._flat()
                             if not isinstance(a, (int, float))
                             else [float(a)])]
    if isinstance(worN, int):
        ws = [_math.pi * k / worN for k in range(worN)]
    else:
        ws = [float(v) for v in _ac.asarray(worN)._flat()]
        if fs is not None:
            ws = [2.0 * _math.pi * v / fs for v in ws]
    from . import _array_core as _ac2
    h = []
    for w in ws:
        zi = complex(_math.cos(-w), _math.sin(-w))
        num = complex(0.0)
        zp = complex(1.0)
        for c in bv:
            num += c * zp
            zp *= zi
        den = complex(0.0)
        zp = complex(1.0)
        for c in av:
            den += c * zp
            zp *= zi
        h.append(num / den)
    wout = ws if fs is None else [v * fs / (2.0 * _math.pi)
                                  for v in ws]
    return _ac.marr(wout), _ac2.carr(h)


def group_delay(system, w=512, fs=None):
    b, a = system
    # numerical derivative of phase
    ws, h = freqz(b, a, worN=w, fs=None)
    wl = list(ws._flat())
    hl = h.tolist()
    ph = []
    prev = None
    acc = 0.0
    for v in hl:
        p = _math.atan2(v.imag, v.real)
        if prev is not None:
            d = p - prev
            while d > _math.pi:
                d -= 2.0 * _math.pi
            while d < -_math.pi:
                d += 2.0 * _math.pi
            acc += d
        ph.append(acc)
        prev = p
    gd = []
    for i in range(len(wl)):
        if i == 0:
            gd.append(-(ph[1] - ph[0]) / (wl[1] - wl[0]))
        elif i == len(wl) - 1:
            gd.append(-(ph[-1] - ph[-2]) / (wl[-1] - wl[-2]))
        else:
            gd.append(-(ph[i + 1] - ph[i - 1])
                      / (wl[i + 1] - wl[i - 1]))
    return _ac.marr(wl), _ac.marr(gd)


def bilinear(b, a, fs=1.0):
    """Bilinear transform of an analog (b, a) to digital."""
    bv = [float(v) for v in _ac.asarray(b)._flat()]
    av = [float(v) for v in _ac.asarray(a)._flat()]
    # roots via companion matrix eig? ponytail: polynomial orders in
    # morie call sites are <= 4 — use numpy-free Durand-Kerner
    def roots(c):
        c = [v / c[0] for v in c]
        n = len(c) - 1
        if n == 0:
            return []
        rs = [complex(0.4, 0.9) ** k for k in range(n)]
        for _ in range(200):
            new = []
            for i in range(n):
                num = complex(1.0)
                for j in range(n):
                    if j != i:
                        num *= (rs[i] - rs[j])
                pv = complex(0.0)
                for cf in c:
                    pv = pv * rs[i] + cf
                new.append(rs[i] - pv / num)
            if max(abs(a_ - b_) for a_, b_ in zip(new, rs)) < 1e-13:
                rs = new
                break
            rs = new
        return rs
    z = roots(bv) if len(bv) > 1 else []
    p = roots(av) if len(av) > 1 else []
    k = bv[0] / av[0]
    zd, pd, kd = _bilinear_zpk(z, p, k, fs)
    return _zpk2tf(zd, pd, kd)


def resample_poly(x, up, down, window=("kaiser", 5.0)):
    """Polyphase resampling: upsample, FIR lowpass, downsample."""
    del window
    xs = list(_ac.asarray(x)._flat())
    up, down = int(up), int(down)
    g = _math.gcd(up, down)
    up //= g
    down //= g
    if up == down == 1:
        return _ac.marr(xs)
    # zero-stuff
    ups = []
    for v in xs:
        ups.append(v * up)
        ups.extend([0.0] * (up - 1))
    # lowpass at min(1/up, 1/down), 10 taps per phase (hamming sinc)
    cutoff = 1.0 / max(up, down)
    ntaps = 10 * max(up, down) + 1
    h = list(firwin(ntaps, cutoff)._flat())
    # filter (linear-phase FIR, centered)
    m = len(h) // 2
    n = len(ups)
    filt = []
    for i in range(n):
        acc = 0.0
        for k in range(len(h)):
            idx = i + m - k
            if 0 <= idx < n:
                acc += h[k] * ups[idx]
        filt.append(acc)
    return _ac.marr(filt[::down])


def convolve2d(a, b, mode="full", boundary="fill", fillvalue=0.0):
    A = _ac.atleast_2d(a)
    B = _ac.atleast_2d(b)
    ma, na = A.shape
    mb, nb = B.shape
    mf, nf = ma + mb - 1, na + nb - 1
    out = [[0.0] * nf for _ in range(mf)]
    for i in range(ma):
        for j in range(na):
            v = A.data[i][j]
            if v == 0.0:
                continue
            for k in range(mb):
                for l_ in range(nb):
                    out[i + k][j + l_] += v * B.data[k][l_]
    if mode == "full":
        return _ac.marr(out)
    if mode == "same":
        r0 = (mb - 1) // 2
        c0 = (nb - 1) // 2
        return _ac.marr([[out[i + r0][j + c0] for j in range(na)]
                         for i in range(ma)])
    if mode == "valid":
        return _ac.marr([[out[i][j]
                          for j in range(nb - 1, na)]
                         for i in range(mb - 1, ma)])
    raise ValueError(mode)


def dpss(M, NW, Kmax=None):
    """Slepian sequences via the symmetric tridiagonal eigenproblem."""
    M = int(M)
    W = float(NW) / M
    diag = [((M - 1.0 - 2.0 * i) / 2.0) ** 2
            * _math.cos(2.0 * _math.pi * W) for i in range(M)]
    off = [i * (M - i) / 2.0 for i in range(1, M)]
    A = [[0.0] * M for _ in range(M)]
    for i in range(M):
        A[i][i] = diag[i]
    for i in range(M - 1):
        A[i][i + 1] = A[i + 1][i] = off[i]
    w, V = _ac.linalg.eigh(_ac.marr(A))
    order = sorted(range(M), key=lambda i: -float(w[i]))
    k = int(Kmax) if Kmax is not None else 1
    Vd = V.tolist()
    out = []
    for kk in range(k):
        col = [Vd[i][order[kk]] for i in range(M)]
        s = _math.fsum(col)
        if s < 0:
            col = [-v for v in col]
        nrm = _math.sqrt(_math.fsum(v * v for v in col))
        col = [v / nrm for v in col]
        if Kmax is None:
            # scipy default norm="approximate":
            # peak to 1, then * M^2/(M^2 + NW)
            mx = max(abs(v) for v in col)
            corr = (M * M) / (M * M + float(NW))
            col = [v / mx * corr for v in col]
        out.append(col)
    return _ac.marr(out if k > 1 else out[0])


class windows:
    dpss = staticmethod(dpss)

    @staticmethod
    def hann(n):
        return _ac.marr(_hann(n))


def iirfilter(N, Wn, rp=None, rs=None, btype="low", ftype="butter",
              output="ba", fs=None):
    if ftype == "butter":
        return butter(N, Wn, btype=btype, output=output, fs=fs)
    if ftype in ("cheby1", "chebyshev1"):
        return cheby1(N, rp, Wn, btype=btype, output=output, fs=fs)
    if ftype in ("cheby2", "chebyshev2"):
        return cheby2(N, rs, Wn, btype=btype, output=output, fs=fs)
    raise NotImplementedError("iirfilter ftype %r" % ftype)


for _n in ("cheby1", "cheby2", "iirnotch", "firwin", "freqz",
           "group_delay", "bilinear", "resample_poly", "convolve2d",
           "dpss", "iirfilter"):
    setattr(signal, _n, staticmethod(globals()[_n]))
signal.windows = windows


# ------------------------------------------------------- elliptic filter

def _landen_seq(k, m=12):
    v = []
    for _ in range(m):
        k = (k / (1.0 + _math.sqrt(1.0 - k * k))) ** 2
        v.append(k)
    return v


def _ellipk_agm(k):
    a, b = 1.0, _math.sqrt(1.0 - k * k)
    for _ in range(60):
        a, b = (a + b) / 2.0, _math.sqrt(a * b)
        if abs(a - b) < 1e-16:
            break
    return _math.pi / (2.0 * a)


def _cde(u, k):
    """Jacobi cd(u*K, k) for complex u (Orfanidis descending Landen)."""
    vs = _landen_seq(k)
    w = _cmath.cos(u * _math.pi / 2.0)
    for v in reversed(vs):
        w = (1.0 + v) * w / (1.0 + v * w * w)
    return w


def _asne(w, k):
    """Inverse sn: u with sn(u*K, k) = w (complex)."""
    vs = _landen_seq(k)
    kp = k
    for v in vs:
        w = 2.0 * w / ((1.0 + v) * (1.0 + _cmath.sqrt(
            1.0 - kp * kp * w * w)))
        kp = v
    return 2.0 / _math.pi * _cmath.asin(w)


def _ellipdeg(n, k1):
    """Solve the degree equation for k given N and k1 (nome method)."""
    kp1 = _math.sqrt(1.0 - k1 * k1)
    K1 = _ellipk_agm(k1)
    K1p = _ellipk_agm(kp1)
    q1 = _math.exp(-_math.pi * K1p / K1)
    q = q1 ** (1.0 / n)
    # k from nome via theta functions
    num = _math.fsum(q ** (m * (m + 1)) for m in range(15))
    den = 1.0 + 2.0 * _math.fsum(q ** (m * m) for m in range(1, 16))
    return 4.0 * _math.sqrt(q) * (num / den) ** 2


def _ellip_analog_zpk(n, rp, rs):
    eps_p = _math.sqrt(10.0 ** (0.1 * rp) - 1.0)
    eps_s = _math.sqrt(10.0 ** (0.1 * rs) - 1.0)
    k1 = eps_p / eps_s
    k = _ellipdeg(n, k1)
    z = []
    p = []
    l_ = n % 2
    v0 = -1j * _asne(1j / eps_p, k1) / n
    for i in range(1, (n - l_) // 2 + 1):
        ui = (2.0 * i - 1.0) / n
        zeta = _cde(ui, k)
        zr = 1j / (k * zeta)
        z.append(zr)
        z.append(zr.conjugate())
        pi_ = 1j * _cde(ui - 1j * v0, k)
        p.append(pi_)
        p.append(pi_.conjugate())
    if l_:
        p.append(1j * _cde(1.0 - 1j * v0, k))
    num = complex(1.0)
    den = complex(1.0)
    for zi in z:
        num *= -zi
    for pi_ in p:
        den *= -pi_
    kgain = (den / num).real
    if l_ == 0:
        kgain /= _math.sqrt(1.0 + eps_p * eps_p)
    return z, p, kgain


def ellip(N, rp, rs, Wn, btype="low", output="ba", fs=None):
    if fs is not None:
        Wn = 2.0 * float(Wn) / fs
    z, p, k = _ellip_analog_zpk(int(N), float(rp), float(rs))
    fs2 = 2.0
    if btype in ("low", "lowpass"):
        warped = 2.0 * fs2 * _math.tan(_math.pi * float(Wn) / fs2)
        z = [zi * warped for zi in z]
        pn = [pi * warped for pi in p]
        k *= warped ** (len(p) - len(z))
        p = pn
    elif btype in ("high", "highpass"):
        warped = 2.0 * fs2 * _math.tan(_math.pi * float(Wn) / fs2)
        num = complex(1.0)
        den = complex(1.0)
        for zi in z:
            num *= -zi
        for pi_ in p:
            den *= -pi_
        k *= (num / den).real
        z = [warped / zi for zi in z] + [0j] * (len(p) - len(z))
        p = [warped / pi_ for pi_ in p]
    else:
        raise NotImplementedError("ellip: lowpass/highpass only")
    zd, pd, kd = _bilinear_zpk(z, p, k, fs2)
    if output == "ba":
        return _zpk2tf(zd, pd, kd)
    return _ac.marr(_zpk2sos(zd, pd, kd))


signal.ellip = staticmethod(ellip)
