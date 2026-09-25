# morie.fn -- function file (rootcoder007/morie)
"""Homomorphic deconvolution via cepstral liftering."""

from __future__ import annotations

from . import _array_core as np

from ._containers import SignalResult


def homomorphic_deconvolve(
    x: np.ndarray,
    *,
    cutoff: int,
    n_fft: int | None = None,
) -> SignalResult:
    """Homomorphic deconvolution separating minimum-phase and all-pass components.

    Uses complex cepstrum + low-time liftering to extract the
    minimum-phase (impulse response) component.

    :param x: 1-D input signal (assumed to be a convolution h * e).
    :param cutoff: Liftering cutoff (quefrency index). Cepstral coefficients
        above this index are zeroed to extract the slow-varying component.
    :param n_fft: FFT length (default: next power of 2 >= len(x)).
    :return: SignalResult with minimum-phase component in ``filtered``
        and excitation in ``extra["excitation"]``.
    """
    import math
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(n)))
    N = int(n_fft)
    X = np.fft.fft(x, n=N).tolist()
    # complex cepstrum as Oppenheim and Schafer define it: log|X| plus the
    # unwrapped phase with its linear (pure delay) term removed, which makes
    # the phase odd and the cepstrum real (MATLAB cceps / rcunwrap). The
    # previous version kept the linear term, so its "cepstrum" had a large
    # imaginary part that the real() calls then discarded.
    ph = [float(v) for v in np.unwrap(np.angle(np.array(X))).tolist()]
    nh = (N + 1) // 2
    nd = int(round(ph[nh] / math.pi)) if N > 1 else 0
    ph = [ph[k] - math.pi * nd * k / nh for k in range(N)]
    logX = [complex(math.log(abs(X[k]) + 1e-300), ph[k]) for k in range(N)]
    cep = [v.real for v in np.fft.ifft(np.array(logX)).tolist()]
    # low-time lifter, symmetric in quefrency: |q| < cutoff is the smooth
    # (system) part, the rest the excitation; together they are the whole
    # cepstrum, so H E = X exactly. The old lifter doubled 1..c, the
    # real-cepstrum minimum-phase fold, which does not split a complex
    # cepstrum and could not reconvolve to the signal.
    c = max(1, min(int(cutoff), N // 2))
    low = [1.0 if (q < c or N - q < c) else 0.0 for q in range(N)]
    cep_min = [cep[q] * low[q] for q in range(N)]
    cep_exc = [cep[q] - cep_min[q] for q in range(N)]
    H = [complex(v) for v in np.fft.fft(np.array(cep_min)).tolist()]
    Hs = [complex(math.exp(v.real) * math.cos(v.imag), math.exp(v.real) * math.sin(v.imag)) for v in H]
    E = [complex(v) for v in np.fft.fft(np.array(cep_exc)).tolist()]
    # the removed delay goes back into the excitation
    Es = []
    for k, v in enumerate(E):
        a = v.imag + math.pi * nd * k / nh
        Es.append(complex(math.exp(v.real) * math.cos(a), math.exp(v.real) * math.sin(a)))
    h = np.array([v.real for v in np.fft.ifft(np.array(Hs)).tolist()][:n])
    e = np.array([v.real for v in np.fft.ifft(np.array(Es)).tolist()][:n])
    cepstrum = np.array(cep)

    return SignalResult(
        name="homomorphic_deconvolve",
        filtered=h,
        fs=0.0,
        n_samples=len(h),
        extra={"excitation": e, "cutoff": cutoff, "n_fft": n_fft,
               "delay": nd, "cepstrum": cepstrum},
    )


hdecon = homomorphic_deconvolve


def cheatsheet() -> str:
    return "homomorphic_deconvolve({}) -> Homomorphic deconvolution via cepstral liftering."
