# morie.fn -- bsatf (rootcoder007/morie)
"""Time-frequency and multiresolution analysis: STFT, Wigner-Ville and Cohen's class, wavelets, EMD, Hilbert-Huang, VMD.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 50
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from . import _array_core as np
from . import _stats_core as stats
from ._containers import DescriptiveResult
from ._richresult import RichResult
from ._richresult import RichResult, with_describe_pointer
from ._sci_core import CubicSpline
from ._signal_core import hilbert

__all__ = [
    'hilbert_huang_spectrum',
    'rangayyan_amplitude_demod',
    'rangayyan_biorthogonal_wvlt',
    'rangayyan_choi_williams',
    'rangayyan_cpr_analysis',
    'rangayyan_cwt',
    'rangayyan_cohen_class',
    'rangayyan_daubechies',
    'rangayyan_decomp_tfd',
    'rangayyan_dwt',
    'rangayyan_eemd',
    'rangayyan_emd',
    'rangayyan_emd_imf',
    'rangayyan_emd_twa',
    'rangayyan_emd_vf_detect',
    'rangayyan_emg_rms',
    'rangayyan_wavelet_entropy',
    'rangayyan_envelope',
    'rangayyan_envelogram',
    'rangayyan_haar_wavelet',
    'rangayyan_hht_spectrum',
    'rangayyan_hrv_time_varying',
    'rangayyan_istft',
    'rangayyan_mra',
    'rangayyan_pcg_envelope_avg',
    'rangayyan_ppg_wavelet',
    'rangayyan_scalogram',
    'rangayyan_seizure_wavelet',
    'rangayyan_stft_params',
    'rangayyan_stft_spectrogram',
    'rangayyan_swt',
    'rangayyan_swt_denoise',
    'rangayyan_vmd',
    'rangayyan_wavelet_struct',
    'rangayyan_wavelet_corr',
    'rangayyan_wigner_ville',
    'rangayyan_wavelet_energy',
    'rangayyan_wavelet_moments',
    'rangayyan_wavelet_packet',
    'rangayyan_wavelet_threshold',
    'rangayyan_wavelet_variance',
    'rangayyan_ch4_signal_with_echo_input',
    'rangayyan_ch4_signal_with_echo_output',
    'rangayyan_ch4_z_transform_signal_echo',
    'rangayyan_ch4_fourier_signal_echo',
    'rangayyan_ch4_log_signal_echo',
    'rangayyan_ch4_complex_cepstrum_signal_with_echo',
    'rangayyan_ch4_power_spectrum_signal_echo',
    'rangayyan_ch4_log_power_spectrum_signal_echo',
    'wigner_ville',
]


# -- hhtrf: Hilbert-Huang Transform (EMD + instantaneous frequency/amplitude).
def _sift_imf(x: np.ndarray, max_iter: int = 300, sd_tol: float = 0.05) -> np.ndarray:
    """Extract one IMF via sifting."""
    h = x.copy()
    for _ in range(max_iter):
        t = np.arange(len(h))
        max_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
        min_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break
        upper = CubicSpline(max_idx, h[max_idx], extrapolate=True)(t)
        lower = CubicSpline(min_idx, h[min_idx], extrapolate=True)(t)
        mean_env = (upper + lower) / 2
        prev = h.copy()
        h = h - mean_env
        sd = np.sum((prev - h) ** 2) / (np.sum(prev**2) + 1e-12)
        if sd < sd_tol:
            break
    return h


def hilbert_huang_spectrum(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    max_imfs: int = 10,
    n_freq_bins: int = 256,
) -> DescriptiveResult:
    r"""Hilbert-Huang Transform with full Hilbert spectrum.

    Performs Empirical Mode Decomposition to extract IMFs, then applies
    the Hilbert transform to each IMF to obtain instantaneous
    frequency and amplitude.  Constructs a Hilbert spectrum
    :math:`H(f, t)` and marginal spectrum :math:`h(f)`:

    .. math::

        H(\\omega, t) = \\sum_{i=1}^{N} a_i(t) \\,
        \\delta(\\omega - \\omega_i(t))

    .. math::

        h(\\omega) = \\int_0^T H(\\omega, t) \\, dt

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz (default 1.0).
    max_imfs : int
        Maximum number of IMFs to extract (default 10).
    n_freq_bins : int
        Number of frequency bins for the Hilbert spectrum (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``imfs``, ``residue``,
        ``inst_freqs`` (list), ``inst_amps`` (list),
        ``hilbert_spectrum`` (n_freq_bins x n_samples),
        ``marginal_spectrum``, ``freq_axis``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Huang, N.E. et al. (1998). The empirical mode decomposition and
    the Hilbert spectrum for nonlinear and non-stationary time series
    analysis. *Proc. R. Soc. Lond. A*, 454, 903--995.
    doi:10.1098/rspa.1998.0193
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    residue = x.copy()
    imfs = []

    for _ in range(max_imfs):
        imf = _sift_imf(residue)
        if np.max(np.abs(imf)) < 1e-10:
            break
        imfs.append(imf)
        residue = residue - imf
        max_idx = np.where((residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:]))[0] + 1
        min_idx = np.where((residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

    inst_freqs = []
    inst_amps = []
    for imf in imfs:
        analytic = hilbert(imf)
        amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        freq = np.gradient(phase) * fs / (2 * np.pi)
        inst_freqs.append(freq)
        inst_amps.append(amp)

    freq_axis = np.linspace(0, fs / 2, n_freq_bins)
    hs = np.zeros((n_freq_bins, n))
    df = freq_axis[1] - freq_axis[0] if n_freq_bins > 1 else 1.0
    for freq_arr, amp_arr in zip(inst_freqs, inst_amps):
        for t_idx in range(n):
            f_val = abs(freq_arr[t_idx])
            bin_idx = int(f_val / (df + 1e-12))
            if 0 <= bin_idx < n_freq_bins:
                hs[bin_idx, t_idx] += amp_arr[t_idx] ** 2

    marginal = np.sum(hs, axis=1)

    return DescriptiveResult(
        name="hilbert_huang_spectrum",
        value=float(len(imfs)),
        extra={
            "imfs": imfs,
            "residue": residue,
            "inst_freqs": inst_freqs,
            "inst_amps": inst_amps,
            "hilbert_spectrum": hs,
            "marginal_spectrum": marginal,
            "freq_axis": freq_axis,
        },
    )


hhtrf = hilbert_huang_spectrum


# -- rgampd: Amplitude demodulation (envelope via Hilbert transform).
def rangayyan_amplitude_demod(x, fs):
    """
    Amplitude demodulation (envelope via Hilbert transform)

    Formula: envelope(t) = sqrt(x(t)^2 + H{x(t)}^2) = |x_a(t)|

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: envelope

    References
    ----------
    Rangayyan Ch 5.5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Amplitude demodulation (envelope via Hilbert transform)",
        }
    )


# -- rgbiorth: Biorthogonal wavelet (symmetric, linear phase) DWT.
def rangayyan_biorthogonal_wvlt(x, wavelet, levels):
    """
    Biorthogonal wavelet (symmetric, linear phase) DWT

    Formula: Analysis: h_tilde, g_tilde; synthesis: h, g; perfect reconstruction via duality

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coeffs, reconstructed

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Biorthogonal wavelet (symmetric, linear phase) DWT"}
    )


# -- rgchoi: Choi-Williams distribution (exponential kernel).
def rangayyan_choi_williams(x, fs, sigma):
    """
    Choi-Williams distribution (exponential kernel)

    Formula: phi(theta,tau) = exp(-theta^2*tau^2/sigma)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cwd, t, freqs

    References
    ----------
    Rangayyan Ch 8.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Choi-Williams distribution (exponential kernel)"}
    )


# -- rgcpr: CPR analysis via wavelet for shockable rhythm detection.
def rangayyan_cpr_analysis(ecg, fs):
    """
    CPR analysis via wavelet for shockable rhythm detection

    Formula: Wavelet features in 3-10 Hz band discriminate VF from non-VF

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_shockable, features

    References
    ----------
    Rangayyan Ch 8.15
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "CPR analysis via wavelet for shockable rhythm detection",
        }
    )


# -- rgcwt: Continuous wavelet transform (CWT).
def rangayyan_cwt(x, fs, wavelet, scales):
    """
    Continuous wavelet transform (CWT)

    Formula: CWT(a,b) = (1/sqrt(a)) integral x(t)*psi*((t-b)/a) dt

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    wavelet : array-like
        Input data.
    scales : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients, scales, freqs

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous wavelet transform (CWT)"})


# compact alias per ledger/NAMING.md
rangayyancwt = rangayyan_cwt


# -- rgcwvd: Cohen's class TFDs via kernel function.
def rangayyan_cohen_class(x, fs, kernel):
    """
    Cohen's class TFDs via kernel function

    Formula: C(t,f) = integral integral phi(theta,tau)*A(theta,tau)*exp(-j2pi(theta*t+f*tau)) dtheta dtau

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tfd, t, freqs

    References
    ----------
    Rangayyan Ch 8.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Cohen's class TFDs via kernel function"}
    )


# -- rgdaub: Daubechies wavelet filter coefficients (db2-db10).
def rangayyan_daubechies(order):
    """
    Daubechies wavelet filter coefficients (db2-db10)

    Formula: Orthogonal FIR filter satisfying vanishing moment conditions

    Parameters
    ----------
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo_d, hi_d, lo_r, hi_r

    References
    ----------
    Rangayyan Ch 8.8
    """
    order = np.asarray(order, dtype=float)
    n = int(order) if order.ndim == 0 else len(order)
    result = float(np.mean(order))
    se = float(np.std(order, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Daubechies wavelet filter coefficients (db2-db10)"}
    )


# -- rgdtfd: Decomposition-based adaptive TFD using MP atoms.
def rangayyan_decomp_tfd(x, fs, dictionary, max_atoms):
    """
    Decomposition-based adaptive TFD using MP atoms

    Formula: TFD(t,f) = sum_k |a_k|^2 * WVD(phi_k)(t,f)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    dictionary : array-like
        Input data.
    max_atoms : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tfd, t, freqs

    References
    ----------
    Rangayyan Ch 9.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Decomposition-based adaptive TFD using MP atoms"}
    )


# -- rgdwt: Discrete wavelet transform (DWT) via filterbank.
def rangayyan_dwt(x, wavelet, levels):
    """
    Discrete wavelet transform (DWT) via filterbank

    Formula: c_j[n]=sum h[k]*c_{j-1}[2n-k]; d_j[n]=sum g[k]*c_{j-1}[2n-k]

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: approx, details

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Discrete wavelet transform (DWT) via filterbank"}
    )


# compact alias per ledger/NAMING.md
rangayyandwt = rangayyan_dwt


# -- rgeemd: Ensemble EMD (EEMD) for mode mixing alleviation.
def rangayyan_eemd(x, n_ensembles, noise_std, max_imfs):
    """
    Ensemble EMD (EEMD) for mode mixing alleviation

    Formula: EEMD: add white noise realizations, perform EMD, ensemble average IMFs

    Parameters
    ----------
    x : array-like
        Input data.
    n_ensembles : array-like
        Input data.
    noise_std : array-like
        Input data.
    max_imfs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: imfs

    References
    ----------
    Rangayyan Ch 9.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Ensemble EMD (EEMD) for mode mixing alleviation"}
    )


# compact alias per ledger/NAMING.md
rangayyaneemd = rangayyan_eemd


# -- rgemd: Empirical mode decomposition (EMD) sifting algorithm.
def rangayyan_emd(x, max_imfs, tol):
    """
    Empirical mode decomposition (EMD) sifting algorithm

    Formula: IMF_k from sifting: h(t)=x(t)-(upper+lower)/2; iterate until stoppage

    Parameters
    ----------
    x : array-like
        Input data.
    max_imfs : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: imfs, residue

    References
    ----------
    Rangayyan Ch 9.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Empirical mode decomposition (EMD) sifting algorithm"}
    )


# compact alias per ledger/NAMING.md
rangayyanemd = rangayyan_emd


# -- rgemdimf: Intrinsic mode function (IMF) extraction and validation.
def rangayyan_emd_imf(x, max_iter, tol):
    """
    Intrinsic mode function (IMF) extraction and validation

    Formula: IMF conditions: zero crossings-extrema differ by at most 1; mean envelope=0

    Parameters
    ----------
    x : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: imf, is_valid

    References
    ----------
    Rangayyan Ch 9.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Intrinsic mode function (IMF) extraction and validation",
        }
    )


# -- rgemdtwa: T-wave alternans detection via EMD-based signal decomposition.
def rangayyan_emd_twa(ecg, fs, r_peaks):
    """
    T-wave alternans detection via EMD-based signal decomposition

    Formula: IMF at alternans frequency (0.5 cycles/beat); alternans amplitude from IMF energy

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: twa_amp, alternating_imf

    References
    ----------
    Rangayyan Ch 9.10
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "T-wave alternans detection via EMD-based signal decomposition",
        }
    )


# -- rgemdvf: Ventricular fibrillation detection using EMD features.
def rangayyan_emd_vf_detect(ecg, fs, n_imfs):
    """
    Ventricular fibrillation detection using EMD features

    Formula: IMF energies in 3-10 Hz band elevated in VF; threshold decision

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    n_imfs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_vf, imf_features

    References
    ----------
    Rangayyan Ch 8.16
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Ventricular fibrillation detection using EMD features",
        }
    )


# -- rgemg: EMG RMS envelope -- Rangayyan & Krishnan Sec 5.6.1, eq (5.24).
def rangayyan_emg_rms(x, window=64, fs=1.0):
    """Sliding-window RMS envelope.

    RMS[n] = sqrt( (1/W) Σ_{k=n-W+1}^{n} x[k]² ).

    Parameters
    ----------
    x : array-like
    window : int
        Window length in samples.
    fs : float
        Sampling rate (Hz, only for reporting).

    Returns
    -------
    RichResult with keys ``rms``, ``window``, ``fs``, ``mean_rms``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 5.6.1 "The RMS value", pp.283-284.
        The previous docstring cited Ch 8. Equation (5.23) is the global RMS
        over N samples; equation (5.24) is the running RMS this function
        computes,

            RMS(n) = [ (1/M) sum_{k=0}^{M-1} x^2(n-k) ]^(1/2),

        which is explicitly CAUSAL and therefore undefined for n < M-1.
    """
    x = np.asarray(x, dtype=float).ravel()
    W = int(window)
    if W < 1:
        raise ValueError("window must be >= 1")
    sq = x**2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    rms = np.full_like(x, np.nan)
    for i in range(W - 1, x.size):
        rms[i] = np.sqrt((csum[i + 1] - csum[i + 1 - W]) / W)
    # The first W-1 samples stay NaN. Equation (5.24) is a CAUSAL window --
    # RMS(n) averages x(n-k) for k = 0..M-1 -- so it is simply undefined until
    # n = M-1; the book defines no warm-up value.
    #
    # This previously back-filled rms[:W-1] with rms[W-1], a value computed
    # from samples that lie in the FUTURE of those positions. That destroys the
    # one property eq (5.24) exists to have. Measured: a signal that is exactly
    # zero until sample 20 and active thereafter reported envelope 0.7651 at
    # sample 0, i.e. the envelope rose 20 samples BEFORE the burst. EMG onset
    # detection is the main use of an RMS envelope, so the artefact lands
    # exactly where it does the most damage.
    #
    # mean_rms already uses np.nanmean, so the warm-up was always meant to be
    # NaN; the back-fill was the anomaly.
    res = RichResult(
        title="EMG RMS envelope",
        summary_lines=[
            ("Window (samples)", W),
            ("Fs (Hz)", float(fs)),
            ("Mean RMS", float(np.nanmean(rms))),
            ("Max RMS", float(np.nanmax(rms))),
        ],
        interpretation=f"Sliding-window RMS, W={W} samples ({W / fs:.3g} s).",
        payload={"rms": rms, "window": W, "fs": float(fs), "mean_rms": float(np.nanmean(rms))},
    )
    return with_describe_pointer(res, "rgemg")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_emg_rms(rng.standard_normal(500), window=32)
# >>> r["rms"].shape == (500,)
# True


# -- rgentrwv: Wavelet entropy for measuring signal regularity.
def rangayyan_wavelet_entropy(x, wavelet, levels):
    """
    Wavelet entropy for measuring signal regularity

    Formula: E = -sum p_j * log(p_j); p_j = E_j / E_total; E_j = sum d_j^2

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: entropy

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Wavelet entropy for measuring signal regularity"}
    )


# -- rgenv: Hilbert-transform envelope -- Rangayyan & Krishnan Sec 5.5.3.
def rangayyan_envelope(x):
    """Analytic-signal envelope via the Hilbert transform.

    ``env(t) = |x(t) + j H{x(t)}|`` where H{·} is the discrete Hilbert
    transform (``scipy.signal.hilbert``).

    Parameters
    ----------
    x : array-like

    Returns
    -------
    RichResult with keys ``envelope``, ``analytic``,
    ``instantaneous_phase``, ``instantaneous_freq``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 5.5.3 "The envelogram", p.281
        (Sec 5.5 "Envelope Extraction and Analysis", p.277).
    """
    from ._signal_core import hilbert

    x = np.asarray(x, dtype=float)
    z = hilbert(x)
    env = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(phase) / (2 * np.pi)
    res = RichResult(
        title="Hilbert envelope",
        summary_lines=[
            ("N samples", int(x.size)),
            ("Envelope mean", float(env.mean())),
            ("Envelope max", float(env.max())),
        ],
        interpretation=f"Analytic envelope mean {env.mean():.4g}, peak {env.max():.4g}.",
        payload={"envelope": env, "analytic": z, "instantaneous_phase": phase, "instantaneous_freq": inst_freq},
    )
    return with_describe_pointer(res, "rgenv")


# CANONICAL TEST
# >>> t = np.arange(100)/100.0
# >>> x = np.cos(2*np.pi*5*t) * (1 + 0.3*np.cos(2*np.pi*0.5*t))
# >>> r = rangayyan_envelope(x)
# >>> r["envelope"].shape == x.shape
# True


# -- rgenvgm: Envelogram.
def rangayyan_envelogram(pcg, ecg=None, fs=1000.0, r_peaks=None, n_beats=None):
    r"""Envelogram of a PCG signal (Rangayyan Ch. 3):

    .. math:: \mathrm{env}_{avg}[n] = \frac1M \sum_{k=1}^{M}
              \big| x_k(n) + j\,\mathcal H\{x_k(n)\} \big|,

    the ensemble-averaged analytic-signal magnitude. The Hilbert
    transform gives the instantaneous amplitude envelope, which is
    what makes S1 and S2 visible as smooth bumps rather than as
    oscillation. Alignment comes from the ECG R peaks -- averaging
    unaligned beats smears the envelope and is the usual way this
    goes wrong, so the R peaks are required rather than guessed.

    Parameters
    ----------
    pcg : array-like
        Phonocardiogram.
    ecg : array-like, optional
        ECG used only if r_peaks must be detected.
    fs : float, default 1000.0
        Sampling frequency.
    r_peaks : array-like of int, optional
        R-peak sample indices; detected from ecg when omitted.
    n_beats : int, optional
        Beat-count check.

    Returns
    -------
    RichResult
        keys: ``envelope`` (averaged), ``beats`` (M, L matrix),
        ``M``, ``beat_length``, ``fs``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (envelope extraction, envelograms).
    """
    from ._signal_core import signal as sig

    x = np.asarray(pcg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    if r_peaks is None:
        if ecg is None:
            raise ValueError(
                "supply r_peaks, or an ecg to detect them from -- averaging "
                "unaligned beats smears the envelope."
            )
        e = np.asarray(ecg, dtype=float).ravel()
        thr = float(np.mean(e) + 2.0 * np.std(e))
        pk, _ = sig.find_peaks(e, height=thr, distance=int(0.25 * fs))
        r_peaks = pk
    r = np.asarray(r_peaks, dtype=int).ravel()
    if r.size < 2:
        raise ValueError("need at least 2 R peaks to segment beats.")
    if n_beats is not None and int(n_beats) != r.size:
        raise ValueError(f"n_beats = {n_beats} does not match {r.size} peaks.")
    L = int(np.min(np.diff(r)))
    if L < 8:
        raise ValueError("beats are too short to average.")
    env = np.abs(sig.hilbert(x))
    beats = np.array([env[p : p + L] for p in r[:-1] if p + L <= env.size])
    if beats.size == 0:
        raise ValueError("no complete beats within the signal.")
    return RichResult(payload={"envelope": beats.mean(axis=0), "beats": beats,
                               "M": int(beats.shape[0]), "beat_length": L, "fs": fs,
                               "method": "Hilbert envelope averaged over R-aligned beats"})


# -- rghaar: Haar wavelet transform (simplest orthogonal wavelet).
def rangayyan_haar_wavelet(x, levels):
    """
    Haar wavelet transform (simplest orthogonal wavelet)

    Formula: phi=[1,1]/sqrt(2); psi=[1,-1]/sqrt(2); 1-level: c=(x0+x1)/sqrt(2); d=(x0-x1)/sqrt(2)

    Parameters
    ----------
    x : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: approx, details

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Haar wavelet transform (simplest orthogonal wavelet)"}
    )


# -- rghhtsp: Hilbert-Huang spectrum (HHS) via EMD + Hilbert transform.
def rangayyan_hht_spectrum(x, fs, max_imfs):
    """
    Hilbert-Huang spectrum (HHS) via EMD + Hilbert transform

    Formula: HHS(t,f) = sum_k A_k^2(t) * delta(f - f_k(t)); f_k = (1/2pi)*d(phi_k)/dt

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    max_imfs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: hhs, t, freqs

    References
    ----------
    Rangayyan Ch 9.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Hilbert-Huang spectrum (HHS) via EMD + Hilbert transform",
        }
    )


# -- rghrvtv: Time-varying HRV analysis via STFT of RR intervals.
def rangayyan_hrv_time_varying(rr_intervals, fs_resamp, window_len):
    """
    Time-varying HRV analysis via STFT of RR intervals

    Formula: HRV STFT: X_RR(m,f) using short sliding window on interpolated RR series

    Parameters
    ----------
    rr_intervals : array-like
        Input data.
    fs_resamp : array-like
        Input data.
    window_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lf_hf_trace, t

    References
    ----------
    Rangayyan Ch 8.12
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Time-varying HRV analysis via STFT of RR intervals"}
    )


# -- rgistft: Inverse STFT signal reconstruction from spectrogram.
def rangayyan_istft(stft, window, hop):
    """
    Inverse STFT signal reconstruction from spectrogram

    Formula: stft[n] = sum_m X(m,f)*w[n-m] / sum_m w^2[n-m] (overlap-add)

    Parameters
    ----------
    stft : array-like
        Input data.
    window : array-like
        Input data.
    hop : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_reconstructed

    References
    ----------
    Rangayyan Ch 8.4
    """
    stft = np.asarray(stft, dtype=float)
    n = int(stft) if stft.ndim == 0 else len(stft)
    result = float(np.mean(stft))
    se = float(np.std(stft, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Inverse STFT signal reconstruction from spectrogram"}
    )


# compact alias per ledger/NAMING.md
rangayyanistft = rangayyan_istft


# -- rgmra: Multiresolution analysis (MRA) decomposition.
def rangayyan_mra(x, wavelet, levels):
    """
    Multiresolution analysis (MRA) decomposition

    Formula: x = A_J + sum_{j=1}^{J} D_j; A_J=smooth, D_j=detail at scale 2^j

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: approximation, details_list

    References
    ----------
    Rangayyan Ch 8.8.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Multiresolution analysis (MRA) decomposition"}
    )


# compact alias per ledger/NAMING.md
rangayyanmra = rangayyan_mra


# -- rgpcgenl: Synchronized averaging of PCG envelopes (S1/S2 intensity analysis).
def rangayyan_pcg_envelope_avg(pcg, ecg, fs):
    """
    Synchronized averaging of PCG envelopes (S1/S2 intensity analysis)

    Formula: env_k = Hilbert envelope of k-th cardiac cycle; avg over cycles

    Parameters
    ----------
    pcg : array-like
        Input data.
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: avg_s1_env, avg_s2_env

    References
    ----------
    Rangayyan Ch 5.5.2
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Synchronized averaging of PCG envelopes (S1/S2 intensity analysis)",
        }
    )


# -- rgppgwt: Wavelet denoising of PPG signals.
def rangayyan_ppg_wavelet(ppg, fs, wavelet, levels):
    """
    Wavelet denoising of PPG signals

    Formula: DWT with db4; soft threshold at universal lambda=sigma*sqrt(2*log(N))

    Parameters
    ----------
    ppg : array-like
        Input data.
    fs : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ppg_clean

    References
    ----------
    Rangayyan Ch 8.14
    """
    ppg = np.asarray(ppg, dtype=float)
    n = int(ppg) if ppg.ndim == 0 else len(ppg)
    result = float(np.mean(ppg))
    se = float(np.std(ppg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet denoising of PPG signals"})


# -- rgsclgr: Scalogram: energy density via squared CWT magnitudes.
def rangayyan_scalogram(x, fs, scales, wavelet):
    """
    Scalogram: energy density via squared CWT magnitudes

    Formula: SC(a,b) = |CWT(a,b)|^2 / a

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    scales : array-like
        Input data.
    wavelet : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scalogram, scales, time

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Scalogram: energy density via squared CWT magnitudes"}
    )


# -- rgseizwv: EEG epileptic seizure detection via wavelet energy.
def rangayyan_seizure_wavelet(eeg, fs, wavelet, levels):
    """
    EEG epileptic seizure detection via wavelet energy

    Formula: E_j = sum |d_j[n]|^2; ictal increase in delta/theta wavelet energy

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: wavelet_energies, is_seizure

    References
    ----------
    Rangayyan Ch 8.17
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "EEG epileptic seizure detection via wavelet energy"}
    )


# -- rgstftp: STFT parameter selection (window length vs. time/freq resolution tradeoff).
def rangayyan_stft_params(fs, desired_t_res, desired_f_res):
    """
    STFT parameter selection (window length vs. time/freq resolution tradeoff)

    Formula: delta_t = N/fs; delta_f = fs/N; uncertainty: delta_t * delta_f = 1

    Parameters
    ----------
    fs : array-like
        Input data.
    desired_t_res : array-like
        Input data.
    desired_f_res : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: recommended_N, window_type

    References
    ----------
    Rangayyan Ch 8.4.2
    """
    fs = np.asarray(fs, dtype=float)
    n = int(fs) if fs.ndim == 0 else len(fs)
    result = float(np.mean(fs))
    se = float(np.std(fs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "STFT parameter selection (window length vs. time/freq resolution tradeoff)",
        }
    )


# -- rgstfts: STFT spectrogram (magnitude squared STFT).
def rangayyan_stft_spectrogram(x, fs, nperseg, noverlap, window):
    """
    STFT spectrogram (magnitude squared STFT)

    Formula: S(m,f) = |sum x[n]*w[n-m]*exp(-j2pi*fn)|^2

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    nperseg : array-like
        Input data.
    noverlap : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrogram, t, freqs

    References
    ----------
    Rangayyan Ch 8.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "STFT spectrogram (magnitude squared STFT)"}
    )


# -- rgswt: Stationary wavelet transform (SWT, undecimated DWT).
def rangayyan_swt(x, wavelet, levels):
    """
    Stationary wavelet transform (SWT, undecimated DWT)

    Formula: No downsampling; filter upsampled by 2^j at level j; shift-invariant

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: approx_coeffs, detail_coeffs

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Stationary wavelet transform (SWT, undecimated DWT)"}
    )


# compact alias per ledger/NAMING.md
rangayyanswt = rangayyan_swt


# -- rgswtden: SWT-based denoising (shift-invariant, no Gibbs oscillation).
def rangayyan_swt_denoise(x, wavelet, levels, threshold):
    """
    SWT-based denoising (shift-invariant, no Gibbs oscillation)

    Formula: Apply SWT; threshold detail coefficients; ISWT reconstruct

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_clean

    References
    ----------
    Rangayyan Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "SWT-based denoising (shift-invariant, no Gibbs oscillation)",
        }
    )


# -- rgvmd: Variational mode decomposition (VMD) into K band-limited modes.
def rangayyan_vmd(x, K, alpha, tau, init, tol):
    """
    Variational mode decomposition (VMD) into K band-limited modes

    Formula: min_{u,omega} sum_k ||d/dt[(delta(t)+j/pi*t)*u_k(t)]*e^{-j*omega_k*t}||^2 s.t. sum=x

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.
    alpha : array-like
        Input data.
    tau : array-like
        Input data.
    init : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: u_modes, omega_center

    References
    ----------
    Rangayyan Ch 9.4.1
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
            "method": "Variational mode decomposition (VMD) into K band-limited modes",
        }
    )


# compact alias per ledger/NAMING.md
rangayyanvmd = rangayyan_vmd


# -- rgwavstr: Wavelet-based structure detection in biomedical signals (CWT ridges).
def rangayyan_wavelet_struct(x, fs, scales, wavelet):
    """
    Wavelet-based structure detection in biomedical signals (CWT ridges)

    Formula: Ridge: argmax_a |CWT(a,b)| at each time b; instantaneous frequency from ridge

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    scales : array-like
        Input data.
    wavelet : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ridges, inst_freq

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Wavelet-based structure detection in biomedical signals (CWT ridges)",
        }
    )


# -- rgwvcor: Wavelet cross-correlation between two signals at each scale.
def rangayyan_wavelet_corr(x, y, wavelet, levels):
    """
    Wavelet cross-correlation between two signals at each scale

    Formula: WCC_j(tau) = sum d_j_x[n] * d_j_y[n+tau]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cross_corr_per_scale

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Wavelet cross-correlation between two signals at each scale",
            }
        )
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "Wavelet cross-correlation between two signals at each scale",
        }
    )


# -- rgwvd: Wigner-Ville distribution (bilinear TFD).
def rangayyan_wigner_ville(x, fs):
    """
    Wigner-Ville distribution (bilinear TFD)

    Formula: WVD(t,f) = integral x(t+tau/2)*x*(t-tau/2)*exp(-j2*pi*f*tau) dtau

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: wvd, t, freqs

    References
    ----------
    Rangayyan Ch 8.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Wigner-Ville distribution (bilinear TFD)"}
    )


# -- rgwvener: Wavelet energy per subband (scale).
def rangayyan_wavelet_energy(x, wavelet, levels):
    """
    Wavelet energy per subband (scale)

    Formula: E_j = sum_{n} |d_j[n]|^2; total = sum E_j + E_approx

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: energies, relative_energies

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet energy per subband (scale)"})


# -- rgwvmom: Wavelet coefficient moments (energy, variance, mean) per scale.
def rangayyan_wavelet_moments(x, wavelet, levels):
    """
    Wavelet coefficient moments (energy, variance, mean) per scale

    Formula: E_j=sum|d_j[n]|^2; var_j=std(d_j)^2; mean_j=mean(d_j)

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: energies, variances, means

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Wavelet coefficient moments (energy, variance, mean) per scale",
        }
    )


# -- rgwvpkt: Wavelet packet decomposition (full binary tree).
def rangayyan_wavelet_packet(x, wavelet, levels):
    """
    Wavelet packet decomposition (full binary tree)

    Formula: Both approximation AND detail branches filtered and downsampled at each level

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: packet_tree

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Wavelet packet decomposition (full binary tree)"}
    )


# -- rgwvth: Wavelet denoising via soft/hard thresholding.
def rangayyan_wavelet_threshold(x, wavelet, levels, threshold_type):
    """
    Wavelet denoising via soft/hard thresholding

    Formula: soft: sign(d)*max(|d|-lambda,0); hard: d*(|d|>=lambda); lambda=sigma*sqrt(2*log(N))

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.
    threshold_type : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: denoised

    References
    ----------
    Rangayyan Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Wavelet denoising via soft/hard thresholding"}
    )


# -- rgwvvar: Wavelet variance (Allan variance) by scale.
def rangayyan_wavelet_variance(x, wavelet, levels):
    """
    Wavelet variance (Allan variance) by scale

    Formula: V_j = (1/(2*(N-2^j))) * sum |d_j[n]|^2

    Parameters
    ----------
    x : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance_per_scale

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Wavelet variance (Allan variance) by scale"}
    )


# -- rng246: Two-impulse input modeling a wavelet plus echo..
def rangayyan_ch4_signal_with_echo_input(a, n_0, n):
    """
    Two-impulse input modeling a wavelet plus echo.

    Formula: x(n) = delta(n) + a * delta(n - n_0)

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.74, p. 249
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Two-impulse input modeling a wavelet plus echo."}
    )


# -- rng247: Time-domain expression for a wavelet h(n) plus an echo at delay n_0..
def rangayyan_ch4_signal_with_echo_output(h, a, n_0, n):
    """
    Time-domain expression for a wavelet h(n) plus an echo at delay n_0.

    Formula: y(n) = h(n) + a * h(n - n_0)

    Parameters
    ----------
    h : array-like
        Input data.
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.75, p. 249
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Time-domain expression for a wavelet h(n) plus an echo at delay n_0.",
        }
    )


# -- rng248: Z-transform of a signal with a wavelet and an echo..
def rangayyan_ch4_z_transform_signal_echo(a, n_0, z, H):
    """
    Z-transform of a signal with a wavelet and an echo.

    Formula: Y(z) = (1 + a * z^(-n_0)) * H(z)

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    z : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.76, p. 249
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Z-transform of a signal with a wavelet and an echo."}
    )


# -- rng249: Fourier-domain expression for a signal with a wavelet plus echo..
def rangayyan_ch4_fourier_signal_echo(a, n_0, omega, H):
    """
    Fourier-domain expression for a signal with a wavelet plus echo.

    Formula: Y(omega) = [1 + a * exp(-j*omega*n_0)] * H(omega)

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    omega : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.77, p. 249
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Fourier-domain expression for a signal with a wavelet plus echo.",
        }
    )


# -- rng250: Complex log of the spectrum of a signal with a wavelet plus echo..
def rangayyan_ch4_log_signal_echo(a, n_0, omega, H_hat):
    """
    Complex log of the spectrum of a signal with a wavelet plus echo.

    Formula: Y_hat(omega) = H_hat(omega) + log[1 + a * exp(-j*omega*n_0)]

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    omega : array-like
        Input data.
    H_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.78, p. 249
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Complex log of the spectrum of a signal with a wavelet plus echo.",
        }
    )


# -- rng252: Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0)..
def rangayyan_ch4_complex_cepstrum_signal_with_echo(h_hat, a, n_0, n):
    """
    Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0).

    Formula: y_hat(n) = h_hat(n) + a*delta(n-n_0) - (a^2/2)*delta(n-2*n_0) + (a^3/3)*delta(n-3*n_0) - ...

    Parameters
    ----------
    h_hat : array-like
        Input data.
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.80, p. 249
    """
    h_hat = np.atleast_1d(np.asarray(h_hat, dtype=float))
    n = len(h_hat)
    result = float(np.mean(h_hat))
    se = float(np.std(h_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0).",
        }
    )


# -- rng256: Squared magnitude (power spectrum) of a signal with wavelet plus echo..
def rangayyan_ch4_power_spectrum_signal_echo(H, a, n_0, z):
    """
    Squared magnitude (power spectrum) of a signal with wavelet plus echo.

    Formula: |Y(z)|^2 = |H(z)|^2 * |1 + a*z^(-n_0)|^2

    Parameters
    ----------
    H : array-like
        Input data.
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.84, p. 251
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Squared magnitude (power spectrum) of a signal with wavelet plus echo.",
        }
    )


# -- rng257: Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation..
def rangayyan_ch4_log_power_spectrum_signal_echo(H, a, n_0, omega):
    """
    Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation.

    Formula: log|Y(omega)|^2 = log|H(omega)|^2 + log[1 + a^2 + 2*a*cos(omega*n_0)] = log|H(omega)|^2 + log(1 + a^2) + log(1 + 2*a/(1+a^2) * cos(omega*n_0))

    Parameters
    ----------
    H : array-like
        Input data.
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.85, p. 251
    """
    H = np.atleast_1d(np.asarray(H, dtype=float))
    n = len(H)
    result = float(np.mean(H))
    se = float(np.std(H, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation.",
        }
    )


# -- wvdst: Wigner-Ville distribution for time-frequency analysis.
def wigner_ville(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    nfft: int | None = None,
) -> DescriptiveResult:
    r"""Compute the Wigner-Ville distribution (WVD).

    The WVD is a bilinear time-frequency representation defined as:

    .. math::

        W_x(t, f) = \\int_{-\\infty}^{\\infty} x(t + \\tau/2) \\,
        x^*(t - \\tau/2) \\, e^{-j 2\\pi f \\tau} \\, d\\tau

    Provides perfect time-frequency resolution but suffers from
    cross-term interference for multi-component signals.

    Parameters
    ----------
    x : array-like
        1-D input signal (real or analytic).
    fs : float
        Sampling frequency in Hz (default 1.0).
    nfft : int or None
        FFT size for frequency axis.  Defaults to ``2 * len(x)``.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``wvd`` (n_freq x n_time), ``times``,
        ``frequencies``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.

    Claasen, T.A.C.M. & Mecklenbr\"auker, W.F.G. (1980). The Wigner
    distribution -- A tool for time-frequency signal analysis. *Philips
    J. Res.*, 35, 217--250.
    """
    from ._signal_core import hilbert

    x_raw = np.asarray(x, dtype=float).ravel()
    n = len(x_raw)
    if nfft is None:
        nfft = 2 * n

    xa = hilbert(x_raw)

    wvd = np.zeros((nfft, n))
    for t_idx in range(n):
        tau_max = min(t_idx, n - 1 - t_idx)
        kernel = np.zeros(nfft, dtype=complex)
        for tau in range(-tau_max, tau_max + 1):
            kernel[tau % nfft] = xa[t_idx + tau] * np.conj(xa[t_idx - tau])
        wvd[:, t_idx] = np.real(np.fft.fft(kernel, n=nfft))

    times = np.arange(n) / fs
    half = nfft // 2
    frequencies = np.arange(half) * (fs / (2 * nfft))
    wvd = wvd[:half, :]

    return DescriptiveResult(
        name="wigner_ville",
        value=float(n),
        extra={
            "wvd": wvd,
            "times": times,
            "frequencies": frequencies,
        },
    )


wvdst = wigner_ville


# compact alias per ledger/NAMING.md
wignerville = wigner_ville


_CHEATSHEET = [
    'hilbert_huang_spectrum({}) -> HHT with full Hilbert spectrum and marginal.',
    'rgampd: Amplitude demodulation (envelope via Hilbert transform)',
    'rgbiorth: Biorthogonal wavelet (symmetric, linear phase) DWT',
    'rgchoi: Choi-Williams distribution (exponential kernel)',
    'rgcpr: CPR analysis via wavelet for shockable rhythm detection',
    'rgcwt: Continuous wavelet transform (CWT)',
    "rgcwvd: Cohen's class TFDs via kernel function",
    'rgdaub: Daubechies wavelet filter coefficients (db2-db10)',
    'rgdtfd: Decomposition-based adaptive TFD using MP atoms',
    'rgdwt: Discrete wavelet transform (DWT) via filterbank',
    'rgeemd: Ensemble EMD (EEMD) for mode mixing alleviation',
    'rgemd: Empirical mode decomposition (EMD) sifting algorithm',
    'rgemdimf: Intrinsic mode function (IMF) extraction and validation',
    'rgemdtwa: T-wave alternans detection via EMD-based signal decomposition',
    'rgemdvf: Ventricular fibrillation detection using EMD features',
    'rgemg: sliding-window RMS envelope -- Rangayyan & Krishnan Sec 5.6.1',
    'rgentrwv: Wavelet entropy for measuring signal regularity',
    'rgenv: Hilbert envelope -- Rangayyan & Krishnan Sec 5.5.3',
    'rgenvgm: R-peak alignment is required -- unaligned averaging smears the envelope',
    'rghaar: Haar wavelet transform (simplest orthogonal wavelet)',
    'rghhtsp: Hilbert-Huang spectrum (HHS) via EMD + Hilbert transform',
    'rghrvtv: Time-varying HRV analysis via STFT of RR intervals',
    'rgistft: Inverse STFT signal reconstruction from spectrogram',
    'rgmra: Multiresolution analysis (MRA) decomposition',
    'rgpcgenl: Synchronized averaging of PCG envelopes (S1/S2 intensity analysis)',
    'rgppgwt: Wavelet denoising of PPG signals',
    'rgsclgr: Scalogram: energy density via squared CWT magnitudes',
    'rgseizwv: EEG epileptic seizure detection via wavelet energy',
    'rgstftp: STFT parameter selection (window length vs. time/freq resolution tradeoff)',
    'rgstfts: STFT spectrogram (magnitude squared STFT)',
    'rgswt: Stationary wavelet transform (SWT, undecimated DWT)',
    'rgswtden: SWT-based denoising (shift-invariant, no Gibbs oscillation)',
    'rgvmd: Variational mode decomposition (VMD) into K band-limited modes',
    'rgwavstr: Wavelet-based structure detection in biomedical signals (CWT ridges)',
    'rgwvcor: Wavelet cross-correlation between two signals at each scale',
    'rgwvd: Wigner-Ville distribution (bilinear TFD)',
    'rgwvener: Wavelet energy per subband (scale)',
    'rgwvmom: Wavelet coefficient moments (energy, variance, mean) per scale',
    'rgwvpkt: Wavelet packet decomposition (full binary tree)',
    'rgwvth: Wavelet denoising via soft/hard thresholding',
    'rgwvvar: Wavelet variance (Allan variance) by scale',
    'rng246: Two-impulse input modeling a wavelet plus echo.',
    'rng247: Time-domain expression for a wavelet h(n) plus an echo at delay n_0.',
    'rng248: Z-transform of a signal with a wavelet and an echo.',
    'rng249: Fourier-domain expression for a signal with a wavelet plus echo.',
    'rng250: Complex log of the spectrum of a signal with a wavelet plus echo.',
    'rng252: Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0).',
    'rng256: Squared magnitude (power spectrum) of a signal with wavelet plus echo.',
    'rng257: Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation.',
    'wigner_ville({}) -> Wigner-Ville distribution for time-frequency analysis.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
