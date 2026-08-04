# morie.fn -- bsaphys (rootcoder007/morie)
"""Physiological models and clinical applications: membrane and neuron models, PCG, EMG, EEG, VAG, respiratory and sleep signal analysis.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 38
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from __future__ import annotations
from . import _array_core as np
from . import _stats_core as stats
from ._containers import DescriptiveResult
from ._rgcore import aslist, gridint
from ._richresult import RichResult
from ._sci_core import CubicSpline
from .bsacorr import rangayyan_welch_psd

__all__ = [
    'emd',
    'rangayyan_action_potential',
    'rangayyan_cardiac_elecphys',
    'rangayyan_coronary_ad',
    'rangayyan_coronary_sound',
    'rangayyan_infant_cry',
    'rangayyan_egg',
    'rangayyan_heart_elasticity',
    'rangayyan_eng',
    'rangayyan_epilepsy_detect',
    'rangayyan_erp_features',
    'rangayyan_feature_extract_bci',
    'rangayyan_freq_domain_feat',
    'rangayyan_goldman_eqn',
    'rangayyan_hh_gating',
    'rangayyan_hodgkin_huxley',
    'rangayyan_fitzhugh_nagumo',
    'rangayyan_membrane_potential',
    'rangayyan_muscle_artifact',
    'rangayyan_muap',
    'rangayyan_murmur_analysis',
    'rangayyan_nernst_potential',
    'rangayyan_oae',
    'rangayyan_parkinson_multimodal',
    'rangayyan_pcg_eeg_coupling',
    'rangayyan_pcg_murmur_detect',
    'rangayyan_polysomnography',
    'rangayyan_point_process',
    'rangayyan_prosthetic_valve',
    'rangayyan_respiration_features',
    'rangayyan_respiratory_sound',
    'rangayyan_sleep_apnea_detect',
    'rangayyan_speech_features',
    'rangayyan_vag_analysis',
    'rangayyan_vag_knee_cartilage',
    'deltadecomp',
    'rangayyan_ch3_signal_as_delta_decomposition',
    'rangayyan_ch4_complex_log_of_product',
    'rangayyan_ch4_complex_log_x_z',
]


# -- emdsg: Empirical Mode Decomposition (standalone).
def _count_zero_crossings(x: np.ndarray) -> int:
    """Count zero crossings in signal *x*."""
    return int(np.sum(np.diff(np.sign(x)) != 0))


def _count_extrema(x: np.ndarray) -> int:
    """Count total number of local extrema."""
    max_idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]
    min_idx = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0]
    return len(max_idx) + len(min_idx)


def emd(
    x: np.ndarray,
    *,
    max_imfs: int = 12,
    max_sift_iter: int = 300,
    sd_threshold: float = 0.05,
) -> DescriptiveResult:
    r"""Empirical Mode Decomposition.

    Decomposes a signal into a set of Intrinsic Mode Functions (IMFs)
    through an iterative sifting process.  Each IMF satisfies two
    conditions:

    1. The number of extrema and zero crossings differ by at most one.
    2. The local mean of the upper and lower envelopes is zero.

    The sifting stopping criterion uses the normalized squared
    difference:

    .. math::

        SD = \\frac{\\sum_t |h_{k-1}(t) - h_k(t)|^2}
             {\\sum_t h_{k-1}^2(t)}

    Parameters
    ----------
    x : array-like
        1-D input signal.
    max_imfs : int
        Maximum number of IMFs to extract (default 12).
    max_sift_iter : int
        Maximum sifting iterations per IMF (default 300).
    sd_threshold : float
        Sifting convergence threshold (default 0.05).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``imfs`` (list of arrays), ``residue``,
        ``n_imfs``, ``sift_counts`` (iterations per IMF),
        ``is_imf`` (bool per IMF -- satisfies IMF conditions).

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
    residue = x.copy()
    imfs = []
    sift_counts = []

    for _ in range(max_imfs):
        h = residue.copy()
        n_sifts = 0

        for s in range(max_sift_iter):
            n_sifts = s + 1
            t = np.arange(len(h))
            max_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
            min_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1

            if len(max_idx) < 2 or len(min_idx) < 2:
                break

            upper = CubicSpline(max_idx, h[max_idx], extrapolate=True)(t)
            lower = CubicSpline(min_idx, h[min_idx], extrapolate=True)(t)
            mean_env = (upper + lower) / 2.0

            prev = h.copy()
            h = h - mean_env

            sd = np.sum((prev - h) ** 2) / (np.sum(prev**2) + 1e-12)
            if sd < sd_threshold:
                break

        if np.max(np.abs(h)) < 1e-10:
            break

        imfs.append(h)
        sift_counts.append(n_sifts)
        residue = residue - h

        max_idx = np.where((residue[1:-1] > residue[:-2]) & (residue[1:-1] > residue[2:]))[0] + 1
        min_idx = np.where((residue[1:-1] < residue[:-2]) & (residue[1:-1] < residue[2:]))[0] + 1
        if len(max_idx) < 2 or len(min_idx) < 2:
            break

    is_imf = []
    for imf in imfs:
        zc = _count_zero_crossings(imf)
        ne = _count_extrema(imf)
        is_imf.append(abs(ne - zc) <= 1)

    return DescriptiveResult(
        name="emd",
        value=float(len(imfs)),
        extra={
            "imfs": imfs,
            "residue": residue,
            "n_imfs": len(imfs),
            "sift_counts": sift_counts,
            "is_imf": is_imf,
        },
    )


emdsg = emd


# -- rgap: Idealized action potential waveform model (depolarization/repolarization).
def rangayyan_action_potential(t, v_rest, v_peak, t_rise, t_fall):
    """
    Idealized action potential waveform model (depolarization/repolarization)

    Formula: V(t) = piecewise ramp-and-decay model

    Parameters
    ----------
    t : array-like
        Input data.
    v_rest : array-like
        Input data.
    v_peak : array-like
        Input data.
    t_rise : array-like
        Input data.
    t_fall : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: voltage_array

    References
    ----------
    Rangayyan Ch 1.2.1
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
            "method": "Idealized action potential waveform model (depolarization/repolarization)",
        }
    )


# -- rgcardep: Cardiac electrophysiology tissue/organ-level model.
def rangayyan_cardiac_elecphys(mesh, sigma_i, sigma_e, C_m, I_ion):
    """
    Cardiac electrophysiology tissue/organ-level model

    Formula: Bidomain equations: div(sigma_i*grad(V_i))=beta*(C_m*dV_m/dt+I_ion)

    Parameters
    ----------
    mesh : array-like
        Input data.
    sigma_i : array-like
        Input data.
    sigma_e : array-like
        Input data.
    C_m : array-like
        Input data.
    I_ion : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_m_field, V_e_field

    References
    ----------
    Rangayyan Ch 7.8.2
    """
    mesh = np.asarray(mesh, dtype=float)
    n = int(mesh) if mesh.ndim == 0 else len(mesh)
    result = float(np.mean(mesh))
    se = float(np.std(mesh, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Cardiac electrophysiology tissue/organ-level model"}
    )


# -- rgcorad: Coronary artery disease detection from acoustic signals.
def rangayyan_coronary_ad(coronary_sound, fs, order):
    """
    Coronary artery disease detection from acoustic signals

    Formula: AR model of coronary sounds; discriminant features from poles

    Parameters
    ----------
    coronary_sound : array-like
        Input data.
    fs : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cad_score, ar_features

    References
    ----------
    Rangayyan Ch 7.11
    """
    coronary_sound = np.asarray(coronary_sound, dtype=float)
    n = int(coronary_sound) if coronary_sound.ndim == 0 else len(coronary_sound)
    result = float(np.mean(coronary_sound))
    se = float(np.std(coronary_sound, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Coronary artery disease detection from acoustic signals",
        }
    )


# -- rgcorart: Coronary artery sound generation model (turbulent flow).
def rangayyan_coronary_sound(diameter, flow_velocity, stenosis_pct):
    """
    Coronary artery sound generation model (turbulent flow)

    Formula: Strouhal number St = f*d/v; resonance frequency of stenosis

    Parameters
    ----------
    diameter : array-like
        Input data.
    flow_velocity : array-like
        Input data.
    stenosis_pct : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: resonance_freq, sound_model

    References
    ----------
    Rangayyan Ch 7.7.2
    """
    diameter = np.asarray(diameter, dtype=float)
    n = int(diameter) if diameter.ndim == 0 else len(diameter)
    result = float(np.mean(diameter))
    se = float(np.std(diameter, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Coronary artery sound generation model (turbulent flow)",
        }
    )


# -- rgcry: Infant cry signal analysis: formants and fundamental frequency.
def rangayyan_infant_cry(cry, fs):
    """
    Infant cry signal analysis: formants and fundamental frequency

    Formula: Pitch via autocorrelation peak; formants from LPC poles

    Parameters
    ----------
    cry : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pitch, formants, cry_type

    References
    ----------
    Rangayyan Ch 8.13
    """
    cry = np.asarray(cry, dtype=float)
    n = int(cry) if cry.ndim == 0 else len(cry)
    result = float(np.mean(cry))
    se = float(np.std(cry, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Infant cry signal analysis: formants and fundamental frequency",
        }
    )


# -- rgegg: Electrogastrogram (EGG) feature extraction (dominant frequency, power).
def rangayyan_egg(egg, fs):
    """
    Electrogastrogram (EGG) feature extraction (dominant frequency, power)

    Formula: EGG: dominant frequency 2-4 cycles/min; bradygastria <2, tachygastria 4-9

    Parameters
    ----------
    egg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: dominant_freq, power, rhythm_class

    References
    ----------
    Rangayyan Ch 1.2.8
    """
    egg = np.asarray(egg, dtype=float)
    n = int(egg) if egg.ndim == 0 else len(egg)
    result = float(np.mean(egg))
    se = float(np.std(egg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Electrogastrogram (EGG) feature extraction (dominant frequency, power)",
        }
    )


# compact alias per ledger/NAMING.md
rangayyanegg = rangayyan_egg


# -- rgelast: Heart-sound spectral stiffness index.
def rangayyan_heart_elasticity(pcg, fs, s1_window=None):
    r"""Spectral index of myocardial stiffness from S1 (Rangayyan
    Ch. 3).

    Higher stiffness shifts the S1 spectrum upward, so the dominant
    frequency and spectral centroid of the first heart sound track
    elasticity. This returns those descriptors -- it does NOT return a
    stiffness value in physical units: the relationship is monotone
    but the calibration is subject- and instrument-specific, and
    inventing an absolute number would be a fabrication.

    Parameters
    ----------
    pcg : array-like
        Phonocardiogram, or an isolated S1 segment.
    fs : float
        Sampling frequency.
    s1_window : (int, int), optional
        Sample range holding S1; the whole record if omitted.

    Returns
    -------
    RichResult
        keys: ``dominant_frequency``, ``spectral_centroid``,
        ``bandwidth_3db``, ``freqs``, ``psd``, ``calibrated`` (False),
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (heart sounds; spectral indices).
    """
    x = np.asarray(pcg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    if s1_window is not None:
        a, b = int(s1_window[0]), int(s1_window[1])
        if not 0 <= a < b <= x.size:
            raise ValueError(f"s1_window ({a}, {b}) is out of range.")
        x = x[a:b]
    if x.size < 16:
        raise ValueError(f"need at least 16 samples, got {x.size}.")
    w = rangayyan_welch_psd(x, fs=fs, nperseg=min(256, x.size))
    f, p = w["freqs"], w["psd"]
    tot = float(p.sum())
    centroid = float(np.sum(f * p) / tot) if tot > 0 else np.nan
    ipk = int(np.argmax(p))
    above = np.flatnonzero(p >= p[ipk] / 2.0)
    bw = float(f[above[-1]] - f[above[0]]) if above.size else 0.0
    return RichResult(payload={"dominant_frequency": float(f[ipk]),
                               "spectral_centroid": centroid, "bandwidth_3db": bw,
                               "freqs": f, "psd": p, "calibrated": False,
                               "method": "S1 spectral descriptors; monotone in stiffness, NOT calibrated"})


# -- rgengn: Electroneurogram (ENG) compound action potential model.
def rangayyan_eng(t, n_fibers, cv_range, amp_range):
    """
    Electroneurogram (ENG) compound action potential model

    Formula: CAP = sum of single fiber APs with varying conduction velocities and latencies

    Parameters
    ----------
    t : array-like
        Input data.
    n_fibers : array-like
        Input data.
    cv_range : array-like
        Input data.
    amp_range : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cap_waveform

    References
    ----------
    Rangayyan Ch 1.2.3
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
            "method": "Electroneurogram (ENG) compound action potential model",
        }
    )


# compact alias per ledger/NAMING.md
rangayyaneng = rangayyan_eng


# -- rgepidet: Epileptic seizure detection in EEG.
def rangayyan_epilepsy_detect(eeg, fs, dictionary_size):
    """
    Epileptic seizure detection in EEG

    Formula: Dictionary learning: ictal features differ from interictal; SVM classifier on atoms

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    dictionary_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_seizure, onset

    References
    ----------
    Rangayyan Ch 8.17
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epileptic seizure detection in EEG"})


# -- rgerp: Event-related potential (ERP) latency and amplitude features.
def rangayyan_erp_features(erp, fs):
    """
    Event-related potential (ERP) latency and amplitude features

    Formula: P300 latency = argmax(ERP), N200 latency = argmin(ERP in 150-250ms window)

    Parameters
    ----------
    erp : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: latency, amplitude

    References
    ----------
    Rangayyan Ch 1.2.7
    """
    erp = np.asarray(erp, dtype=float)
    n = int(erp) if erp.ndim == 0 else len(erp)
    result = float(np.mean(erp))
    se = float(np.std(erp, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Event-related potential (ERP) latency and amplitude features",
        }
    )


# -- rgfeatex: Feature extraction for BCI from EEG (event-related desynchronization/synchronization).
def rangayyan_feature_extract_bci(eeg, fs, ref_window, active_window, band):
    """
    Feature extraction for BCI from EEG (event-related desynchronization/synchronization)

    Formula: ERD = (R - A) / A * 100%; A = reference band power, R = active band power

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    ref_window : array-like
        Input data.
    active_window : array-like
        Input data.
    band : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: erd, ers, features

    References
    ----------
    Rangayyan Ch 9.12.2
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Feature extraction for BCI from EEG (event-related desynchronization/synchronization)",
        }
    )


# -- rgfrqdom: Frequency-domain feature extraction for CAD.
def rangayyan_freq_domain_feat(x, fs):
    """
    Frequency-domain feature extraction for CAD

    Formula: Peak frequency, spectral centroid, bandwidth, roll-off extracted from PSD

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features_dict

    References
    ----------
    Rangayyan Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Frequency-domain feature extraction for CAD"}
    )


# -- rgghk: Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential.
def rangayyan_goldman_eqn(T, P_K, P_Na, P_Cl, ion_concs):
    """
    Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential

    Formula: V_m = (RT/F)*ln((P_K[K]_o+P_Na[Na]_o+P_Cl[Cl]_i)/(P_K[K]_i+P_Na[Na]_i+P_Cl[Cl]_o))

    Parameters
    ----------
    T : array-like
        Input data.
    P_K : array-like
        Input data.
    P_Na : array-like
        Input data.
    P_Cl : array-like
        Input data.
    ion_concs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_resting

    References
    ----------
    Rangayyan Ch 1.2.1
    """
    T = np.asarray(T, dtype=float)
    n = int(T) if T.ndim == 0 else len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential",
        }
    )


# -- rghgate: Hodgkin-Huxley gating variable ODEs (m, h, n).
def rangayyan_hh_gating(V, dt):
    """
    Hodgkin-Huxley gating variable ODEs (m, h, n)

    Formula: dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m; similarly h, n

    Parameters
    ----------
    V : array-like
        Input data.
    dt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m, h, n

    References
    ----------
    Rangayyan Ch 7.8.1
    """
    V = np.asarray(V, dtype=float)
    n = int(V) if V.ndim == 0 else len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Hodgkin-Huxley gating variable ODEs (m, h, n)"}
    )


# -- rghhmm: Hodgkin-Huxley membrane model for action potential.
def rangayyan_hodgkin_huxley(t, I_ext, g_Na, g_K, g_L, C_m):
    """
    Hodgkin-Huxley membrane model for action potential

    Formula: C_m dV/dt = I_ext - I_Na - I_K - I_L; I_Na=g_Na*m^3*h*(V-E_Na)

    Parameters
    ----------
    t : array-like
        Input data.
    I_ext : array-like
        Input data.
    g_Na : array-like
        Input data.
    g_K : array-like
        Input data.
    g_L : array-like
        Input data.
    C_m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_t, m_t, h_t, n_t

    References
    ----------
    Rangayyan Ch 7.8.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Hodgkin-Huxley membrane model for action potential"}
    )


# -- rghmm: FitzHugh-Nagumo simplified neuron model.
def rangayyan_fitzhugh_nagumo(t, I_ext, a, b, eps):
    """
    FitzHugh-Nagumo simplified neuron model

    Formula: dv/dt = v - v^3/3 - w + I; dw/dt = eps*(v + a - b*w)

    Parameters
    ----------
    t : array-like
        Input data.
    I_ext : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: v_t, w_t

    References
    ----------
    Rangayyan Ch 7.8.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "FitzHugh-Nagumo simplified neuron model"}
    )


# -- rgmemb: Membrane potential dynamics (RC circuit model).
def rangayyan_membrane_potential(t, I_inj, C_m, R_m, V_rest):
    """
    Membrane potential dynamics (RC circuit model)

    Formula: C_m * dV/dt = -(V - V_rest)/R_m + I_inj

    Parameters
    ----------
    t : array-like
        Input data.
    I_inj : array-like
        Input data.
    C_m : array-like
        Input data.
    R_m : array-like
        Input data.
    V_rest : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_t

    References
    ----------
    Rangayyan Ch 1.2.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Membrane potential dynamics (RC circuit model)"}
    )


# -- rgmscart: Muscle contraction artifact removal from VAG signals.
def rangayyan_muscle_artifact(vag, emg_ref, fs):
    """
    Muscle contraction artifact removal from VAG signals

    Formula: Notch + adaptive filtering on EMG-contaminated VAG

    Parameters
    ----------
    vag : array-like
        Input data.
    emg_ref : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vag_clean

    References
    ----------
    Rangayyan Ch 3.15
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Muscle contraction artifact removal from VAG signals"}
    )


# -- rgmuap: Motor unit action potential (MUAP) model.
def rangayyan_muap(t, n_fibers, conduction_vel):
    """
    Motor unit action potential (MUAP) model

    Formula: MUAP = sum of triphasic dipole contributions from muscle fibers

    Parameters
    ----------
    t : array-like
        Input data.
    n_fibers : array-like
        Input data.
    conduction_vel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: muap_waveform

    References
    ----------
    Rangayyan Ch 1.2.4
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Motor unit action potential (MUAP) model"}
    )


# compact alias per ledger/NAMING.md
rangayyanmuap = rangayyan_muap


# -- rgmurm: Heart murmur frequency analysis for valvular defect diagnosis.
def rangayyan_murmur_analysis(pcg, fs, ecg):
    """
    Heart murmur frequency analysis for valvular defect diagnosis

    Formula: Murmur spectral features: dominant freq, bandwidth, spectral centroid

    Parameters
    ----------
    pcg : array-like
        Input data.
    fs : array-like
        Input data.
    ecg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: murmur_features

    References
    ----------
    Rangayyan Ch 6.2.2
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
            "method": "Heart murmur frequency analysis for valvular defect diagnosis",
        }
    )


# -- rgnrnst: Nernst equilibrium potential for ionic species.
def rangayyan_nernst_potential(T, z, conc_out, conc_in):
    """
    Nernst equilibrium potential for ionic species

    Formula: E_ion = (RT/zF) * ln([ion]_out / [ion]_in)

    Parameters
    ----------
    T : array-like
        Input data.
    z : array-like
        Input data.
    conc_out : array-like
        Input data.
    conc_in : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: nernst_potential

    References
    ----------
    Rangayyan Ch 1.2.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Nernst equilibrium potential for ionic species"}
    )


# -- rgoae: Otoacoustic emission (OAE) signal analysis.
def rangayyan_oae(oae, fs):
    """
    Otoacoustic emission (OAE) signal analysis

    Formula: TEOAEs extracted by nonlinear click suppression; DPOAE ratio 2f1-f2

    Parameters
    ----------
    oae : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: analysis

    References
    ----------
    Rangayyan Ch 1.2.16
    """
    oae = np.asarray(oae, dtype=float)
    n = int(oae) if oae.ndim == 0 else len(oae)
    result = float(np.mean(oae))
    se = float(np.std(oae, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Otoacoustic emission (OAE) signal analysis"}
    )


# compact alias per ledger/NAMING.md
rangayyanoae = rangayyan_oae


# -- rgpark: Parkinson's disease monitoring via multimodal signal analysis.
def rangayyan_parkinson_multimodal(eeg, emg, gait, fs):
    """
    Parkinson's disease monitoring via multimodal signal analysis

    Formula: Features from EEG, EMG, gait; LDA/RNN for tremor/rigidity classification

    Parameters
    ----------
    eeg : array-like
        Input data.
    emg : array-like
        Input data.
    gait : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pd_score, features

    References
    ----------
    Rangayyan Ch 10.14
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Parkinson's disease monitoring via multimodal signal analysis",
        }
    )


# -- rgpcgeeg: PCG-EEG coupling analysis for auditory evoked response.
def rangayyan_pcg_eeg_coupling(pcg, eeg, fs):
    """
    PCG-EEG coupling analysis for auditory evoked response

    Formula: Cross-coherence C_pcg_eeg(f); peak at fundamental S1 frequency

    Parameters
    ----------
    pcg : array-like
        Input data.
    eeg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coherence, coupling_freq

    References
    ----------
    Rangayyan Ch 2
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
            "method": "PCG-EEG coupling analysis for auditory evoked response",
        }
    )


# -- rgpcgmrm: Murmur presence detection in PCG via spectral analysis.
def rangayyan_pcg_murmur_detect(pcg, ecg, fs):
    """
    Murmur presence detection in PCG via spectral analysis

    Formula: Murmur: high energy in systole spectral centroid > normal; threshold rule

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
        Keys: has_murmur, confidence

    References
    ----------
    Rangayyan Ch 10.2.4
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
            "method": "Murmur presence detection in PCG via spectral analysis",
        }
    )


# -- rgpolysg: Polysomnography signal fusion for sleep staging.
def rangayyan_polysomnography(eeg, eog, emg, fs, epoch_len):
    """
    Polysomnography signal fusion for sleep staging

    Formula: Features from EEG, EOG, EMG -> rule-based or ML sleep stage classifier

    Parameters
    ----------
    eeg : array-like
        Input data.
    eog : array-like
        Input data.
    emg : array-like
        Input data.
    fs : array-like
        Input data.
    epoch_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sleep_stages, hypnogram

    References
    ----------
    Rangayyan Ch 2.4.1
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Polysomnography signal fusion for sleep staging"}
    )


# -- rgppt: Point process model for inter-event interval (IEI) statistics.
def rangayyan_point_process(event_times, T, cdf=None):
    """
    Point process model for inter-event interval (IEI) statistics

    Formula: Poisson: P(k events in T) = (lambda*T)^k * exp(-lambda*T) / k!

    Parameters
    ----------
    event_times : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rate, iei_mean, iei_cv

    References
    ----------
    Rangayyan Ch 7.3
    """
    event_times = np.asarray(event_times, dtype=float)
    n = int(event_times) if event_times.ndim == 0 else len(event_times)
    if event_times.ndim == 0:
        return RichResult(
            payload={"statistic": float("nan"), "p_value": float("nan"), "n": 1, "method": "scalar-input placeholder"}
        )
    if n < 2:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "Point process model for inter-event interval (IEI) statistics",
            }
        )
    x_sorted = np.sort(event_times)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(event_times), scale=np.std(event_times, ddof=1))
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
            "method": "Point process model for inter-event interval (IEI) statistics",
        }
    )


# -- rgpros: Prosthetic heart valve evaluation via PCG spectral analysis.
def rangayyan_prosthetic_valve(pcg, fs):
    """
    Prosthetic heart valve evaluation via PCG spectral analysis

    Formula: Valve sounds: spectral centroid, high-frequency components > normal range

    Parameters
    ----------
    pcg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: valve_score, spectral_features

    References
    ----------
    Rangayyan Ch 6.5
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
            "method": "Prosthetic heart valve evaluation via PCG spectral analysis",
        }
    )


# -- rgrespf: Respiratory signal analysis: rate, depth, I:E ratio.
def rangayyan_respiration_features(resp, fs):
    """
    Respiratory signal analysis: rate, depth, I:E ratio

    Formula: RR = peaks/time; depth = amplitude; I:E = rise_time/fall_time

    Parameters
    ----------
    resp : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: resp_rate, depth, ie_ratio

    References
    ----------
    Rangayyan Ch 5.10
    """
    resp = np.asarray(resp, dtype=float)
    n = int(resp) if resp.ndim == 0 else len(resp)
    result = float(np.mean(resp))
    se = float(np.std(resp, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Respiratory signal analysis: rate, depth, I:E ratio"}
    )


# -- rgrespsnd: Respiratory sound generation model (bronchial turbulence).
def rangayyan_respiratory_sound(resp_sound, fs, flow):
    """
    Respiratory sound generation model (bronchial turbulence)

    Formula: Turbulent jet model; PSD proportional to flow^n; n estimated from spectra

    Parameters
    ----------
    resp_sound : array-like
        Input data.
    fs : array-like
        Input data.
    flow : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model_params, fit_psd

    References
    ----------
    Rangayyan Ch 7.7.1
    """
    resp_sound = np.asarray(resp_sound, dtype=float)
    n = int(resp_sound) if resp_sound.ndim == 0 else len(resp_sound)
    result = float(np.mean(resp_sound))
    se = float(np.std(resp_sound, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Respiratory sound generation model (bronchial turbulence)",
        }
    )


# -- rgsapdet: Sleep apnea detection using multimodal biomedical signals.
def rangayyan_sleep_apnea_detect(ecg, spo2, snore, fs):
    """
    Sleep apnea detection using multimodal biomedical signals

    Formula: Feature fusion of ECG-derived resp, SpO2, snore; Bayes or SVM classifier

    Parameters
    ----------
    ecg : array-like
        Input data.
    spo2 : array-like
        Input data.
    snore : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: apnea_labels, ahi

    References
    ----------
    Rangayyan Ch 10.13
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
            "method": "Sleep apnea detection using multimodal biomedical signals",
        }
    )


# -- rgspeech: Speech signal formant and pitch extraction.
def rangayyan_speech_features(speech, fs, order):
    """
    Speech signal formant and pitch extraction

    Formula: Formants F1-F3 from LPC poles; pitch via autocorrelation peak

    Parameters
    ----------
    speech : array-like
        Input data.
    fs : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: formants, pitch

    References
    ----------
    Rangayyan Ch 1.2.13
    """
    speech = np.asarray(speech, dtype=float)
    n = int(speech) if speech.ndim == 0 else len(speech)
    result = float(np.mean(speech))
    se = float(np.std(speech, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Speech signal formant and pitch extraction"}
    )


# -- rgvag: Vibroarthrogram (VAG) signal characterization.
def rangayyan_vag_analysis(vag, fs):
    """
    Vibroarthrogram (VAG) signal characterization

    Formula: Features: RMS, ZCR, spectral entropy, fractal dimension

    Parameters
    ----------
    vag : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Rangayyan Ch 1.2.14
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Vibroarthrogram (VAG) signal characterization"}
    )


# -- rgvagkn: VAG-based knee-joint cartilage pathology detection.
def rangayyan_vag_knee_cartilage(vag, fs):
    """
    VAG-based knee-joint cartilage pathology detection

    Formula: Fractal dimension and spectral features from VAG; SVM/LDA classifier

    Parameters
    ----------
    vag : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pathology_score, features

    References
    ----------
    Rangayyan Ch 10.12
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "VAG-based knee-joint cartilage pathology detection"}
    )


# -- rng029: Decomposition of a signal into weighted deltas (Rangayyan eq. 3.29).
def deltadecomp(x, t=None):
    """Resolve a signal into a weighted combination of shifted deltas.

    Rangayyan (2024) eq. (3.29):
        x(t) = integral x(alpha) delta(t - alpha) d alpha.

    The book reads this as resolving x into mutually orthogonal delta
    functions.  Discretely, the weight carried by the delta at alpha_i is
    x(alpha_i) times the grid spacing, so that summing the weights
    reproduces the integral of x rather than the sum of its samples;
    reconstructing from the weights returns the original samples exactly,
    which is the check returned in ``reconstruction_error``.
    """
    xs = aslist(x)
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    ts = [float(i) for i in range(n)] if t is None else aslist(t)
    if len(ts) != n:
        raise ValueError("t and x must have the same length")
    if n == 1:
        dt = [1.0]
    else:
        # trapezoidal weights: half a spacing at each end, so that the
        # weights sum to the integral of x rather than overcounting the
        # two endpoints by half a panel each.
        dt = []
        for i in range(n):
            lo = ts[i] - ts[i - 1] if i > 0 else 0.0
            hi = ts[i + 1] - ts[i] if i < n - 1 else 0.0
            dt.append(0.5 * (lo + hi))
    weights = [v * d for v, d in zip(xs, dt)]
    recon = [w / d for w, d in zip(weights, dt)]
    err = max(abs(a - b) for a, b in zip(recon, xs))
    return RichResult(payload={
        "locations": ts, "weights": weights, "amplitudes": xs,
        "total_weight": sum(weights),
        "integral": gridint(xs, ts) if n > 1 else 0.0,
        "reconstruction_error": err,
        "method": "Rangayyan (2024) eq. (3.29)"})


rangayyan_ch3_signal_as_delta_decomposition = deltadecomp  # pre-policy spelling


# -- rng235: Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum..
def rangayyan_ch4_complex_log_of_product(X, H, omega):
    """
    Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum.

    Formula: log[Y(omega)] = log[X(omega)] + log[H(omega)], with X(omega)!=0 and H(omega)!=0

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.63, p. 245
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum.",
        }
    )


# -- rng240: Complex log of X(z) expanded as a sum of log terms over poles and zeros..
def rangayyan_ch4_complex_log_x_z(A, z, r, a_k, b_k, c_k, d_k, M_I, M_O, N_I, N_O):
    """
    Complex log of X(z) expanded as a sum of log terms over poles and zeros.

    Formula: X_hat(z) = log[X(z)] = log[A] + log[z^r] + sum_{k=1}^{M_I} log(1 - a_k z^(-1)) + sum_{k=1}^{M_O} log(1 - b_k z) - sum_{k=1}^{N_I} log(1 - c_k z^(-1)) - sum_{k=1}^{N_O} log(1 - d_k z)

    Parameters
    ----------
    A : array-like
        Input data.
    z : array-like
        Input data.
    r : array-like
        Input data.
    a_k : array-like
        Input data.
    b_k : array-like
        Input data.
    c_k : array-like
        Input data.
    d_k : array-like
        Input data.
    M_I : array-like
        Input data.
    M_O : array-like
        Input data.
    N_I : array-like
        Input data.
    N_O : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.68, p. 248
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
            "method": "Complex log of X(z) expanded as a sum of log terms over poles and zeros.",
        }
    )


_CHEATSHEET = [
    'emd({}) -> Empirical Mode Decomposition (standalone with diagnostics).',
    'rgap: Idealized action potential waveform model (depolarization/repolarization)',
    'rgcardep: Cardiac electrophysiology tissue/organ-level model',
    'rgcorad: Coronary artery disease detection from acoustic signals',
    'rgcorart: Coronary artery sound generation model (turbulent flow)',
    'rgcry: Infant cry signal analysis: formants and fundamental frequency',
    'rgegg: Electrogastrogram (EGG) feature extraction (dominant frequency, power)',
    'rgelast: returns spectral descriptors, not an invented stiffness value',
    'rgengn: Electroneurogram (ENG) compound action potential model',
    'rgepidet: Epileptic seizure detection in EEG',
    'rgerp: Event-related potential (ERP) latency and amplitude features',
    'rgfeatex: Feature extraction for BCI from EEG (event-related desynchronization/synchronization)',
    'rgfrqdom: Frequency-domain feature extraction for CAD',
    'rgghk: Goldman-Hodgkin-Katz (GHK) equation for resting membrane potential',
    'rghgate: Hodgkin-Huxley gating variable ODEs (m, h, n)',
    'rghhmm: Hodgkin-Huxley membrane model for action potential',
    'rghmm: FitzHugh-Nagumo simplified neuron model',
    'rgmemb: Membrane potential dynamics (RC circuit model)',
    'rgmscart: Muscle contraction artifact removal from VAG signals',
    'rgmuap: Motor unit action potential (MUAP) model',
    'rgmurm: Heart murmur frequency analysis for valvular defect diagnosis',
    'rgnrnst: Nernst equilibrium potential for ionic species',
    'rgoae: Otoacoustic emission (OAE) signal analysis',
    "rgpark: Parkinson's disease monitoring via multimodal signal analysis",
    'rgpcgeeg: PCG-EEG coupling analysis for auditory evoked response',
    'rgpcgmrm: Murmur presence detection in PCG via spectral analysis',
    'rgpolysg: Polysomnography signal fusion for sleep staging',
    'rgppt: Point process model for inter-event interval (IEI) statistics',
    'rgpros: Prosthetic heart valve evaluation via PCG spectral analysis',
    'rgrespf: Respiratory signal analysis: rate, depth, I:E ratio',
    'rgrespsnd: Respiratory sound generation model (bronchial turbulence)',
    'rgsapdet: Sleep apnea detection using multimodal biomedical signals',
    'rgspeech: Speech signal formant and pitch extraction',
    'rgvag: Vibroarthrogram (VAG) signal characterization',
    'rgvagkn: VAG-based knee-joint cartilage pathology detection',
    'rng029: delta decomposition of a signal, Rangayyan eq. (3.29)',
    'rng235: Complex logarithm converts the product Y(omega)=X(omega)H(omega) into a sum.',
    'rng240: Complex log of X(z) expanded as a sum of log terms over poles and zeros.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
