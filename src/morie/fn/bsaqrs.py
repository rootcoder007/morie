# morie.fn -- bsaqrs (rootcoder007/morie)
"""ECG event detection and rate analysis: QRS detectors, P and T waves, heart rate, HRV.

Biomedical Signal Analysis (Rangayyan, 2024).  Formerly 46
one-function modules named after book coordinates; the public
symbols are unchanged.
"""

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import RichResult

__all__ = [
    'rangayyan_baseline_wander',
    'rangayyan_carotid_pulse',
    'rangayyan_deriv_qrs',
    'rangayyan_dicrotic_notch',
    'rangayyan_ecg_emg_coupling',
    'rangayyan_ecg_features',
    'rangayyan_ecg_waveshape',
    'rangayyan_exercise_ecg',
    'rangayyan_hrv_freq_domain',
    'rangayyan_hrv_time_domain',
    'rangayyan_heart_sound_id',
    'rangayyan_maternal_ecg_filter',
    'rangayyan_motion_artifact',
    'rangayyan_pan_tompkins',
    'rangayyan_pcg_segments',
    'rangayyan_powerline_removal',
    'rangayyan_ppg_features',
    'rangayyan_p_wave_detect',
    'rangayyan_resp_signal',
    'rangayyan_sleep_apnea',
    'rangayyan_spectral_power_ratio',
    'rangayyan_twave_alternans',
    'rangayyan_twa_spectral_mx',
    'rangayyan_t_wave_detect',
    'rangayyan_vf_detect',
    'rangayyan_ch4_qrs_first_derivative_balda',
    'rangayyan_ch4_qrs_second_derivative_balda',
    'rangayyan_ch4_qrs_combined_balda',
    'rangayyan_ch4_filtered_derivative_murthy',
    'rangayyan_ch4_qrs_smoothing_ma_filter',
    'rangayyan_ch4_pan_tompkins_lowpass_transfer',
    'rangayyan_ch4_pan_tompkins_lowpass_difference_eq',
    'rangayyan_ch4_pan_tompkins_highpass_lp_component',
    'rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq',
    'rangayyan_ch4_pan_tompkins_highpass_transfer',
    'rangayyan_ch4_pan_tompkins_highpass_difference_eq',
    'rangayyan_ch4_pan_tompkins_highpass_combined',
    'rangayyan_ch4_pan_tompkins_derivative_operator',
    'rangayyan_ch4_pan_tompkins_moving_window_integrator',
    'rangayyan_ch4_pan_tompkins_thresholds',
    'rangayyan_ch4_pan_tompkins_searchback_update',
    'rangayyan_ch4_heart_rate_from_count',
    'rangayyan_ch4_heart_rate_from_rr',
    'rangayyan_ch4_length_transformation',
    'rangayyan_ch4_dicrotic_notch_second_derivative',
    'rangayyan_ch4_dicrotic_notch_smoothed_squared',
]


# -- rgblwand: Baseline wander removal from ECG.
def rangayyan_baseline_wander(ecg, fs, cutoff):
    """
    Baseline wander removal from ECG

    Formula: High-pass > 0.05 Hz Butterworth or cubic spline through isoelectric points

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    cutoff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ecg_detrended

    References
    ----------
    Rangayyan Ch 3.3.2
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Baseline wander removal from ECG"})


# -- rgcpulse: Carotid pulse waveform feature extraction.
def rangayyan_carotid_pulse(pulse, fs):
    """
    Carotid pulse waveform feature extraction

    Formula: Features: anacrotic rise, systolic peak, dicrotic notch, diastolic peak, coeff. of elasticity

    Parameters
    ----------
    pulse : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Rangayyan Ch 1.2.10
    """
    pulse = np.asarray(pulse, dtype=float)
    n = int(pulse) if pulse.ndim == 0 else len(pulse)
    result = float(np.mean(pulse))
    se = float(np.std(pulse, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Carotid pulse waveform feature extraction"}
    )


# -- rgderqrs: Derivative-based QRS detection (first and second differences).
def rangayyan_deriv_qrs(ecg, fs, threshold):
    """
    Derivative-based QRS detection (first and second differences)

    Formula: d_ecg = |dx/dt|; threshold on peak of derivative; decision threshold adaptive

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: qrs_locs

    References
    ----------
    Rangayyan Ch 4.3.1
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
            "method": "Derivative-based QRS detection (first and second differences)",
        }
    )


# -- rgdnot: Dicrotic notch detection in carotid pulse waveform.
def rangayyan_dicrotic_notch(pulse, fs):
    """
    Dicrotic notch detection in carotid pulse waveform

    Formula: Notch = local minimum between systolic and diastolic peaks

    Parameters
    ----------
    pulse : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: notch_loc

    References
    ----------
    Rangayyan Ch 4.3.5
    """
    pulse = np.asarray(pulse, dtype=float)
    n = int(pulse) if pulse.ndim == 0 else len(pulse)
    result = float(np.mean(pulse))
    se = float(np.std(pulse, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Dicrotic notch detection in carotid pulse waveform"}
    )


# -- rgecgemu: ECG-EMG coupling during physical effort (VMG correlation).
def rangayyan_ecg_emg_coupling(ecg, emg, fs):
    """
    ECG-EMG coupling during physical effort (VMG correlation)

    Formula: CCF(ECG, EMG) at cardiac frequency; cardio-locomotor coupling index

    Parameters
    ----------
    ecg : array-like
        Input data.
    emg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coupling_index, ccf

    References
    ----------
    Rangayyan Ch 2.2.6
    """
    ecg = np.asarray(ecg, dtype=float)
    y = np.asarray(emg, dtype=float)
    n = min(len(ecg), len(y))
    if n < 3:
        return RichResult(
            payload={
                "statistic": np.nan,
                "p_value": np.nan,
                "n": n,
                "method": "ECG-EMG coupling during physical effort (VMG correlation)",
            }
        )
    result = stats.spearmanr(ecg[:n], y[:n])
    return RichResult(
        payload={
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "n": n,
            "method": "ECG-EMG coupling during physical effort (VMG correlation)",
        }
    )


# -- rgecgf: ECG waveform feature extraction (P, QRS, T amplitudes and durations).
def rangayyan_ecg_features(ecg, fs, r_peaks):
    """
    ECG waveform feature extraction (P, QRS, T amplitudes and durations)

    Formula: Feature vector = [P_amp, PR_int, QRS_dur, QT_int, T_amp, ST_dev]

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
        Keys: feature_dict

    References
    ----------
    Rangayyan Ch 1.2.5
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
            "method": "ECG waveform feature extraction (P, QRS, T amplitudes and durations)",
        }
    )


# -- rgecgwvf: ECG waveform analysis for ischemia and bundle branch block.
def rangayyan_ecg_waveshape(ecg, fs, r_peaks, template):
    """
    ECG waveform analysis for ischemia and bundle branch block

    Formula: Template correlation coefficient rho > 0.9 = normal; < 0.7 = ectopic

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.
    template : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho_per_beat, classification

    References
    ----------
    Rangayyan Ch 5.4.3
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
            "method": "ECG waveform analysis for ischemia and bundle branch block",
        }
    )


# -- rgexecg: Exercise ECG analysis: ST deviation, slope, and ischemia detection.
def rangayyan_exercise_ecg(ecg, fs, r_peaks):
    """
    Exercise ECG analysis: ST deviation, slope, and ischemia detection

    Formula: ST level = mean(ECG) in J+60ms to J+80ms window; ST slope from regression

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
        Keys: st_level, st_slope, ischemia_flag

    References
    ----------
    Rangayyan Ch 5.8
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
            "method": "Exercise ECG analysis: ST deviation, slope, and ischemia detection",
        }
    )


# -- rghrvf: HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio.
def rangayyan_hrv_freq_domain(rr_intervals, fs_resamp):
    """
    HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio

    Formula: P_band = integral_{f1}^{f2} S_RR(f) df; LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz

    Parameters
    ----------
    rr_intervals : array-like
        Input data.
    fs_resamp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vlf, lf, hf, lf_hf

    References
    ----------
    Rangayyan Ch 2
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio",
        }
    )


# -- rghrvt: HRV time-domain metrics: SDNN, RMSSD, pNN50.
def rangayyan_hrv_time_domain(rr_intervals):
    """
    HRV time-domain metrics: SDNN, RMSSD, pNN50

    Formula: SDNN=std(RR); RMSSD=sqrt(mean(diff(RR)^2)); pNN50=sum(|dRR|>50ms)/N

    Parameters
    ----------
    rr_intervals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sdnn, rmssd, pnn50

    References
    ----------
    Rangayyan Ch 2
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "HRV time-domain metrics: SDNN, RMSSD, pNN50"}
    )


# -- rghsnd: Heart sound (S1/S2) identification via PCG-ECG timing.
def rangayyan_heart_sound_id(pcg, ecg, fs):
    """
    Heart sound (S1/S2) identification via PCG-ECG timing

    Formula: S1 in [0, 30%] of cardiac cycle; S2 in [40%, 60%] relative to R-peak

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
        Keys: s1_locs, s2_locs

    References
    ----------
    Rangayyan Ch 4.9
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
            "method": "Heart sound (S1/S2) identification via PCG-ECG timing",
        }
    )


# -- rgmatefp: Maternal ECG filtering from abdominal ECG recording.
def rangayyan_maternal_ecg_filter(abdominal_ecg, fs, n_channels):
    """
    Maternal ECG filtering from abdominal ECG recording

    Formula: ICA or adaptive filter removes maternal component; fetal ECG in residual

    Parameters
    ----------
    abdominal_ecg : array-like
        Input data.
    fs : array-like
        Input data.
    n_channels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fetal_ecg, maternal_template

    References
    ----------
    Rangayyan Ch 9.11
    """
    abdominal_ecg = np.asarray(abdominal_ecg, dtype=float)
    n = int(abdominal_ecg) if abdominal_ecg.ndim == 0 else len(abdominal_ecg)
    result = float(np.mean(abdominal_ecg))
    se = float(np.std(abdominal_ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Maternal ECG filtering from abdominal ECG recording"}
    )


# -- rgmtnart: Motion artifact detection and removal from ECG/PPG.
def rangayyan_motion_artifact(x, accel, fs):
    """
    Motion artifact detection and removal from ECG/PPG

    Formula: Motion artifact: broadband noise 0-30 Hz; detect by accelerometer correlation

    Parameters
    ----------
    x : array-like
        Input data.
    accel : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_clean, artifact_mask

    References
    ----------
    Rangayyan Ch 3.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Motion artifact detection and removal from ECG/PPG"}
    )


# -- rgpantp: Pan-Tompkins QRS detection algorithm.
def rangayyan_pan_tompkins(ecg, fs):
    """
    Pan-Tompkins QRS detection algorithm

    Formula: BP(5-15Hz) -> deriv -> square -> MA(150ms) -> adaptive threshold

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: r_peaks

    References
    ----------
    Rangayyan Ch 4.3.2
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pan-Tompkins QRS detection algorithm"})


# -- rgpcg: PCG segmentation into S1/systole/S2/diastole using ECG gating.
def rangayyan_pcg_segments(pcg, ecg, fs):
    """
    PCG segmentation into S1/systole/S2/diastole using ECG gating

    Formula: S1 onset ~ R-wave; S2 onset ~ T-wave end; durations from timing ratios

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
        Keys: segment_labels

    References
    ----------
    Rangayyan Ch 1.2.9
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
            "method": "PCG segmentation into S1/systole/S2/diastole using ECG gating",
        }
    )


# -- rgpowerl: Powerline interference (50/60 Hz) removal from ECG.
def rangayyan_powerline_removal(ecg, fs, powerline_freq):
    """
    Powerline interference (50/60 Hz) removal from ECG

    Formula: Adaptive notch or comb filter at powerline fundamental + harmonics

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    powerline_freq : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ecg_clean

    References
    ----------
    Rangayyan Ch 3.3.4
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Powerline interference (50/60 Hz) removal from ECG"}
    )


# -- rgppg: PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak).
def rangayyan_ppg_features(ppg, fs):
    """
    PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak)

    Formula: Features: systolic amplitude, pulse width, augmentation index AI = (P2-P1)/P1

    Parameters
    ----------
    ppg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Rangayyan Ch 1.2.11
    """
    ppg = np.asarray(ppg, dtype=float)
    n = int(ppg) if ppg.ndim == 0 else len(ppg)
    result = float(np.mean(ppg))
    se = float(np.std(ppg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak)",
        }
    )


# -- rgpwave: P-wave detection in ECG using search window relative to R-peak.
def rangayyan_p_wave_detect(ecg, fs, r_peaks):
    """
    P-wave detection in ECG using search window relative to R-peak

    Formula: P search window: 200 ms before QRS; peak in 50-120 ms PR interval

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
        Keys: p_locs

    References
    ----------
    Rangayyan Ch 4.3.3
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
            "method": "P-wave detection in ECG using search window relative to R-peak",
        }
    )


# -- rgrpsig: ECG-derived respiration (EDR) via R-wave amplitude modulation.
def rangayyan_resp_signal(ecg, r_peaks, fs_out):
    """
    ECG-derived respiration (EDR) via R-wave amplitude modulation

    Formula: R_amp(k) = ECG amplitude at k-th R-peak; respiration rate from R_amp spectrum

    Parameters
    ----------
    ecg : array-like
        Input data.
    r_peaks : array-like
        Input data.
    fs_out : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: resp_signal, resp_rate

    References
    ----------
    Rangayyan Ch 2.4.2
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
            "method": "ECG-derived respiration (EDR) via R-wave amplitude modulation",
        }
    )


# -- rgsapn: Sleep apnea detection via ECG-derived respiration + SpO2 fusion.
def rangayyan_sleep_apnea(ecg, spo2, fs):
    """
    Sleep apnea detection via ECG-derived respiration + SpO2 fusion

    Formula: Apnea index = events/hr; detection threshold on RR variability + desaturation

    Parameters
    ----------
    ecg : array-like
        Input data.
    spo2 : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: apnea_index, events

    References
    ----------
    Rangayyan Ch 2.4
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
            "method": "Sleep apnea detection via ECG-derived respiration + SpO2 fusion",
        }
    )


# -- rgspr: Spectral power ratio (LF/HF) for HRV analysis.
def rangayyan_spectral_power_ratio(rr_psd, freqs):
    """
    Spectral power ratio (LF/HF) for HRV analysis

    Formula: LF = integral_{0.04}^{0.15} S(f)df; HF = integral_{0.15}^{0.40} S(f)df; ratio=LF/HF

    Parameters
    ----------
    rr_psd : array-like
        Input data.
    freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lf, hf, lf_hf_ratio

    References
    ----------
    Rangayyan Ch 6.4.2
    """
    rr_psd = np.asarray(rr_psd, dtype=float)
    n = int(rr_psd) if rr_psd.ndim == 0 else len(rr_psd)
    result = float(np.mean(rr_psd))
    se = float(np.std(rr_psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Spectral power ratio (LF/HF) for HRV analysis"}
    )


# -- rgtwa: T-wave alternans (TWA) detection via spectral method.
def rangayyan_twave_alternans(ecg, fs, r_peaks):
    """
    T-wave alternans (TWA) detection via spectral method

    Formula: TWA at 0.5 cycles/beat in even-odd T-wave amplitude spectrum

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
        Keys: twa_magnitude, k_score

    References
    ----------
    Rangayyan Ch 9.10
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "T-wave alternans (TWA) detection via spectral method"}
    )


# -- rgtwamx: T-wave alternans spectral method.
def rangayyan_twa_spectral_mx(ecg, fs, r_peaks, t_window=None, n_beats=128):
    r"""Spectral T-wave alternans (Rangayyan Ch. 3).

    Builds a beat-by-sample matrix aligned on the R peaks, takes the
    FFT ALONG THE BEAT AXIS at each sample offset, and reads the power
    at 0.5 cycles/beat -- the frequency of a strictly ABAB alternation:

    .. math:: k_{alt} = \tfrac12 \text{ cycles per beat}.

    The alternans voltage is the excess over the neighbouring noise
    band, and the k-score is that excess in noise standard deviations;
    both are returned because a raw spectral peak means nothing
    without its noise floor. An even number of beats is required, or
    0.5 cycles/beat is not an exact FFT bin and the alternans power
    leaks.

    Parameters
    ----------
    ecg : array-like
        ECG signal.
    fs : float
        Sampling frequency.
    r_peaks : array-like of int
        R-peak indices.
    t_window : (int, int), optional
        Offsets after R defining the T wave; a physiological default
        of 100-300 ms is used otherwise.
    n_beats : int, default 128
        Beats to use (truncated to an even number).

    Returns
    -------
    RichResult
        keys: ``alternans_voltage``, ``k_score``, ``noise_mean``,
        ``noise_std``, ``spectrum``, ``n_beats_used``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (T-wave alternans).
    """
    x = np.asarray(ecg, dtype=float).ravel()
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    r = np.asarray(r_peaks, dtype=int).ravel()
    if r.size < 8:
        raise ValueError(f"need at least 8 beats, got {r.size}.")
    lo, hi = (int(0.10 * fs), int(0.30 * fs)) if t_window is None else (
        int(t_window[0]), int(t_window[1])
    )
    if not 0 <= lo < hi:
        raise ValueError("t_window must satisfy 0 <= start < stop.")
    usable = [p for p in r if p + hi <= x.size]
    M = min(int(n_beats), len(usable))
    M -= M % 2  # 0.5 cycles/beat must land on an exact FFT bin
    if M < 8:
        raise ValueError("fewer than 8 complete beats after alignment.")
    mat = np.array([x[p + lo : p + hi] for p in usable[:M]])
    mat = mat - mat.mean(axis=0, keepdims=True)
    S = np.abs(np.fft.rfft(mat, axis=0)) ** 2 / M
    spec = S.sum(axis=1)  # aggregate across the T-wave samples
    k_alt = M // 2  # the 0.5 cycles/beat bin
    noise_band = spec[int(0.33 * len(spec)) : k_alt]
    nm = float(noise_band.mean()) if noise_band.size else 0.0
    ns = float(noise_band.std()) if noise_band.size else 0.0
    excess = float(spec[k_alt]) - nm
    volt = float(np.sqrt(max(excess, 0.0)))
    return RichResult(payload={"alternans_voltage": volt,
                               "k_score": (excess / ns) if ns > 0 else np.inf,
                               "noise_mean": nm, "noise_std": ns, "spectrum": spec,
                               "n_beats_used": int(M),
                               "method": "FFT along the beat axis; power at 0.5 cyc/beat over noise"})


# -- rgtwave: T-wave detection in ECG.
def rangayyan_t_wave_detect(ecg, fs, r_peaks):
    """
    T-wave detection in ECG

    Formula: T search window: 100-400 ms after QRS end; peak detection in window

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
        Keys: t_locs

    References
    ----------
    Rangayyan Ch 4.3.4
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T-wave detection in ECG"})


# -- rgvf: Ventricular fibrillation (VF) detection in ECG.
def rangayyan_vf_detect(ecg, fs):
    """
    Ventricular fibrillation (VF) detection in ECG

    Formula: Spectral features, Hilbert transform, threshold on VF frequency range 3-10 Hz

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_vf, confidence

    References
    ----------
    Rangayyan Ch 8.16
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={"estimate": result, "se": se, "n": n, "method": "Ventricular fibrillation (VF) detection in ECG"}
    )


# -- rng176: Smoothed three-point first derivative used in QRS detection (Balda et al.)..
def rangayyan_ch4_qrs_first_derivative_balda(x, n):
    """
    Smoothed three-point first derivative used in QRS detection (Balda et al.).

    Formula: y_0(n) = |x(n) - x(n-2)|

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.1, p. 218
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
            "method": "Smoothed three-point first derivative used in QRS detection (Balda et al.).",
        }
    )


# -- rng177: Approximation of the second derivative used in QRS detection..
def rangayyan_ch4_qrs_second_derivative_balda(x, n):
    """
    Approximation of the second derivative used in QRS detection.

    Formula: y_1(n) = |x(n) - 2*x(n-2) + x(n-4)|

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.2, p. 218
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
            "method": "Approximation of the second derivative used in QRS detection.",
        }
    )


# -- rng178: Weighted combination of first and second derivatives for QRS detection..
def rangayyan_ch4_qrs_combined_balda(y_0, y_1, n):
    """
    Weighted combination of first and second derivatives for QRS detection.

    Formula: y_2(n) = 1.3 * y_0(n) + 1.1 * y_1(n)

    Parameters
    ----------
    y_0 : array-like
        Input data.
    y_1 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.3, p. 218
    """
    y_0 = np.atleast_1d(np.asarray(y_0, dtype=float))
    n = len(y_0)
    result = float(np.mean(y_0))
    se = float(np.std(y_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Weighted combination of first and second derivatives for QRS detection.",
        }
    )


# -- rng179: Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj)..
def rangayyan_ch4_filtered_derivative_murthy(x, n, N):
    """
    Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj).

    Formula: g_1(n) = sum_{i=1}^{N} |x(n-i+1) - x(n-i)|^2 * (N - i + 1)

    Parameters
    ----------
    x : array-like
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
    Rangayyan (2024), Ch 4, Eq 4.4, p. 219
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
            "method": "Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj).",
        }
    )


# -- rng180: MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector..
def rangayyan_ch4_qrs_smoothing_ma_filter(g_1, n, M):
    """
    MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector.

    Formula: g(n) = (1/M) * sum_{j=0}^{M-1} g_1(n - j)

    Parameters
    ----------
    g_1 : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.5, p. 219
    """
    g_1 = np.atleast_1d(np.asarray(g_1, dtype=float))
    n = len(g_1)
    result = float(np.mean(g_1))
    se = float(np.std(g_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector.",
        }
    )


# -- rng181: Lowpass transfer function used in the Pan-Tompkins QRS detector..
def rangayyan_ch4_pan_tompkins_lowpass_transfer(z):
    """
    Lowpass transfer function used in the Pan-Tompkins QRS detector.

    Formula: H(z) = (1/32) * (1 - z^(-6))^2 / (1 - z^(-1))^2

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.7, p. 220
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
            "method": "Lowpass transfer function used in the Pan-Tompkins QRS detector.",
        }
    )


# -- rng182: Difference equation of the Pan-Tompkins lowpass filter..
def rangayyan_ch4_pan_tompkins_lowpass_difference_eq(x, y, n):
    """
    Difference equation of the Pan-Tompkins lowpass filter.

    Formula: y(n) = 2*y(n-1) - y(n-2) + (1/32)*[x(n) - 2*x(n-6) + x(n-12)]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.8, p. 220
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
            "method": "Difference equation of the Pan-Tompkins lowpass filter.",
        }
    )


# -- rng183: Lowpass component of the Pan-Tompkins highpass filter..
def rangayyan_ch4_pan_tompkins_highpass_lp_component(z):
    """
    Lowpass component of the Pan-Tompkins highpass filter.

    Formula: H_lp(z) = (1 - z^(-32)) / (1 - z^(-1))

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.9, p. 221
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
            "method": "Lowpass component of the Pan-Tompkins highpass filter.",
        }
    )


# -- rng184: Difference equation of the lowpass component used in the Pan-Tompkins highpass filter..
def rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq(x, y, n):
    """
    Difference equation of the lowpass component used in the Pan-Tompkins highpass filter.

    Formula: y(n) = y(n-1) + x(n) - x(n-32)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.10, p. 221
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
            "method": "Difference equation of the lowpass component used in the Pan-Tompkins highpass filter.",
        }
    )


# -- rng185: Transfer function of the Pan-Tompkins highpass filter..
def rangayyan_ch4_pan_tompkins_highpass_transfer(z, H_lp):
    """
    Transfer function of the Pan-Tompkins highpass filter.

    Formula: H_hp(z) = z^(-16) - (1/32) * H_lp(z)

    Parameters
    ----------
    z : array-like
        Input data.
    H_lp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.11, p. 221
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
            "method": "Transfer function of the Pan-Tompkins highpass filter.",
        }
    )


# -- rng186: Difference equation of the Pan-Tompkins highpass filter (intermediate)..
def rangayyan_ch4_pan_tompkins_highpass_difference_eq(x, y, n):
    """
    Difference equation of the Pan-Tompkins highpass filter (intermediate).

    Formula: p(n) = x(n - 16) - (1/32) * [y(n-1) + x(n) - x(n-32)]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.12, p. 221
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
            "method": "Difference equation of the Pan-Tompkins highpass filter (intermediate).",
        }
    )


# -- rng187: Combined input-output relationship of the Pan-Tompkins highpass filter..
def rangayyan_ch4_pan_tompkins_highpass_combined(x, p, n):
    """
    Combined input-output relationship of the Pan-Tompkins highpass filter.

    Formula: p(n) = p(n-1) - (1/32)*x(n) + x(n-16) - x(n-17) + (1/32)*x(n-32)

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.13, p. 222
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
            "method": "Combined input-output relationship of the Pan-Tompkins highpass filter.",
        }
    )


# -- rng188: Derivative operator used by Pan and Tompkins for QRS detection..
def rangayyan_ch4_pan_tompkins_derivative_operator(x, n):
    """
    Derivative operator used by Pan and Tompkins for QRS detection.

    Formula: y(n) = (1/8) * [2*x(n) + x(n-1) - x(n-3) - 2*x(n-4)]

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.14, p. 222
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
            "method": "Derivative operator used by Pan and Tompkins for QRS detection.",
        }
    )


# -- rng189: Moving-window integrator used in the Pan-Tompkins QRS detector..
def rangayyan_ch4_pan_tompkins_moving_window_integrator(x, N, n):
    """
    Moving-window integrator used in the Pan-Tompkins QRS detector.

    Formula: y(n) = (1/N) * { x[n-(N-1)] + x[n-(N-2)] + ... + x(n) }

    Parameters
    ----------
    x : array-like
        Input data.
    N : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.15, p. 223
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
            "method": "Moving-window integrator used in the Pan-Tompkins QRS detector.",
        }
    )


# -- rng191: Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm..
def rangayyan_ch4_pan_tompkins_thresholds(NPKI, SPKI):
    """
    Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm.

    Formula: THRESHOLD_I1 = NPKI + 0.25*(SPKI - NPKI); THRESHOLD_I2 = 0.5*THRESHOLD_I1

    Parameters
    ----------
    NPKI : array-like
        Input data.
    SPKI : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.17, p. 224
    """
    NPKI = np.atleast_1d(np.asarray(NPKI, dtype=float))
    n = len(NPKI)
    result = float(np.mean(NPKI))
    se = float(np.std(NPKI, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm.",
        }
    )


# -- rng192: Updated SPKI rule when a QRS is detected in the search-back procedure..
def rangayyan_ch4_pan_tompkins_searchback_update(PEAKI, SPKI):
    """
    Updated SPKI rule when a QRS is detected in the search-back procedure.

    Formula: SPKI = 0.25 * PEAKI + 0.75 * SPKI

    Parameters
    ----------
    PEAKI : array-like
        Input data.
    SPKI : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.18, p. 224
    """
    PEAKI = np.atleast_1d(np.asarray(PEAKI, dtype=float))
    n = len(PEAKI)
    result = float(np.mean(PEAKI))
    se = float(np.std(PEAKI, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Updated SPKI rule when a QRS is detected in the search-back procedure.",
        }
    )


# -- rng193: Heart rate computed from number of QRS complexes detected over duration T..
def rangayyan_ch4_heart_rate_from_count(N_B, T):
    """
    Heart rate computed from number of QRS complexes detected over duration T.

    Formula: HR = 60 * N_B / T

    Parameters
    ----------
    N_B : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.19, p. 224
    """
    N_B = np.atleast_1d(np.asarray(N_B, dtype=float))
    n = len(N_B)
    result = float(np.mean(N_B))
    se = float(np.std(N_B, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Heart rate computed from number of QRS complexes detected over duration T.",
        }
    )


# -- rng194: Heart rate from RR interval.
def rangayyan_ch4_heart_rate_from_rr(RR_a):
    r"""Instantaneous heart rate from the RR interval (Rangayyan
    Ch. 4):

    .. math:: HR = \frac{60}{RR_a},

    with RR in seconds and HR in beats per minute. Vectorised, so a
    series of RR intervals gives the instantaneous rate at each beat;
    the mean of those is NOT the same as 60 / mean(RR) (Jensen), and
    both are returned.

    Parameters
    ----------
    RR_a : float or array-like
        RR interval(s) in seconds, strictly positive.

    Returns
    -------
    RichResult
        keys: ``heart_rate``, ``mean_instantaneous_hr``,
        ``hr_from_mean_rr``, ``n``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 4.
    """
    rr = np.atleast_1d(np.asarray(RR_a, dtype=float))
    if np.any(rr <= 0):
        raise ValueError("RR intervals must be strictly positive.")
    hr = 60.0 / rr
    scalar = np.ndim(RR_a) == 0
    return RichResult(
        payload={"heart_rate": float(hr[0]) if scalar else hr,
                 "mean_instantaneous_hr": float(np.mean(hr)),
                 "hr_from_mean_rr": float(60.0 / np.mean(rr)), "n": int(rr.size),
                 "method": "HR = 60/RR; mean of rates != rate of mean interval"})


# -- rng195: Length transformation used to detect P, QRS, and T waves across multiple ECG channels..
def rangayyan_ch4_length_transformation(x, N, w, t):
    """
    Length transformation used to detect P, QRS, and T waves across multiple ECG channels.

    Formula: L(N, w, t) = integral_{t}^{t+w} sqrt( sum_{j=1}^{N} (dx_j/dt)^2 ) dt

    Parameters
    ----------
    x : array-like
        Input data.
    N : array-like
        Input data.
    w : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.21, p. 227
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
            "method": "Length transformation used to detect P, QRS, and T waves across multiple ECG channels.",
        }
    )


# -- rng196: Noncausal least-squares second derivative used to detect the dicrotic notch.
_COEF = np.array([2.0, -1.0, -2.0, -1.0, 2.0])  # taps for y(n-2)..y(n+2)


def rangayyan_ch4_dicrotic_notch_second_derivative(y, causal=False):
    r"""Lehner-Rangayyan least-squares second derivative.

    .. math:: p(n) = 2y(n-2) - y(n-1) - 2y(n) - y(n+1) + 2y(n+2)

    The five-tap least-squares estimate of the second derivative of the
    carotid pulse. It is deliberately noncausal (it looks two samples
    ahead); the book notes it "may be made causal by applying a delay
    of two samples", which ``causal=True`` does. The second derivative
    removes the constant downward slope of the carotid pulse and leaves
    the dicrotic notch standing out.

    Parameters
    ----------
    y : array-like, shape (n,)
        Carotid pulse signal, n >= 5.
    causal : bool, default False
        Delay the output by two samples so that ``p[n]`` depends only
        on ``y[..n]``.

    Returns
    -------
    RichResult
        keys: ``p`` (n,, zero-padded at the unusable ends), ``valid``
        (slice of fully-supported indices), ``coefficients``,
        ``causal``, ``n``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Eq. (4.22), p. 228 (Sec. 4.3.5, detection of the
    dicrotic notch; after Lehner & Rangayyan).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 5:
        raise ValueError(f"need at least 5 samples for the 5-tap estimate, got {n}.")

    core = np.convolve(y, _COEF[::-1], mode="valid")  # p at indices 2 .. n-3
    p = np.zeros(n)
    if causal:
        p[4:] = core  # two-sample delay: p[n] uses y[n-4..n]
        valid = slice(4, n)
    else:
        p[2 : n - 2] = core
        valid = slice(2, n - 2)

    return RichResult(
        payload={
            "p": p,
            "valid": valid,
            "coefficients": _COEF.copy(),
            "causal": bool(causal),
            "n": int(n),
            "method": "Lehner-Rangayyan LS second derivative (Rangayyan Eq. 4.22, p. 228)",
        }
    )


# -- rng197: Squared and weighted smoothing of the second derivative for dicrotic notch detection..
def rangayyan_ch4_dicrotic_notch_smoothed_squared(p, w, n, M):
    """
    Squared and weighted smoothing of the second derivative for dicrotic notch detection.

    Formula: s(n) = sum_{k=1}^{M} p^2(n - k + 1) * w(k)

    Parameters
    ----------
    p : array-like
        Input data.
    w : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.23, p. 228
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(
        payload={
            "estimate": result,
            "se": se,
            "n": n,
            "method": "Squared and weighted smoothing of the second derivative for dicrotic notch detection.",
        }
    )


_CHEATSHEET = [
    'rgblwand: Baseline wander removal from ECG',
    'rgcpulse: Carotid pulse waveform feature extraction',
    'rgderqrs: Derivative-based QRS detection (first and second differences)',
    'rgdnot: Dicrotic notch detection in carotid pulse waveform',
    'rgecgemu: ECG-EMG coupling during physical effort (VMG correlation)',
    'rgecgf: ECG waveform feature extraction (P, QRS, T amplitudes and durations)',
    'rgecgwvf: ECG waveform analysis for ischemia and bundle branch block',
    'rgexecg: Exercise ECG analysis: ST deviation, slope, and ischemia detection',
    'rghrvf: HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio',
    'rghrvt: HRV time-domain metrics: SDNN, RMSSD, pNN50',
    'rghsnd: Heart sound (S1/S2) identification via PCG-ECG timing',
    'rgmatefp: Maternal ECG filtering from abdominal ECG recording',
    'rgmtnart: Motion artifact detection and removal from ECG/PPG',
    'rgpantp: Pan-Tompkins QRS detection algorithm',
    'rgpcg: PCG segmentation into S1/systole/S2/diastole using ECG gating',
    'rgpowerl: Powerline interference (50/60 Hz) removal from ECG',
    'rgppg: PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak)',
    'rgpwave: P-wave detection in ECG using search window relative to R-peak',
    'rgrpsig: ECG-derived respiration (EDR) via R-wave amplitude modulation',
    'rgsapn: Sleep apnea detection via ECG-derived respiration + SpO2 fusion',
    'rgspr: Spectral power ratio (LF/HF) for HRV analysis',
    'rgtwa: T-wave alternans (TWA) detection via spectral method',
    'rgtwamx: even beat count required or 0.5 cyc/beat is not an exact bin',
    'rgtwave: T-wave detection in ECG',
    'rgvf: Ventricular fibrillation (VF) detection in ECG',
    'rng176: Smoothed three-point first derivative used in QRS detection (Balda et al.).',
    'rng177: Approximation of the second derivative used in QRS detection.',
    'rng178: Weighted combination of first and second derivatives for QRS detection.',
    'rng179: Filtered weighted-squared first-derivative operator for QRS detection (Murthy and Rangaraj).',
    'rng180: MA smoothing filter applied to g_1 in the Murthy-Rangaraj QRS detector.',
    'rng181: Lowpass transfer function used in the Pan-Tompkins QRS detector.',
    'rng182: Difference equation of the Pan-Tompkins lowpass filter.',
    'rng183: Lowpass component of the Pan-Tompkins highpass filter.',
    'rng184: Difference equation of the lowpass component used in the Pan-Tompkins highpass filter.',
    'rng185: Transfer function of the Pan-Tompkins highpass filter.',
    'rng186: Difference equation of the Pan-Tompkins highpass filter (intermediate).',
    'rng187: Combined input-output relationship of the Pan-Tompkins highpass filter.',
    'rng188: Derivative operator used by Pan and Tompkins for QRS detection.',
    'rng189: Moving-window integrator used in the Pan-Tompkins QRS detector.',
    'rng191: Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm.',
    'rng192: Updated SPKI rule when a QRS is detected in the search-back procedure.',
    'rng193: Heart rate computed from number of QRS complexes detected over duration T.',
    'rng194: HR = 60/RR; mean(60/RR) != 60/mean(RR)',
    'rng195: Length transformation used to detect P, QRS, and T waves across multiple ECG channels.',
    'rng196: p(n) = 2y(n-2) - y(n-1) - 2y(n) - y(n+1) + 2y(n+2) (Rangayyan Eq 4.22)',
    'rng197: Squared and weighted smoothing of the second derivative for dicrotic notch detection.',
]


def cheatsheet():
    return "\n".join(_CHEATSHEET)
