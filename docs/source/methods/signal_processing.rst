Signal Processing & Biomedical Analysis
========================================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE provides 25 biomedical signal processing functions via ``morie.signal``
and individual ``morie.fn.*`` modules. All functions are dataset-agnostic:
numpy arrays in, result objects out.

Digital Filters
---------------

Butterworth zero-phase filters (via ``scipy.signal``):

.. code-block:: python

   import math
   from morie.signal import buttlp, butthp, buttbp, buttbs, sgolay

   # a 5 Hz wave with 60 Hz mains hum, sampled at 256 Hz (use your own ECG samples here)
   signal = [math.sin(2 * math.pi * 5 * i / 256) + 0.3 * math.sin(2 * math.pi * 60 * i / 256) for i in range(512)]
   result = buttlp(signal, fs=256, cutoff=40, order=4)
   filtered = result.filtered

- ``buttlp`` -- Lowpass
- ``butthp`` -- Highpass
- ``buttbp`` -- Bandpass
- ``buttbs`` -- Bandstop (notch, default 59-61 Hz for mains hum)
- ``sgolay`` -- Savitzky-Golay polynomial smoothing

Spectral Analysis
-----------------

.. code-block:: python

   import math
   from morie.signal import welch, pburg

   signal = [math.sin(2 * math.pi * 5 * i / 256) for i in range(512)]
   psd = welch(signal, fs=256)                # Welch PSD
   ar_psd = pburg(signal, fs=256, order=16)   # Burg AR PSD (parametric)

Fractal Complexity
------------------

Pure-numpy implementations for nonlinear time-series characterization:

.. code-block:: python

   import math
   from morie.signal import hfd, kfd, pfd, dfa, sampen, hurst

   signal = [math.sin(2 * math.pi * 5 * i / 256) for i in range(512)]
   result = hfd(signal, kmax=10)    # Higuchi fractal dimension
   alpha = dfa(signal).value        # DFA scaling exponent

- ``hfd`` -- Higuchi fractal dimension (Higuchi, 1988)
- ``kfd`` -- Katz fractal dimension
- ``pfd`` -- Petrosian fractal dimension
- ``dfa`` -- Detrended fluctuation analysis
- ``sampen`` -- Sample entropy
- ``hurst`` -- Hurst exponent (R/S analysis)

Typical values: Brownian motion HFD ~1.5, white noise DFA alpha ~0.5.

ECG and Heart Rate Variability
------------------------------

.. code-block:: python

   from morie.signal import ecgdet, rrint, hrvtd, hrvfd, hrvnl
   import math
   # a synthetic ECG-like train: one spike per beat at 72 bpm, sampled at 360 Hz (use your own lead)
   ecg = [1.0 if i % 300 < 4 else 0.05 * math.sin(2 * math.pi * 1.2 * i / 360) for i in range(3600)]

   peaks = ecgdet(ecg, fs=360)              # Pan-Tompkins QRS detection
   rr = rrint(peaks.extra["r_peaks"], fs=360).extra["rr_ms"]   # RR intervals in ms
   td = hrvtd(rr)                            # SDNN, RMSSD, pNN50
   fd = hrvfd(rr)                            # VLF/LF/HF power
   nl = hrvnl(rr)                            # Poincare SD1/SD2

Pan-Tompkins detector (Pan & Tompkins, 1985) with adaptive thresholding.
HRV metrics follow Task Force (1996) standards.

Phonocardiogram (PCG) Analysis
------------------------------

For cardiotoxicity studies in addiction/substance use research:

.. code-block:: python

   from morie.signal import pcgflt, pcgenv, pcgseg, pcgmur
   import math
   # a synthetic phonocardiogram: two short bursts per cycle (S1, S2) at 2 kHz (use your own recording)
   pcg = [(math.sin(2 * math.pi * 60 * i / 2000) if (i % 1600) < 120 or 700 <= (i % 1600) < 780 else 0.0) for i in range(8000)]

   filtered = pcgflt(pcg, fs=2000)           # 25-400 Hz bandpass
   envelope = pcgenv(pcg, fs=2000)           # Shannon energy envelope
   segments = pcgseg(envelope.filtered, fs=2000)  # S1/S2 segmentation
   score = pcgmur(pcg, fs=2000)              # Murmur detection score

The murmur detection score combines Higuchi fractal dimension, high-frequency
energy ratio, and spectral entropy. Scores are uncalibrated (0-1 range);
calibration requires labeled clinical data.

References
----------

- Higuchi, T. (1988). Approach to an irregular time series on the basis of
  the fractal theory. *Physica D*, 31(2), 277-283.
- Pan, J. & Tompkins, W.J. (1985). A real-time QRS detection algorithm.
  *IEEE Trans. Biomed. Eng.*, 32(3), 230-236.
- Task Force of ESC/NASPE (1996). Heart rate variability: Standards of
  measurement. *Circulation*, 93(5), 1043-1065.
- Rangayyan, R.M. (2015). *Biomedical Signal Analysis*. IEEE Press.
