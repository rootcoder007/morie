Homomorphic Deconvolution & Cepstral Analysis
==============================================

Part of :doc:`index` — MORIE's statistical-methods reference.

Cepstral analysis transforms convolution into addition via the log domain,
enabling separation of mixed signals (e.g., excitation from impulse response
in PCG recordings).

Real Cepstrum
-------------

The real cepstrum is defined as:

.. math::

   c[n] = \text{IFFT}\left(\log |X(k)|\right)

where :math:`X(k) = \text{FFT}(x[n])`.

.. code-block:: python

   import math
   from morie.signal import cepst

   signal = [math.sin(2 * math.pi * 5 * i / 256) for i in range(512)]
   result = cepst(signal)   # SignalResult: the real cepstrum is result.filtered

Complex Cepstrum
----------------

The complex cepstrum preserves phase information:

.. math::

   \hat{x}[n] = \text{IFFT}\left(\log |X(k)| + j \cdot \text{unwrap}(\angle X(k))\right)

This is invertible: ``inverse_complex_cepstrum(complex_cepstrum(x)) = x``.

.. code-block:: python

   from morie.signal import hcepst
   import math
   signal = [math.sin(2 * math.pi * 5 * i / 256) for i in range(512)]
   result = hcepst(signal)  # SignalResult: the complex cepstrum is result.filtered

Liftering and Deconvolution
---------------------------

Given a signal :math:`y = h * e` (convolution of impulse response and
excitation), the complex cepstrum satisfies:

.. math::

   \hat{y}[n] = \hat{h}[n] + \hat{e}[n]

A low-time lifter (zeroing high-quefrency bins) isolates the slowly varying
minimum-phase component from the rapidly varying excitation.

.. code-block:: python

   from morie.signal import hdecon
   import math
   signal = [math.sin(2 * math.pi * 5 * i / 256) for i in range(512)]

   result = hdecon(signal, cutoff=64)   # low-time lifter at quefrency 64
   print(sorted(result.extra))          # the components the deconvolution returns

Application: PCG S1/S2 Decomposition
-------------------------------------

In phonocardiogram analysis, homomorphic deconvolution separates valve
closure transients (minimum-phase) from chest wall resonance. Combined
with Shannon-energy envelope analysis and Higuchi fractal dimension,
this enables murmur detection in cardiotoxicity studies.

.. code-block:: python

   from morie.signal import pcgflt, hdecon, pcgmur
   import math
   # a synthetic phonocardiogram: two short bursts per cycle (S1, S2) at 2 kHz (use your own recording)
   pcg = [(math.sin(2 * math.pi * 60 * i / 2000) if (i % 1600) < 120 or 700 <= (i % 1600) < 780 else 0.0) for i in range(8000)]

   filtered = pcgflt(pcg, fs=2000)
   decomposed = hdecon(filtered.filtered, cutoff=128)
   score = pcgmur(pcg, fs=2000)

References
----------

- Oppenheim, A.V. & Schafer, R.W. (2009). *Discrete-Time Signal
  Processing*. Prentice Hall, Ch. 13.
- Rangayyan, R.M. (2015). *Biomedical Signal Analysis*. IEEE Press, p. 338.
