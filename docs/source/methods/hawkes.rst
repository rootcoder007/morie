Hawkes Self-Exciting Point Processes
=====================================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE implements both the classical Markovian Hawkes process
(constant baseline, exponential excitation kernel) and the
non-stationary, non-Markovian generalisation of Kwan-Chen-Dunsmuir
(2024).

Modules
-------

- ``morie.tps_stochastic.hawkes_temporal_fit`` — classical Markovian
  Hawkes fit (one-parameter family, :math:`O(n)` recursive intensity).
- ``morie.tps_hawkes_advanced.fit_hawkes_general`` — MLE for the
  general (kernel, baseline) pair: projected BFGS on the analytic
  gradient in morie's compiled core (L-BFGS-B without it). ``method``
  picks how the likelihood is evaluated: ``"exact"`` (Ozaki's
  :math:`O(n)` recursion for the exponential kernel; the Weibull and
  gamma double sums stop where the kernel underflows to 0), ``"soe"``
  (Lomax and gamma as a sum of exponentials, Beylkin & Monzón 2010,
  relative error ``eps``), ``"truncate"`` (lags whose kernel tail mass
  exceeds ``eps``), ``"em"`` (Veen & Schoenberg 2008) or ``"inar"``
  (Kirchner 2017, binned counts, constant baseline); ``"auto"`` is
  exact for the exponential kernel, truncate for Weibull and soe for
  Lomax and gamma.
- ``morie.tps_hawkes_advanced.compare_hawkes_kernels`` — fits all
  eight (kernel \times baseline) combinations and ranks by AIC and
  time-rescaling-residual Kolmogorov-Smirnov goodness-of-fit.
- ``morie.tps_hawkes_advanced.hawkes_markovian_vs_nonmarkovian`` —
  focused 2-way comparison: Markovian classical vs Gamma + sinusoidal.

Mathematical content
--------------------

For a simple point process :math:`N` on :math:`[0, T]` with conditional
intensity

.. math::

    \lambda(t) \;=\; \nu(t) \;+\; \int_{0}^{t-}\! g(t - s)\, dN_s,

the four supported excitation kernels :math:`g(u) = \eta\, \tilde
g(u; \psi)` (with branching ratio :math:`\eta \in (0, 1)`) are:

- **Exponential**: :math:`\tilde g_{\mathrm{exp}}(u; \beta) = \beta\,
  e^{-\beta u}` --- the classical Markovian case.
- **Gamma**: :math:`\tilde g_{\mathrm{gam}}(u; \alpha, \beta) =
  \beta^{\alpha}\, u^{\alpha - 1}\, e^{-\beta u} / \Gamma(\alpha)`.
- **Weibull**: :math:`\tilde g_{\mathrm{wb}}(u; \alpha, \lambda) =
  (\alpha / \lambda)\, (u / \lambda)^{\alpha - 1}\, e^{-(u /
  \lambda)^{\alpha}}`.
- **Lomax (power-law)**: :math:`\tilde g_{\mathrm{lmx}}(u; \alpha, c)
  = (\alpha - 1)\, c^{\alpha - 1}\, (u + c)^{-\alpha}` for
  :math:`\alpha > 1`.

The baseline takes the log-link form

.. math::

    \nu(t; \alpha) \;=\; \exp\!\left(\alpha_0 + \alpha_1\, t / T +
        \alpha_2 \sin(2 \pi t / 365.25) +
        \alpha_3 \cos(2 \pi t / 365.25)\right).

Inference is by maximum likelihood (the fit returns the estimates,
log-likelihood, AIC and BIC; standard errors are not computed).
Goodness-of-fit by the Daley-Vere-Jones / Brown-Frank-Mitra
time-rescaling theorem: the Kolmogorov-Smirnov statistic of the
rescaled inter-event times.

Asymptotic theory
-----------------

Strong consistency and asymptotic normality of the MLE follow from
Kwan-Chen-Dunsmuir (2024). Under regularity conditions on
:math:`\nu(\cdot)` and :math:`\tilde g(\cdot)` (continuity, bounded
moments, :math:`\eta < 1`), the intensity process is asymptotically
ergodic and

.. math::

    \sqrt{n}\, (\hat\theta^n - \theta_0)
    \;\converginD\; \mathcal{N}\!\left(0,\; I(\theta_0)^{-1}\right).

Application
-----------

Applied to Toronto Police Service Assault data (every event from
2014-01-01 to 2026-03-31: :math:`n = 252{,}971`, :math:`T = 4{,}473`
days), the four sinusoidal-baseline rows beat every constant-baseline
row. The best fit (Weibull kernel, sinusoidal baseline, AIC
:math:`-1{,}546{,}549.5`) improves on the Markovian classical Hawkes
(exponential kernel, constant baseline, AIC :math:`-1{,}546{,}007.5`)
by :math:`\Delta\mathrm{AIC} = 542.0`; gamma (:math:`-1{,}546{,}546.3`),
exponential (:math:`-1{,}546{,}544.9`) and Lomax
(:math:`-1{,}546{,}537.1`) follow, all with the sinusoidal baseline.
Branching ratios sit in :math:`[0.561, 0.568]` with the sinusoidal
baseline and :math:`[0.670, 0.780]` with the constant one: part of
what a constant baseline attributes to self-excitation is seasonal.

Reference
---------

The full methodology and Toronto application will appear in a
forthcoming companion paper (in preparation; will be linked here
once publicly available with a DOI or preprint URL).
