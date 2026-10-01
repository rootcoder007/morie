Pollution to health
===================

``morie.envhealth`` (Python) and ``morie_envhealth_*`` (R) turn an ambient
exposure into a disease burden. Every function cites the paper whose
formula it implements, and the two arms agree on fixed inputs to 1e-12 (the
parity test ships with the packages). The command ``verify-pollution`` runs
the chain end to end and logs its assumptions.

Concentration-response
----------------------

PM2.5, the Integrated Exposure-Response curve (Burnett et al. 2014, Eq. 1):

.. math::

   RR(z) = 1 + \alpha \left(1 - e^{-\gamma (z - z_{cf})^{\delta}}\right) \quad (z > z_{cf}),
   \qquad RR(z) = 1 \text{ otherwise}

with the GBD 2013 adult triples (all-cause mortality
:math:`(\alpha, \gamma, \delta) = (1.2, 0.34, 0.72)`, IHD
:math:`(1.91, 0.14, 0.49)`, stroke :math:`(1.46, 0.13, 0.61)`) and the
counterfactual :math:`z_{cf} = 5.8\ \mu g/m^3` (WHO 2021 interim target).

NO2, log-linear (Atkinson et al. 2018; WHO 2021):

.. math::

   RR(z) = \exp\!\left(\beta \, \frac{z - z_{cf}}{10}\right)

with :math:`\beta` per 10 :math:`\mu g/m^3` of 0.039 (all-cause
mortality, childhood asthma) or 0.029 (respiratory), :math:`z_{cf} = 10`.

.. code-block:: python

   from morie import envhealth
   envhealth.concentration_response_pm25(12.0).rr        # 1.8612...
   envhealth.concentration_response_no2(25.0, outcome="respiratory").log_rr

.. code-block:: r

   rmorie::morie_envhealth_crf_pm25(12)$rr
   rmorie::morie_envhealth_crf_no2(25, outcome = "respiratory")$log_rr

Attributable fraction
---------------------

Levin's formula (Rothman, Greenland and Lash 2008, chapter 5), with
:math:`p` the exposure prevalence:

.. math::

   PAF = \frac{p\,(RR - 1)}{1 + p\,(RR - 1)}

Deaths displaced
----------------

The BenMAP-CE health-impact function (US EPA 2018; Anenberg et al. 2010):

.. math::

   \Delta Y = y_0 \, N \, \left(1 - e^{-\beta \Delta x}\right)

with :math:`y_0` the baseline rate per person-year, :math:`N` the
population and :math:`\beta` the log-RR per unit of exposure.

Burden
------

``burden_of_pollution`` / ``morie_envhealth_burden()`` chain the three:
RR at the mean exposure, PAF at the prevalence, attributable cases
:math:`= PAF \times y_0 N` (GBD 2019 Risk Factors Collaborators 2020).
``burden_by_fsa`` applies it per area and sorts the worst first.

Equity
------

The concentration index (Wagstaff, Paci and van Doorslaer 1991), with
:math:`R_i` the fractional income rank :math:`(rank_i - 0.5)/n` and the
population covariance:

.. math::

   CI = \frac{2}{\mu}\,\mathrm{cov}(h_i, R_i)

Negative values mean lower-income units bear more exposure.

Sensitivity
-----------

``exposure_response_sensitivity`` / ``morie_envhealth_sensitivity()`` fit
the partially linear double-ML model (Chernozhukov et al. 2018, section
4.2) with the exposure as a continuous treatment and add a percentile
bootstrap over resamples (Efron and Tibshirani 1993).

The command
-----------

.. code-block:: bash

   morie verify-pollution --pollutant no2 --demo
   morie verify-pollution --pollutant pm25 --exposure-csv exposure.csv --baseline-rate 500 --population 1000000
   rmorie verify-pollution --pollutant no2 --exposure-mean 25 --exposure-prevalence 0.9 --json

The report prints the inputs, an assumption log (exposure above the
reference, prevalence in [0, 1], non-negative baseline, positive
population, supported pollutant), then the RR with its citation, the PAF,
the deaths displaced, the attributable cases and, when an ``income``
column is present, the concentration index. Exit status 0 when every
assumption holds, 1 when one fails (the pipeline is skipped), 2 on a data
error.

References
----------

- Burnett, R. T. et al. (2014). An integrated risk function for estimating
  the global burden of disease attributable to ambient fine particulate
  matter exposure. *Environmental Health Perspectives*, 122(4), 397-403.
- Atkinson, R. W. et al. (2018). Long-term concentration-response functions
  for mortality and NO2. *Environmental Research*, 161, 101-113.
- WHO (2021). *Global Air Quality Guidelines*.
- Rothman, K. J., Greenland, S. and Lash, T. L. (2008). *Modern
  Epidemiology*, 3rd ed., chapter 5.
- US EPA (2018). *BenMAP-CE User's Manual Appendices*; Anenberg, S. C. et
  al. (2010). *Environmental Health Perspectives*, 118(9), 1189-1195.
- Wagstaff, A., Paci, P. and van Doorslaer, E. (1991). On the measurement
  of inequalities in health. *Social Science and Medicine*, 33(5), 545-557.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning.
  *Econometrics Journal*, 21(1), C1-C68.
