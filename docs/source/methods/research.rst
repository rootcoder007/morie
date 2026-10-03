Research: the hardest problems, with proofs
===========================================

``morie.research`` carries the research programme on the hardest open problems
in criminology and sociolegal studies. Every function rests on a theorem checked
in Lean 4 with Mathlib (``research/lean`` in the rmorie repository: 0 ``sorry``,
axioms ``propext``, ``Classical.choice`` and ``Quot.sound`` only), names the
theorems it uses in its ``theorems`` field, and says where the empirical
assumption enters. Lean certifies the implication, never the antecedent: no proof
shows that two lists are independent, that a noise box is right, or that
spillover stops at the ring the analyst named.

The R package (``rmorie``) carries the same functions with the ``morie_`` prefix.
The two arms agree to rounding on the same inputs, and the stochastic ones draw
from the shared Philox stream, so a seed gives the same path in both.

.. code-block:: python

   from morie import research as R

   R.dark_figure_two_source(400, 250, 80, kappa=2)   # Lincoln-Petersen with a dependence box
   R.feedback_loop_limit(0.3, 0.2, 10, 10)            # the proved limit of the patrol share
   R.meta_random_effects(estimates, variances)        # DerSimonian-Laird, with the truncation flag

Problems and functions
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Problem
     - Theorems (``Research.*``)
     - Functions
   * - P1 dark figure
     - ``P1.TwoSource.petersen_bounds``, ``P1.true_rate_bounds``, ``P1.dark_figure_bounds``, ``P1.conclusion_holds_below_breakdown``, ``P1.three_list_saturated_fits``, ``P1.offence_count_bounds``
     - :func:`dark_figure_two_source`, :func:`dark_figure_bounds`, :func:`dark_figure_breakdown`, :func:`dark_figure_three_list`, :func:`dark_figure_hierarchy`
   * - P2 selection in police records
     - ``P2.rate_bounds``, ``P2.disparity_bounds``, ``P2.benchmark_product``, ``P2.offset_shift``, ``P2.rr_between``, ``P2.dyad_ratio``, ``P2.collider_or_eq_background``
     - :func:`disparity_exposure_bounds`, :func:`disparity_benchmark`, :func:`relative_risk_from_or`, :func:`interracial_rates`, :func:`collider_arrest`
   * - P3 interference
     - ``P3.Model.exposure_adjustment``, ``P3.Design.ht_unbiased``, ``P3.Design.ht_variance``, ``P3.SYG.syg_eq_ht``, ``P3.cheeger_easy``
     - :func:`spillover_exposure`, :func:`spillover_effects`, :func:`spillover_ht`, :func:`spillover_ht_variance`, :func:`spillover_exposure_probs`, :func:`cheeger_bound`
   * - P4 predictive-policing feedback
     - ``P4.naive_step_drift``, ``P4.naiveShare_tendsto_one``, ``P4.corrected_share_tendsto``, ``P4.naiveShare_rate_bound``, ``P4.rho_cap``, ``P4.Urn.polya_uniform``
     - :func:`feedback_loop_meanfield`, :func:`feedback_loop_limit`, :func:`feedback_loop_urn_law`, :func:`feedback_loop_sim`, :func:`feedback_loop_bound`
   * - P5 risk scores under label bias
     - ``P5.Table.chouldechova``, ``P5.impossibility``, ``P5.true_base_rate_bounds``, ``P5.compare_decided``, ``P5.reduced_coefficient``, ``P5.rank_stable_of_gap``, ``P5.hr2_gt_one_of_depletion``, ``P5.no_mle``
     - :func:`fairness_rates`, :func:`fairness_implied_fpr`, :func:`fairness_base_rate_bounds`, :func:`fairness_true_rate`, :func:`fairness_compare_groups`, :func:`logit_rescale`, :func:`ranking_resolution`, :func:`hazard_selection`, :func:`logit_separation`
   * - P6 age-crime curve
     - ``P6.aggregate_not_identifying``
     - :func:`age_crime_aggregate`
   * - P7 crime concentration
     - ``P7.gini_zero_decomposition``, ``P7.poisson_zero_prob``, ``P7.Mixture.mixture_var_ge_mean``, ``P7.expectedDistinct_bounds``
     - :func:`concentration_gini`, :func:`concentration_decompose`, :func:`concentration_dispersion`, :func:`concentration_distinct_growth`
   * - P8 deterrence
     - ``P8.constant_dimension_not_identified``, ``P8.certainty_monotone``, ``P8.necessity_bounds``
     - :func:`deterrence_design_check`, :func:`deterrence_response`, :func:`probability_of_necessity`
   * - P9 recording as a linear map
     - ``P9.total_invariant_of_colStochastic``, ``P9.detection_rate_rises``
     - :func:`recording_map`, :func:`detection_rate_shift`
   * - P10 near-repeat contagion
     - ``P10.cluster_size_of_lt_one``, ``P10.endogeneity_share``
     - :func:`contagion_branching`
   * - P11 sentencing effects as intervals
     - ``P11.Pop.outcome_bounds``, ``P11.clean_bounds``, ``P11.Pop.mtr_upper``, ``P11.Pop.mts_ate_le_naive``, ``P11.im_cutoff_antitone``, ``P11.two_sided_overcovers``
     - :func:`sentence_effect_bounds`, :func:`contaminated_bounds`, :func:`sentence_effect_mtr`, :func:`sentence_effect_mts`, :func:`bounds_confidence`
   * - P12 ecological inference
     - ``P12.cov_decomp``, ``P12.ecological_ge``, ``P12.dd_bounds``
     - :func:`ecological_decompose`, :func:`ecological_bounds`
   * - P13 pooling evaluations
     - ``P13.truncation_bias``, ``P13.dl_biased_under_homogeneity``, ``P13.re_var_ge``
     - :func:`meta_random_effects`, :func:`meta_dl_bias`

Reference
---------

.. automodule:: morie.research
   :members:
   :undoc-members:
