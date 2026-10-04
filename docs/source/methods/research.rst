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
   R.meta_random_effects([0.20, 0.35, 0.10], [0.010, 0.020, 0.015])  # DerSimonian-Laird, with the truncation flag

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
   * - P1 Le Cam two-point bound
     - ``P1LeCam.sum_min``, ``P1LeCam.two_point``, ``P1LeCam.minimax``
     - :func:`two_point_bound`
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
     - ``P10.cluster_size_of_lt_one``, ``P10.endogeneity_share``, ``P10.extinction_le_fixed``, ``P10.supercritical_extinction_lt_one``
     - :func:`contagion_branching`, :func:`contagion_extinction`
   * - P11 sentencing effects as intervals
     - ``P11.Pop.outcome_bounds``, ``P11.clean_bounds``, ``P11.Pop.mtr_upper``, ``P11.Pop.mts_ate_le_naive``, ``P11.im_cutoff_antitone``, ``P11.two_sided_overcovers``
     - :func:`sentence_effect_bounds`, :func:`contaminated_bounds`, :func:`sentence_effect_mtr`, :func:`sentence_effect_mts`, :func:`bounds_confidence`
   * - P12 ecological inference
     - ``P12.cov_decomp``, ``P12.ecological_ge``, ``P12.dd_bounds``
     - :func:`ecological_decompose`, :func:`ecological_bounds`
   * - P13 pooling evaluations
     - ``P13.truncation_bias``, ``P13.dl_biased_under_homogeneity``, ``P13.re_var_ge``
     - :func:`meta_random_effects`, :func:`meta_dl_bias`
   * - P13 HKSJ interval
     - ``P13HKSJ.hksj_wider_iff``, ``P13HKSJ.Q_eq_zero_iff``, ``P13HKSJ.hksj_equal_weights``
     - :func:`meta_hksj`
   * - P14 judge leniency
     - ``P14.itt_decomposition``, ``P14.first_stage_decomposition``, ``P14.late_identification``, ``P14.wald_with_defiers``, ``P14.defiers_can_flip``
     - :func:`judge_iv_population`, :func:`judge_iv`
   * - P14 many-judge slope test
     - ``P14Slope.propensity_mono``, ``P14Slope.outcome_diff``, ``P14Slope.slope_bound``, ``P14Slope.violation_refutes_monotonicity``
     - :func:`judge_slope_test`
   * - P15 disparity decomposition
     - ``P15.twofold_B``, ``P15.threefold``, ``P15.reference_dependence``, ``P15.attribution_shift``
     - :func:`disparity_decomposition`
   * - P15 DFL reweighting
     - ``P15Reweight.reweighting_matches``, ``P15Reweight.reweighted_mass``, ``P15Reweight.counterfactual_outcome``
     - :func:`dfl_reweight`
   * - P16 court backlog
     - ``P16.occupancy_integral``, ``P16.little``, ``P16.little_backlog``, ``P16.little_target``
     - :func:`court_backlog`
   * - P16 disposed-cases mean as a bound
     - ``P16Censoring.true_mean_ge``, ``P16Censoring.bias_lower``, ``P16Censoring.disposed_understates``, ``P16Censoring.no_upper_bound``
     - :func:`backlog_censoring`
   * - P17 incapacitation
     - ``P17.steady_state_rate``, ``P17.prevented_share_lt_one``, ``P17.marginal_prevention_eq``, ``P17.high_rate_more_prevented``
     - :func:`incapacitation`
   * - P17 desistance and replacement
     - ``P17Replacement.prevented_le_const``, ``P17Replacement.prevented_ge_const``, ``P17Replacement.later_sentence_prevents_less``, ``P17Replacement.prevented_net_le``
     - :func:`incapacitation_career`
   * - P18 selective labels
     - ``P18.nested_rate_identified``, ``P18.unobserved_bounds``, ``P18.unobserved_width``
     - :func:`selective_labels`
   * - P19 regression to the mean
     - ``P19.exchange_cross``, ``P19.indicator_bound``, ``P19.selected_change_nonpos``
     - :func:`regression_to_mean`
   * - P19 empirical-Bayes shrinkage
     - ``P19Shrinkage.loss_eq``, ``P19Shrinkage.loss_min``, ``P19Shrinkage.loss_bstar_le_raw``, ``P19Shrinkage.predicted_fall``
     - :func:`hotspot_shrinkage`, :func:`shrinkage_loss`

Reference
---------

.. automodule:: morie.research
   :members:
   :undoc-members:
