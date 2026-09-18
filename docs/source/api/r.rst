R API
=====

Part of :doc:`index` — MORIE API reference.

A curated reference for the R package (published as ``rmorie`` on
r-universe; the repository copy under ``r-package/morie`` is the same
code). Signatures and descriptions come from the Roxygen2 ``.Rd`` files
in ``r-package/morie/man/``; see :doc:`../methods/index` for the
methodology behind each function.

The package exports 12,046 functions, 4,851 of them ``morie_*`` entry
points; this page lists the ones a first analysis reaches for. The
complete reference, one page per function, is the pkgdown site at
https://rootcoder007.github.io/rmorie/reference/ .

Causal estimators
-----------------

.. r:function:: morie_estimate_aipw
.. r:function:: morie_estimate_atc
.. r:function:: estimate_ate
.. r:function:: morie_estimate_att
.. r:function:: morie_estimate_cate
.. r:function:: morie_estimate_g_computation
.. r:function:: morie_estimate_gate
.. r:function:: morie_estimate_late
.. r:function:: morie_estimate_propensity_scores
.. r:function:: morie_dml_clustered

Causal DAG toolkit (native)
---------------------------

Native DAG construction, backdoor identification, estimation, and
refutation — a DoWhy-style workflow with no external graph packages.

.. r:function:: morie_dag
.. r:function:: morie_dag_identify
.. r:function:: morie_dag_estimate
.. r:function:: morie_dag_refute
.. r:function:: morie_mrm_dags

Matching (native engines)
-------------------------

Nearest-neighbour, Mahalanobis, exact, coarsened-exact, optimal-pair,
genetic, and cardinality matching, plus balance diagnostics — all
implemented natively (no MatchIt/Matching/optmatch runtime
dependency).

.. r:function:: morie_matching_nearest_neighbor
.. r:function:: morie_matching_mahalanobis
.. r:function:: morie_matching_exact
.. r:function:: morie_matching_cem
.. r:function:: morie_matching_optimal_pair
.. r:function:: morie_matching_genetic
.. r:function:: morie_matching_cardinality
.. r:function:: morie_matching_balance

Effect sizes + tests
--------------------

.. r:function:: morie_anova_one_way
.. r:function:: morie_chi_square_test
.. r:function:: cohens_d
.. r:function:: cramers_v
.. r:function:: e_value
.. r:function:: effective_sample_size
.. r:function:: eta_squared
.. r:function:: fisher_exact_test
.. r:function:: hedges_g
.. r:function:: kendall_tau
.. r:function:: morie_kruskal_wallis_test
.. r:function:: levene_test
.. r:function:: morie_mann_whitney_test
.. r:function:: morie_odds_ratio_ci
.. r:function:: omega_squared
.. r:function:: morie_one_sample_t_test

Survey + sampling
-----------------

.. r:function:: morie_bootstrap_sample
.. r:function:: morie_calibration_weights
.. r:function:: morie_cluster_sample
.. r:function:: morie_compute_design_weights
.. r:function:: design_effect
.. r:function:: morie_generate_synthetic_data
.. r:function:: morie_jackknife_estimate

Datasets + I/O
--------------

.. r:function:: morie_canonicalize_cpads_data
.. r:function:: morie_load_cpads_data
.. r:function:: morie_builtin_db
.. r:function:: morie_cache_file
.. r:function:: morie_cache_list
.. r:function:: morie_cache_load
.. r:function:: morie_cache_store
.. r:function:: morie_dataset_catalog
.. r:function:: morie_dataset_info
.. r:function:: morie_db_connect
.. r:function:: morie_download_bootstrap
.. r:function:: morie_fetch_ckan
.. r:function:: morie_list_datasets
.. r:function:: morie_load_cpads
.. r:function:: morie_load_dataset
.. r:function:: morie_paths
.. r:function:: morie_userguide

Workflow + audit
----------------

.. r:function:: morie_ask_percy
.. r:function:: morie_audit_public_outputs
.. r:function:: morie_build_outputs_manifest
.. r:function:: morie_build_prompt
.. r:function:: morie_cpads_contract
.. r:function:: morie_default_synthetic_name_map
.. r:function:: morie_default_workflow_map
.. r:function:: morie_find_project_root
.. r:function:: morie_list_morie_modules

Other
-----

.. r:function:: morie_paired_t_test
.. r:function:: morie_point_biserial_r
.. r:function:: morie_power_prop_test
.. r:function:: morie_power_t_test
.. r:function:: morie_pps_sample
.. r:function:: morie_proportion_ci
.. r:function:: morie_read_outputs_manifest
.. r:function:: morie_risk_difference_ci
.. r:function:: morie_risk_ratio_ci
.. r:function:: morie_run_ebac_selection_ipw_analysis
.. r:function:: morie_run_morie_module
.. r:function:: morie_run_morie_modules
.. r:function:: morie_run_pipeline
.. r:function:: morie_run_propensity_ipw_analysis
.. r:function:: morie_run_workflow_step
.. r:function:: morie_sample_size_logistic
.. r:function:: sensitivity_rosenbaum
.. r:function:: morie_shapiro_wilk_test
.. r:function:: morie_simple_random_sample
.. r:function:: morie_spearman_rho
.. r:function:: morie_stratified_sample
.. r:function:: morie_summarize_output_audit
.. r:function:: morie_two_sample_t_test
.. r:function:: morie_validate_cpads_data
.. r:function:: morie_validate_outputs_manifest
.. r:function:: morie_wilcoxon_signed_rank_test
.. r:function:: morie_write_synthetic_data

