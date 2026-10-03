"""Purpose lines for every module output table, and a glossary of the columns they share.

``morie explain FILE`` prints the purpose of the table and, when FILE exists, explains each
column actually in its header (so the text never describes a column the table lacks).
"""

from __future__ import annotations

PURPOSE: dict[str, str] = {
    # distribution-tests
    "distribution_tests.csv": "Distributional checks of the key measures (normality tests and summary shape) before any model assumes a form.",
    "alcohol_correlation_matrix.csv": "Pairwise correlations between the alcohol measures.",
    "clt_convergence.csv": "How fast sample means settle as the sample grows: the observed spread of means against the CLT standard error.",
    # bayesian-inference
    "bayesian_posterior_summaries.csv": "Beta-binomial posterior for each endpoint's prevalence: prior, posterior parameters, mean and credible interval.",
    "bayesian_bayes_factors.csv": "Bayes factors (BF10) comparing the hypotheses for each endpoint; above 1 favours H1.",
    "bayesian_vs_frequentist_ci.csv": "Each endpoint's Bayesian credible interval beside its frequentist confidence interval.",
    # logistic-models
    "logistic_odds_ratios.csv": "Survey-weighted logistic model of heavy drinking: one odds ratio per term (categories against their reference level).",
    "logistic_interaction_odds_ratios.csv": "The same model with a cannabis x gender interaction: odds ratios including the interaction terms.",
    "logistic_interaction_tests.csv": "Joint Wald test of the cannabis x gender interaction terms.",
    "logistic_smote_status.csv": "Whether the SMOTE sensitivity check ran (it is skipped on an already balanced outcome) and the class counts before/after.",
    "logistic_smote_odds_ratios.csv": "Odds ratios refitted on SMOTE-rebalanced data, to compare with the survey-weighted ones (empty when SMOTE was skipped).",
    # model-comparison
    "model_comparison_summary.csv": "Nested heavy-drinking models side by side: fit statistics (deviance, AIC, pseudo R2) as terms are added.",
    "model_comparison_full_coefs.csv": "Every coefficient of every nested model, for tracing how estimates move as covariates enter.",
    "model_comparison_interaction.csv": "Coefficients of the model with the cannabis x gender interaction.",
    "model_comparison_wald_tests.csv": "Wald tests for the blocks of terms each nested model adds.",
    # regression-models
    "regression_coefficients.csv": "Survey-weighted regression of the eBAC outcomes: coefficients, standard errors and intervals.",
    "regression_model_comparison.csv": "Fit of the competing eBAC regression specifications.",
    # propensity-scores
    "ipw_results.csv": "Inverse-probability-weighted effect of cannabis use on heavy drinking, with the unweighted contrast for comparison.",
    "ipw_diagnostics.csv": "Weight diagnostics for the IPW fit: weight range and effective sample size; extreme weights flag poor overlap.",
    # causal-estimators
    "causal_estimator_comparison.csv": "The same effect estimated by IPW, outcome regression and AIPW (doubly robust) side by side.",
    # treatment-effects
    "treatment_effects_summary.csv": "Average effect of cannabis use on heavy drinking: ATE, ATT and ATC with intervals.",
    "cate_subgroup_estimates.csv": "Conditional (subgroup) treatment effects: the effect within each level of a grouping variable.",
    # dag-specification
    "official_doc_alignment_checklist.csv": "Checklist mapping each design requirement from the official documentation to where the analysis meets it.",
    # ebac-core
    "ebac_data_quality_checks.csv": "Data-quality checks on the eBAC inputs (valid ranges, missing values) with pass/fail.",
    "ebac_distribution_unweighted.csv": "Unweighted distribution of eBAC: quantiles, mean and spread.",
    "ebac_model_samples.csv": "Size of each analysis sample the eBAC models use, after their exclusions.",
    "ebac_weighted_summaries.csv": "Survey-weighted eBAC summaries by group (means and prevalence of the legal-limit flag).",
    "ebac_missingness_weighted.csv": "Weighted share of respondents with a missing eBAC, by group.",
    "ebac_missingness_or.csv": "Logistic model of eBAC missingness: which characteristics predict a missing eBAC (odds ratios).",
    "ebac_missingness_or_eligible_drinkers.csv": "The missingness model restricted to eligible drinkers.",
    "ebac_logistic_or_primary.csv": "Primary logistic model of the eBAC legal-limit flag: odds ratios.",
    "ebac_linear_coefficients_primary.csv": "Primary linear model of continuous eBAC: coefficients.",
    "ebac_logistic_or_sensitivity_with_heavy.csv": "Sensitivity version of the logistic eBAC model adding heavy drinking as a covariate.",
    "ebac_linear_coefficients_sensitivity_with_heavy.csv": "Sensitivity version of the linear eBAC model adding heavy drinking as a covariate.",
    # ebac-selection-adjustment-ipw
    "ebac_ipw_weight_diagnostics.csv": "Diagnostics of the selection weights for having an observed eBAC (range, effective sample size).",
    "ebac_ipw_logistic_or.csv": "Selection-weighted logistic eBAC model: odds ratios corrected for who reports eBAC.",
    "ebac_ipw_linear_coefficients.csv": "Selection-weighted linear eBAC model: coefficients.",
    "ebac_ipw_cannabis_comparison.csv": "The cannabis coefficient with and without the selection weights.",
    "ebac_ipw_observation_model_or.csv": "The observation (selection) model: odds of having an observed eBAC.",
    "ebac_ipw_covariate_balance.csv": "Covariate balance before and after weighting: standardized mean differences (|SMD| < 0.1 is the usual target).",
    "ebac_final_ipw_diagnostics.csv": "Final-report copy of the selection-weight diagnostics.",
    "ebac_final_ipw_or.csv": "Final-report copy of the selection-weighted odds ratios.",
    "ebac_final_ipw_linear.csv": "Final-report copy of the selection-weighted linear coefficients.",
    "ebac_final_ipw_comparison.csv": "Final-report comparison of weighted and unweighted eBAC estimates.",
    # ebac-integrations
    "ebac_final_domain_samples.csv": "Sample size of each analysis domain in the integrated eBAC report.",
    "ebac_final_formula_input_audit.csv": "Audit of the inputs to the eBAC formula (drinks, weight, hours, sex): presence and valid ranges.",
    "ebac_final_formula_validation.csv": "eBAC recomputed by the Widmark formula against the stored eBAC, row checks summarised.",
    "ebac_final_interaction_tests.csv": "Interaction tests of the integrated eBAC models.",
    "ebac_final_weighted_descriptives.csv": "Weighted eBAC descriptives for the integrated report.",
    "ebac_final_weighted_linear.csv": "Weighted linear eBAC model for the integrated report.",
    "ebac_final_weighted_or.csv": "Weighted logistic eBAC model (odds ratios) for the integrated report.",
    "ebac_final_smote_compare.csv": "Integrated report: odds ratios with and without SMOTE rebalancing.",
    "ebac_final_smote_or.csv": "Integrated report: SMOTE-refitted odds ratios.",
    "ebac_final_smote_status.csv": "Integrated report: whether SMOTE ran and the class counts.",
    "ebac_final_causal_effects.csv": "Integrated report: causal effect estimates of cannabis on eBAC outcomes.",
    "ebac_final_cate.csv": "Integrated report: subgroup (conditional) effects on eBAC.",
    "ebac_final_consistency_checks.csv": "Consistency checks between the integrated tables (the same quantity agrees across them).",
    "ebac_final_crosswalk_previous.csv": "Crosswalk of these results against the previous run's.",
    "ebac_final_dml_results.csv": "Cross-fitted double machine learning estimate of the cannabis effect on eBAC.",
    "ebac_final_dml_status.csv": "Whether the DML stage ran and with which learners.",
    "ebac_final_key_summary.csv": "One-page summary of the headline eBAC numbers.",
    "ebac_final_user_guide_variable_map.csv": "Map from each analysis variable to its description in the survey user guide.",
    "ebac_final_variable_audit.csv": "Audit of every analysis variable: present in the wrangled data and coded as documented.",
    # ebac-gender-smote-sensitivity
    "ebac_gender_interaction_svy_or.csv": "Survey-weighted eBAC model with a gender interaction: odds ratios.",
    "ebac_gender_interaction_tests.csv": "Tests of the gender interaction terms.",
    "ebac_gender_marginal_probs.csv": "Predicted probability of exceeding the legal limit by gender (marginal probabilities).",
    "ebac_smote_status.csv": "Whether the eBAC SMOTE sensitivity ran and the class counts before/after.",
    "ebac_smote_or.csv": "eBAC odds ratios refitted on SMOTE-rebalanced data.",
    "ebac_smote_compare.csv": "eBAC odds ratios with and without SMOTE, side by side.",
    # final-report
    "ebac_final_output_coverage.csv": "Which expected output files exist after the run.",
    "ebac_final_output_shapes.csv": "Row and column counts of each output file.",
    "ebac_final_script_run_status.csv": "Run status of each analysis step (completed, warnings).",
    "ebac_final_audit_checks.csv": "Final audit checks with pass/fail.",
    # otis-analysis
    "otis_descriptives.csv": "Descriptive counts from the OTIS restrictive-confinement data (placements, individuals, days).",
    "otis_alert_combos.csv": "How often each combination of mental-health/suicide alerts occurs among placements.",
    "otis_dml_results.csv": "Cross-fitted double machine learning estimate of an alert's effect on confinement duration.",
    "otis_trends.csv": "Counts by fiscal year: the trend over time.",
    # mapq-psychometrics
    "mapq_reliability.csv": "Reliability of each MAPQ subscale and the total: Cronbach's alpha and McDonald's omega.",
    "mapq_factor_loadings.csv": "Factor loadings of the 20 MAPQ items on the four factors, with each item's assigned subscale.",
    "mapq_dml_results.csv": "Cross-fitted DML estimate of the gender effect on the Knowledge Scale score.",
}

GLOSSARY: dict[str, str] = {
    "term": "the model term (a category appears as its level against the reference)",
    "estimate": "the point estimate",
    "coef": "the coefficient",
    "log_odds": "the coefficient on the log-odds scale",
    "se": "standard error of the estimate",
    "std.error": "standard error of the estimate",
    "SE": "standard error of the estimate",
    "statistic": "the test statistic",
    "t_value": "the t statistic (estimate / SE)",
    "F_stat": "the F statistic",
    "F_statistic": "the F statistic",
    "df_num": "numerator degrees of freedom",
    "df_den": "denominator degrees of freedom",
    "df_denom": "denominator degrees of freedom",
    "df_residual": "residual degrees of freedom",
    "p_value": "p-value (not corrected for multiple comparisons)",
    "p.value": "p-value (not corrected for multiple comparisons)",
    "significant": "* when p < 0.05",
    "OR": "odds ratio = exp(coefficient); above 1 raises the odds",
    "or": "odds ratio = exp(coefficient); above 1 raises the odds",
    "OR_lower95": "lower bound of the 95% confidence interval of the odds ratio",
    "OR_upper95": "upper bound of the 95% confidence interval of the odds ratio",
    "or_lower95": "lower bound of the 95% confidence interval of the odds ratio",
    "or_upper95": "upper bound of the 95% confidence interval of the odds ratio",
    "OR_lower": "lower bound of the odds ratio's interval",
    "OR_upper": "upper bound of the odds ratio's interval",
    "OR_original": "odds ratio on the original data",
    "OR_smote": "odds ratio on the SMOTE-rebalanced data",
    "p_original": "p-value on the original data",
    "p_smote": "p-value on the SMOTE-rebalanced data",
    "ci_lower": "lower bound of the 95% interval",
    "ci_upper": "upper bound of the 95% interval",
    "ci_lower95": "lower bound of the 95% interval",
    "ci_upper95": "upper bound of the 95% interval",
    "conf.low": "lower bound of the 95% interval",
    "conf.high": "upper bound of the 95% interval",
    "ci_width": "width of the interval",
    "ate": "average treatment effect",
    "cate": "conditional (subgroup) average treatment effect",
    "estimand": "which effect the row estimates (ATE, ATT, ATC)",
    "n": "number of records",
    "n_total": "number of records in total",
    "n_treated": "number of treated records",
    "n_control": "number of control records",
    "n_nonmissing": "records with a value",
    "sample_size": "records in the sample",
    "weight": "survey weight",
    "mean": "mean",
    "median": "median",
    "sd": "standard deviation",
    "min": "minimum",
    "max": "maximum",
    "p25": "25th percentile",
    "p75": "75th percentile",
    "p90": "90th percentile",
    "p95": "95th percentile",
    "prevalence": "share of records with the outcome",
    "method": "how the row was computed",
    "model": "which model the row comes from",
    "outcome": "the outcome variable",
    "treatment": "the treatment variable",
    "predictor": "the predictor variable",
    "variable": "the variable",
    "covariate": "the covariate",
    "group": "the group",
    "subgroup_var": "the variable that defines the subgroups",
    "subgroup_level": "the subgroup's level",
    "smd_raw": "standardized mean difference before weighting",
    "smd_ipw": "standardized mean difference after weighting (|SMD| < 0.1 is the usual balance target)",
    "deviance": "model deviance (lower fits better on the same data)",
    "AIC_approx": "approximate AIC (lower is better)",
    "pseudo_R2": "McFadden pseudo R-squared",
    "bf10": "Bayes factor for H1 over H0 (> 1 favours H1)",
    "alpha_prior": "Beta prior alpha",
    "beta_prior": "Beta prior beta",
    "alpha_post": "Beta posterior alpha",
    "beta_post": "Beta posterior beta",
    "post_mean": "posterior mean",
    "post_sd": "posterior standard deviation",
    "alpha_raw": "Cronbach's alpha from the raw items",
    "alpha_std": "Cronbach's alpha from the standardized items",
    "alpha_ci_low": "lower bound of alpha's interval",
    "alpha_ci_high": "upper bound of alpha's interval",
    "omega_total": "McDonald's omega total",
    "omega_hier": "omega hierarchical (the general factor's share, Schmid-Leiman)",
    "splithalf_sb": "split-half reliability, Spearman-Brown corrected",
    "n_items": "items in the scale",
    "status": "run status",
    "check": "the check",
    "check_name": "the check",
    "pass": "TRUE when the check passed",
    "note": "a note on the row",
    "metric": "the quantity named on the row",
    "value": "its value",
    "power": "statistical power (probability of detecting the effect)",
    "marginal_prob": "predicted probability averaged over the sample",
}


def explain_table(name: str, path: str | None = None) -> str | None:
    """The purpose of a module table plus its columns (read from the file when it exists); None if unknown."""
    import csv
    import os

    purpose = PURPOSE.get(name)
    if purpose is None:
        return None
    lines = [purpose]
    if path and os.path.isfile(path):
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            header = next(csv.reader(fh), [])
        if header:
            width = max(len(c) for c in header)
            lines += ["", "Columns:"]
            for col in header:
                lines.append(
                    f"  {col:<{width}}  {GLOSSARY.get(col, '(module-specific; see the module documentation)')}"
                )
    else:
        lines += ["", "Run `morie explain` on the file itself to see its columns explained."]
    return "\n".join(lines)
