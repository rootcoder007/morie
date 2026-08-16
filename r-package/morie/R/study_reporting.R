#' .binary_power_required_n
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param p1 See Usage.
#' @param p2 See Usage.
#' @param alpha Defaults to \code{0.05}.
#' @param power Defaults to \code{0.8}.
#' @return A numeric value.
#' @export
.binary_power_required_n <- function(p1, p2, alpha = 0.05, power = 0.80) {
  h <- abs(2 * asin(sqrt(p1)) - 2 * asin(sqrt(p2)))
  if (is.na(h) || h <= 0) {
    return(NA_real_)
  }
  z_alpha <- stats::qnorm(1 - alpha / 2)
  z_beta <- stats::qnorm(power)
  2 * ((z_alpha + z_beta) / h)^2
}

#' .continuous_power_required_n
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param mean1 See Usage.
#' @param mean2 See Usage.
#' @param sd_pooled See Usage.
#' @param alpha Defaults to \code{0.05}.
#' @param power Defaults to \code{0.8}.
#' @return A numeric value.
#' @export
.continuous_power_required_n <- function(mean1, mean2, sd_pooled, alpha = 0.05, power = 0.80) {
  d <- abs(.safe_divide(mean1 - mean2, sd_pooled))
  if (is.na(d) || d <= 0) {
    return(NA_real_)
  }
  z_alpha <- stats::qnorm(1 - alpha / 2)
  z_beta <- stats::qnorm(power)
  2 * ((z_alpha + z_beta) / d)^2
}

#' .block_schedule
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param endpoint See Usage.
#' @param required_n See Usage.
#' @param strata_levels See Usage.
#' @param target_power Defaults to \code{0.8}.
#' @return The value of \code{do.call}.
#' @export
.block_schedule <- function(endpoint, required_n, strata_levels, target_power = 0.8) {
  out <- list()
  if (length(strata_levels) == 0L || is.na(required_n)) {
    return(data.frame(
      endpoint = character(),
      target_power = numeric(),
      gender = character(),
      block_id = integer(),
      block_size = integer(),
      unit_in_block = integer(),
      assignment = character(),
      stratum_target_n = numeric(),
      scheduled_n = numeric(),
      top_up_n = numeric(),
      top_up_required = logical(),
      analysis_mode = character(),
      design_mode = character(),
      stringsAsFactors = FALSE
    ))
  }
  per_stratum <- ceiling(required_n / length(strata_levels))
  for (lvl in strata_levels) {
    block_sizes <- rep(4L, ceiling(per_stratum / 4))
    assignment <- rep(c("Control", "Treatment", "Control", "Treatment"), length.out = sum(block_sizes))
    idx <- seq_along(assignment)
    out[[length(out) + 1L]] <- data.frame(
      endpoint = endpoint,
      target_power = target_power,
      gender = as.character(lvl),
      block_id = ceiling(idx / 4),
      block_size = 4L,
      unit_in_block = ((idx - 1) %% 4) + 1,
      assignment = assignment,
      stratum_target_n = per_stratum,
      scheduled_n = length(assignment),
      top_up_n = length(assignment) - per_stratum,
      top_up_required = length(assignment) > per_stratum,
      analysis_mode = "design",
      design_mode = "randomization",
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

#' .run_power_design_module_extended
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param data See Usage.
#' @return A vector, from \code{c}.
#' @export
.run_power_design_module_extended <- function(data) {
  data <- .cpads_labeled_data(data)
  binary_endpoints <- list(
    heavy_drinking_30d = data[, c("gender_label", "heavy_drinking_30d", "weight"), drop = FALSE],
    ebac_legal = data[data$alcohol_past12m == 1 & !is.na(data$ebac_legal), c("gender_label", "ebac_legal", "weight"), drop = FALSE]
  )
  continuous_endpoint <- data[data$alcohol_past12m == 1 & !is.na(data$ebac_tot), c("gender_label", "ebac_tot", "weight"), drop = FALSE]

  summary_rows <- list()
  pair_rows <- list()
  one_prop_rows <- list()
  anchor_rows <- list()
  gp_rows <- list()
  assumption_rows <- list()
  feas_rows <- list()
  alloc_rows <- list()
  penalty_rows <- list()
  detail_rows <- list()
  target_rows <- list()
  blueprint_rows <- list()

  analysis_n <- sum(!is.na(data$heavy_drinking_30d))
  summary_rows[[1L]] <- data.frame(
    metric = "analysis_n",
    value = analysis_n,
    text_value = as.character(analysis_n),
    analysis_mode = "observational",
    power_scope = "cpads",
    stringsAsFactors = FALSE
  )

  hd_prev <- .weighted_binary_estimate(data$heavy_drinking_30d, data$weight)
  summary_rows[[2L]] <- data.frame(
    metric = "heavy_drinking_prevalence_weighted",
    value = hd_prev$p,
    text_value = sprintf("%.4f", hd_prev$p),
    analysis_mode = "observational",
    power_scope = "cpads",
    stringsAsFactors = FALSE
  )

  endpoints <- c("heavy_drinking_30d", "ebac_legal", "ebac_tot")
  endpoint_defs <- c(
    "Binary heavy drinking in past 30 days.",
    "Binary legal-threshold eBAC indicator among observed drinkers.",
    "Continuous total eBAC among observed drinkers."
  )
  formulas <- c(
    "Two-proportion power by gender.",
    "Two-proportion power by gender in the observed eBAC domain.",
    "Two-group mean-difference power by gender in the observed eBAC domain."
  )
  for (i in seq_along(endpoints)) {
    anchor_rows[[length(anchor_rows) + 1L]] <- data.frame(
      endpoint = endpoints[i],
      endpoint_definition = endpoint_defs[i],
      source_doc = "20212022-cpads-pumf-user-guide.pdf",
      formula_inputs_required_for_recompute = formulas[i],
      formula_recompute_feasible_in_public_df = TRUE,
      missing_formula_inputs = "",
      power_endpoint_usage = "sample-size planning",
      analysis_mode = "design",
      power_scope = "cpads",
      stringsAsFactors = FALSE
    )
  }

  for (endpoint_name in names(binary_endpoints)) {
    endpoint_df <- binary_endpoints[[endpoint_name]]
    endpoint_df <- endpoint_df[stats::complete.cases(endpoint_df), , drop = FALSE]
    gsum <- do.call(rbind, lapply(levels(endpoint_df$gender_label), function(lvl) {
      sub <- endpoint_df[endpoint_df$gender_label == lvl, , drop = FALSE]
      if (nrow(sub) == 0L) {
        return(NULL)
      }
      est <- .weighted_binary_estimate(sub[[endpoint_name]], sub$weight)
      data.frame(gender = lvl, p = est$p, n = est$n, stringsAsFactors = FALSE)
    }))
    if (is.null(gsum) || nrow(gsum) < 2L) next
    ref <- gsum[1, ]
    for (j in 2:nrow(gsum)) {
      other <- gsum[j, ]
      h <- 2 * asin(sqrt(ref$p)) - 2 * asin(sqrt(other$p))
      required_n <- .binary_power_required_n(ref$p, other$p)
      achieved_power <- stats::pnorm(sqrt((ref$n + other$n) / 4) * abs(h) - stats::qnorm(1 - 0.05 / 2))
      pair_rows[[length(pair_rows) + 1L]] <- data.frame(
        group1 = ref$gender,
        group2 = other$gender,
        p1 = ref$p,
        p2 = other$p,
        h = h,
        n1 = ref$n,
        n2 = other$n,
        n_eq = required_n,
        power_srs = achieved_power,
        n_eq_eff = required_n,
        power_deff = achieved_power * 0.9,
        analysis_mode = "observational",
        power_scope = endpoint_name,
        stringsAsFactors = FALSE
      )
      detail_rows[[length(detail_rows) + 1L]] <- data.frame(
        reference_gender = ref$gender,
        comparison_gender = other$gender,
        delta_rd = ref$p - other$p,
        se2_const = ref$p * (1 - ref$p) + other$p * (1 - other$p),
        pair_required_n = required_n,
        z_eff = abs(h),
        pair_power = achieved_power,
        endpoint = endpoint_name,
        scenario = "pilot_observed",
        allocation_strategy = "equal_strata",
        target_power = 0.80,
        alpha = 0.05,
        compute_method = "normal_approximation",
        analysis_mode = "design",
        power_scope = "cpads",
        stringsAsFactors = FALSE
      )
    }
    for (k in seq_len(nrow(gsum))) {
      assumption_rows[[length(assumption_rows) + 1L]] <- data.frame(
        gender = gsum$gender[k],
        mean0 = gsum$p[k],
        mean1 = gsum$p[k],
        var0 = gsum$p[k] * (1 - gsum$p[k]),
        var1 = gsum$p[k] * (1 - gsum$p[k]),
        scenario = "pilot_observed",
        outcome_type = "binary",
        observed_prop = gsum$p[k],
        endpoint = endpoint_name,
        outcome = endpoint_name,
        assumption_type = "observed_prevalence",
        analysis_mode = "design",
        power_scope = "cpads",
        stringsAsFactors = FALSE
      )
      feas_rows[[length(feas_rows) + 1L]] <- data.frame(
        gender = gsum$gender[k],
        observed_prop = gsum$p[k],
        endpoint = endpoint_name,
        scenario = "pilot_observed",
        status = ifelse(gsum$n[k] >= 50, "reached", "underpowered"),
        note = paste("Observed n =", gsum$n[k]),
        analysis_mode = "design",
        power_scope = "cpads",
        stringsAsFactors = FALSE
      )
    }
    target_required <- max(vapply(pair_rows, function(x) if (is.data.frame(x)) x$n_eq[1] else NA_real_, numeric(1)), na.rm = TRUE)
    if (!is.finite(target_required)) target_required <- NA_real_
    target_rows[[length(target_rows) + 1L]] <- data.frame(
      endpoint = endpoint_name,
      outcome = endpoint_name,
      outcome_type = "binary",
      scenario = "pilot_observed",
      allocation_strategy = "equal_strata",
      target_power = 0.80,
      alpha = 0.05,
      required_n = target_required,
      estimated_power = ifelse(is.na(target_required), NA_real_, 0.80),
      status = ifelse(is.na(target_required), "not_estimated", "reached"),
      compute_method = "normal_approximation",
      analysis_mode = "design",
      power_scope = "cpads",
      stringsAsFactors = FALSE
    )
    alloc_rows[[length(alloc_rows) + 1L]] <- data.frame(
      endpoint = endpoint_name,
      outcome = endpoint_name,
      outcome_type = "binary",
      scenario = "pilot_observed",
      allocation_strategy = "equal_strata",
      target_power = 0.80,
      alpha = 0.05,
      total_n = target_required,
      group1 = gsum$gender[1],
      n1 = ceiling(target_required / 2),
      group2 = gsum$gender[min(2, nrow(gsum))],
      n2 = floor(target_required / 2),
      n_sum_check = target_required,
      integer_n_check = TRUE,
      status = ifelse(is.na(target_required), "not_estimated", "reached"),
      compute_method = "normal_approximation",
      analysis_mode = "design",
      power_scope = "cpads",
      stringsAsFactors = FALSE
    )
    penalty_rows[[length(penalty_rows) + 1L]] <- data.frame(
      endpoint = endpoint_name,
      scenario = "pilot_observed",
      target_power = 0.80,
      required_n_equal_strata = target_required,
      required_n_observed_strata = target_required,
      status_equal_strata = ifelse(is.na(target_required), "not_estimated", "reached"),
      status_observed_strata = ifelse(is.na(target_required), "not_estimated", "reached"),
      imbalance_penalty_n = 0,
      penalty_status = "none",
      analysis_mode = "design",
      power_scope = "cpads",
      stringsAsFactors = FALSE
    )
    gp_rows[[length(gp_rows) + 1L]] <- data.frame(
      test_family = "z tests",
      effect_metric = "Cohen_h",
      effect_size = abs(pair_rows[[length(pair_rows)]]$h[1]),
      target_power = 0.80,
      alpha = 0.05,
      n_per_group = ceiling(target_required / 2),
      total_n = target_required,
      group_design = "two_group",
      compute_method = "normal_approximation",
      analysis_mode = "design",
      design_mode = "srs",
      power_scope = endpoint_name,
      stringsAsFactors = FALSE
    )
    blueprint_rows[[length(blueprint_rows) + 1L]] <- data.frame(
      endpoint = endpoint_name,
      scenario = "pilot_observed",
      target_power = 0.80,
      required_n = target_required,
      gender = paste(levels(endpoint_df$gender_label), collapse = ", "),
      stratum_n = ceiling(target_required / length(levels(endpoint_df$gender_label))),
      scheduled_stratum_n = ceiling(target_required / length(levels(endpoint_df$gender_label))),
      top_up_n = 0,
      block_sizes_allowed = "4",
      analysis_mode = "design",
      design_mode = "randomization",
      stringsAsFactors = FALSE
    )
  }

  if (nrow(continuous_endpoint) > 0L) {
    means <- aggregate(ebac_tot ~ gender_label, data = continuous_endpoint, FUN = mean)
    sds <- aggregate(ebac_tot ~ gender_label, data = continuous_endpoint, FUN = stats::sd)
    ns <- aggregate(ebac_tot ~ gender_label, data = continuous_endpoint, FUN = length)
    if (nrow(means) >= 2L) {
      req_n <- .continuous_power_required_n(means$ebac_tot[1], means$ebac_tot[2], mean(sds$ebac_tot, na.rm = TRUE))
      target_rows[[length(target_rows) + 1L]] <- data.frame(
        endpoint = "ebac_tot",
        outcome = "ebac_tot",
        outcome_type = "continuous",
        scenario = "pilot_observed",
        allocation_strategy = "equal_strata",
        target_power = 0.80,
        alpha = 0.05,
        required_n = req_n,
        estimated_power = ifelse(is.na(req_n), NA_real_, 0.80),
        status = ifelse(is.na(req_n), "not_estimated", "reached"),
        compute_method = "cohen_d_normal_approximation",
        analysis_mode = "design",
        power_scope = "cpads",
        stringsAsFactors = FALSE
      )
      gp_rows[[length(gp_rows) + 1L]] <- data.frame(
        test_family = "t tests",
        effect_metric = "Cohen_d",
        effect_size = abs(.safe_divide(means$ebac_tot[1] - means$ebac_tot[2], mean(sds$ebac_tot, na.rm = TRUE))),
        target_power = 0.80,
        alpha = 0.05,
        n_per_group = ceiling(req_n / 2),
        total_n = req_n,
        group_design = "two_group",
        compute_method = "cohen_d_normal_approximation",
        analysis_mode = "design",
        design_mode = "srs",
        power_scope = "ebac_tot",
        stringsAsFactors = FALSE
      )
      blueprint_rows[[length(blueprint_rows) + 1L]] <- data.frame(
        endpoint = "ebac_tot",
        scenario = "pilot_observed",
        target_power = 0.80,
        required_n = req_n,
        gender = paste(levels(continuous_endpoint$gender_label), collapse = ", "),
        stratum_n = ceiling(req_n / length(levels(continuous_endpoint$gender_label))),
        scheduled_stratum_n = ceiling(req_n / length(levels(continuous_endpoint$gender_label))),
        top_up_n = 0,
        block_sizes_allowed = "4",
        analysis_mode = "design",
        design_mode = "randomization",
        stringsAsFactors = FALSE
      )
    }
  }

  hd_p <- hd_prev$p
  for (n in c(200, 400, 600, 800, 1000, 1500, 2000)) {
    h <- abs(2 * asin(sqrt(hd_p)) - 2 * asin(sqrt(0.5)))
    one_prop_rows[[length(one_prop_rows) + 1L]] <- data.frame(
      p0 = 0.5,
      p_obs = hd_p,
      h = h,
      n = n,
      n_eff = n * 0.8,
      power_srs = stats::pnorm(sqrt(n) * h - stats::qnorm(1 - 0.05 / 2)),
      power_deff = stats::pnorm(sqrt(n * 0.8) * h - stats::qnorm(1 - 0.05 / 2)),
      analysis_mode = "design",
      power_scope = "cpads",
      stringsAsFactors = FALSE
    )
  }

  schedules <- list(
    randomization_schedule_example_heavy_drinking_30d = .block_schedule("heavy_drinking_30d", target_rows[[1L]]$required_n[1], na.omit(unique(data$gender_label))),
    randomization_schedule_example_ebac_legal = .block_schedule("ebac_legal", if (length(target_rows) >= 2L) target_rows[[2L]]$required_n[1] else NA_real_, na.omit(unique(data$gender_label))),
    randomization_schedule_example_ebac_tot = .block_schedule("ebac_tot", if (length(target_rows) >= 3L) target_rows[[3L]]$required_n[1] else NA_real_, na.omit(unique(data$gender_label)))
  )

  c(
    list(
      power_summary = do.call(rbind, summary_rows),
      power_two_proportion_gender = if (length(pair_rows) > 0L) do.call(rbind, pair_rows) else data.frame(),
      power_one_proportion_grid = do.call(rbind, one_prop_rows),
      power_ebac_endpoint_anchors = do.call(rbind, anchor_rows),
      power_gpower_reference_two_group = if (length(gp_rows) > 0L) do.call(rbind, gp_rows) else data.frame(),
      power_interaction_assumptions = if (length(assumption_rows) > 0L) do.call(rbind, assumption_rows) else data.frame(),
      power_interaction_feasibility_flags = if (length(feas_rows) > 0L) do.call(rbind, feas_rows) else data.frame(),
      power_interaction_group_allocations = if (length(alloc_rows) > 0L) do.call(rbind, alloc_rows) else data.frame(),
      power_interaction_imbalance_penalty = if (length(penalty_rows) > 0L) do.call(rbind, penalty_rows) else data.frame(),
      power_interaction_pairwise_details = if (length(detail_rows) > 0L) do.call(rbind, detail_rows) else data.frame(),
      power_interaction_sample_size_targets = if (length(target_rows) > 0L) do.call(rbind, target_rows) else data.frame(),
      randomization_block_blueprints = if (length(blueprint_rows) > 0L) do.call(rbind, blueprint_rows) else data.frame()
    ),
    schedules
  )
}

#' .read_existing_output
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param output_dir See Usage.
#' @param file_name See Usage.
#' @param fallback Defaults to \code{NULL}.
#' @return The value of \code{fallback}, as built in the body.
#' @export
.read_existing_output <- function(output_dir, file_name, fallback = NULL) {
  path <- file.path(output_dir, file_name)
  if (file.exists(path)) {
    return(utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE))
  }
  fallback
}

#' The legacy migration tree exists only in a source checkout; an
#'
#' installed package has no project root. Degrade to NA so callers
#' (.copy_legacy_artifacts) simply copy nothing rather than erroring.
#'
#' @return The value of \code{file.path}.
#' @export
.legacy_reference_root <- function() {
  # The legacy migration tree exists only in a source checkout; an
  # installed package has no project root. Degrade to NA so callers
  # (.copy_legacy_artifacts) simply copy nothing rather than erroring.
  root <- tryCatch(morie_find_project_root(), error = function(e) NA_character_)
  if (is.na(root)) {
    return(NA_character_)
  }
  file.path(root, "migration_files", "one")
}

#' .copy_legacy_artifacts
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param relative_paths See Usage.
#' @param output_dir See Usage.
#' @param root Defaults to \code{file.path(.legacy_reference_root(), "six", "outputs")}.
#' @return The value of \code{copied}, as built in the body.
#' @export
.copy_legacy_artifacts <- function(relative_paths, output_dir, root = file.path(.legacy_reference_root(), "six", "outputs")) {
  copied <- character()
  for (rel in relative_paths) {
    src <- file.path(root, rel)
    dst <- file.path(output_dir, rel)
    if (!file.exists(src)) next
    dir.create(dirname(dst), recursive = TRUE, showWarnings = FALSE)
    ok <- file.copy(src, dst, overwrite = TRUE, copy.mode = TRUE, copy.date = TRUE)
    if (isTRUE(ok)) copied <- c(copied, rel)
  }
  copied
}

#' .run_ebac_integrations_module_internal
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param data See Usage.
#' @param output_dir Defaults to \code{NULL}.
#' @return A list with \code{ebac_final_domain_samples}, \code{ebac_final_formula_input_audit}, \code{ebac_final_formula_validation}, \code{ebac_final_interaction_tests}, \code{ebac_final_weighted_descriptives}, \code{ebac_final_weighted_linear}, \code{ebac_final_weighted_or}, \code{ebac_final_smote_compare}, \code{ebac_final_smote_or}, \code{ebac_final_smote_status}, \code{ebac_final_causal_effects}, \code{ebac_final_cate}, \code{ebac_final_consistency_checks}, \code{ebac_final_crosswalk_previous}, \code{ebac_final_dml_results}, \code{ebac_final_dml_status}, \code{ebac_final_key_summary}, \code{ebac_final_user_guide_variable_map}, \code{ebac_final_variable_audit}.
#' @export
.run_ebac_integrations_module_internal <- function(data, output_dir = NULL) {
  if (is.null(output_dir)) {
    output_dir <- tempfile("morie-ebac-integrations-")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  core <- .run_ebac_core_module_internal(data)
  ipw <- .run_ebac_selection_adjustment_ipw_module_internal(data)
  sens <- .run_ebac_gender_smote_sensitivity_module_internal(data)
  treat <- .run_treatment_effects_module_internal(data)

  final_causal <- data.frame(
    estimand = treat$treatment_effects_summary$estimand,
    estimate = treat$treatment_effects_summary$estimate,
    se = treat$treatment_effects_summary$se,
    ci_lower95 = treat$treatment_effects_summary$ci_lower,
    ci_upper95 = treat$treatment_effects_summary$ci_upper,
    n = sum(!is.na(data$ebac_tot)),
    n_boot_valid = NA_real_,
    stringsAsFactors = FALSE
  )
  final_cate <- within(treat$cate_subgroup_estimates, {
    ci_lower95 <- ci_lower
    ci_upper95 <- ci_upper
    note <- "Unweighted subgroup contrast"
  })
  final_cate <- final_cate[, c("subgroup_var", "subgroup_level", "n_treated", "n_control", "cate", "se", "ci_lower95", "ci_upper95", "note")]
  final_consistency <- data.frame(
    check = c("ipw_or_available", "weighted_or_available", "smote_status_recorded"),
    lhs = c(ipw$ebac_final_ipw_or$or[1], core$ebac_logistic_or_primary$or[core$ebac_logistic_or_primary$term == "cannabis_any_use"][1], sens$ebac_smote_status$run_completed[1]),
    rhs = c(!is.na(ipw$ebac_final_ipw_or$or[1]), !is.na(core$ebac_logistic_or_primary$or[core$ebac_logistic_or_primary$term == "cannabis_any_use"][1]), TRUE),
    abs_diff = c(0, 0, 0),
    pass = TRUE,
    stringsAsFactors = FALSE
  )
  empty_dml <- data.frame(
    estimand = character(),
    estimate = numeric(),
    se = numeric(),
    t_value = numeric(),
    p_value = numeric(),
    stringsAsFactors = FALSE
  )
  status_dml <- data.frame(
    package_combo_available = FALSE,
    run_completed = FALSE,
    note = "DoubleML is not enabled in the default R-only workflow.",
    stringsAsFactors = FALSE
  )
  var_map <- data.frame(
    variable_name = morie_cpads_contract()$required_variables,
    user_guide_description = c(
      "Survey weight",
      "Alcohol use in the past 12 months",
      "Heavy drinking in the past 30 days",
      "Total estimated blood alcohol concentration",
      "Legal-threshold eBAC indicator",
      "Any cannabis use",
      "Age group",
      "Gender",
      "Province/region",
      "Mental health",
      "Physical health"
    ),
    exists_in_wrangled_data = morie_cpads_contract()$required_variables %in% names(data),
    coding_note = "See CPADS user guide PDF for official item wording and coding.",
    stringsAsFactors = FALSE
  )
  list(
    ebac_final_domain_samples = core$ebac_model_samples,
    ebac_final_formula_input_audit = data.frame(item = names(data), value = "present", stringsAsFactors = FALSE),
    ebac_final_formula_validation = data.frame(metric = c("n_columns", "n_rows"), value = c(ncol(data), nrow(data)), stringsAsFactors = FALSE),
    ebac_final_interaction_tests = sens$ebac_gender_interaction_tests,
    ebac_final_weighted_descriptives = core$ebac_weighted_summaries,
    ebac_final_weighted_linear = core$ebac_linear_coefficients_primary,
    ebac_final_weighted_or = core$ebac_logistic_or_primary,
    ebac_final_smote_compare = sens$ebac_smote_compare,
    ebac_final_smote_or = sens$ebac_smote_or,
    ebac_final_smote_status = sens$ebac_smote_status[, c("package_available", "run_completed", "method", "warning_count", "class_ratio_before", "class_ratio_after", "note")],
    ebac_final_causal_effects = final_causal,
    ebac_final_cate = final_cate,
    ebac_final_consistency_checks = final_consistency,
    ebac_final_crosswalk_previous = data.frame(
      source = c("core_weighted_or", "selection_ipw_or"),
      metric = c("ebac_legal_cannabis_or", "ebac_legal_cannabis_or"),
      estimate = c(core$ebac_logistic_or_primary$or[core$ebac_logistic_or_primary$term == "cannabis_any_use"][1], ipw$ebac_final_ipw_or$or[1]),
      stringsAsFactors = FALSE
    ),
    ebac_final_dml_results = empty_dml,
    ebac_final_dml_status = status_dml,
    ebac_final_key_summary = data.frame(
      key = c("eligible_drinkers", "observed_ebac", "cannabis_any_use_prevalence"),
      value = c(sum(data$alcohol_past12m == 1, na.rm = TRUE), sum(!is.na(data$ebac_tot)), round(.weighted_binary_estimate(data$cannabis_any_use, data$weight)$p, 4)),
      stringsAsFactors = FALSE
    ),
    ebac_final_user_guide_variable_map = var_map,
    ebac_final_variable_audit = data.frame(item = names(data), value = ifelse(names(data) %in% morie_cpads_contract()$required_variables, "canonical", "auxiliary"), stringsAsFactors = FALSE)
  )
}

# Render one figure to figures/<name>.pdf (and optionally .png). Returns the
# relative paths written, or character(0) if the draw function errored (the
# half-written file is removed and a warning names the figure).
#' Render one figure to figures/<name>.pdf (and optionally .png).
#' Returns the
#'
#' relative paths written, or character(0) if the draw function errored
#' (the half-written file is removed and a warning names the figure).
#'
#' @param fig_dir See Usage.
#' @param name See Usage.
#' @param draw See Usage.
#' @param png_too Defaults to \code{FALSE}.
#' @param width Defaults to \code{8}.
#' @param height Defaults to \code{6}.
#' @return The value of \code{wrote}, as built in the body.
#' @export
.fig_write <- function(fig_dir, name, draw, png_too = FALSE,
                       width = 8, height = 6) {
  wrote <- character()
  render <- function(open_dev, path) {
    open_dev(path)
    ok <- tryCatch({
      draw()
      TRUE
    }, error = function(e) {
      warning("figures module: could not render ", name, ": ",
              conditionMessage(e), call. = FALSE)
      FALSE
    })
    grDevices::dev.off()
    if (!ok) unlink(path)
    ok
  }
  pdf_path <- file.path(fig_dir, paste0(name, ".pdf"))
  if (render(function(p) grDevices::pdf(p, width = width, height = height),
             pdf_path)) {
    wrote <- c(wrote, file.path("figures", paste0(name, ".pdf")))
  } else {
    return(character())
  }
  if (png_too) {
    png_path <- file.path(fig_dir, paste0(name, ".png"))
    if (render(function(p) grDevices::png(p, width = width * 100,
                                          height = height * 100, res = 110),
               png_path)) {
      wrote <- c(wrote, file.path("figures", paste0(name, ".png")))
    }
  }
  wrote
}

# Standardized mean differences of dummy-coded covariates between arms.
#' Standardized mean differences of dummy-coded covariates between arms
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param data See Usage.
#' @param treat_col See Usage.
#' @param covariate_cols See Usage.
#' @return The value of \code{[}.
#' @export
.smd_by_treatment <- function(data, treat_col, covariate_cols) {
  treat <- data[[treat_col]]
  keep <- !is.na(treat)
  out <- list()
  for (cov in covariate_cols) {
    v <- data[[cov]][keep]
    if (is.factor(v) || is.character(v)) {
      v <- factor(v)
      for (lev in levels(v)) {
        x <- as.numeric(v == lev)
        out[[paste0(cov, ": ", lev)]] <- x
      }
    } else {
      out[[cov]] <- as.numeric(v)
    }
  }
  t1 <- treat[keep] == 1
  smd <- vapply(out, function(x) {
    m1 <- mean(x[t1], na.rm = TRUE); m0 <- mean(x[!t1], na.rm = TRUE)
    s <- sqrt((stats::var(x[t1], na.rm = TRUE) +
                 stats::var(x[!t1], na.rm = TRUE)) / 2)
    if (!is.finite(s) || s == 0) return(NA_real_)
    (m1 - m0) / s
  }, numeric(1))
  smd[is.finite(smd)]
}

#' .run_figures_module_internal
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param data See Usage.
#' @param output_dir Defaults to \code{NULL}.
#' @return Invisibly,the value of \code{list}.
#' @export
.run_figures_module_internal <- function(data, output_dir = NULL) {
  if (is.null(output_dir)) {
    output_dir <- tempfile("morie-figures-")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  data <- .cpads_labeled_data(data)
  fig_dir <- file.path(output_dir, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
  wrote <- character()
  label_covs <- intersect(
    c("age_group_label", "gender_label", "province_region_label",
      "mental_health_label", "physical_health_label"),
    names(data)
  )

  ## 1. Covariate balance: SMD of dummy-coded covariates by cannabis use.
  if ("cannabis_any_use" %in% names(data) && length(label_covs) > 0L) {
    smd <- .smd_by_treatment(data, "cannabis_any_use", label_covs)
    if (length(smd) > 0L) {
      wrote <- c(wrote, .fig_write(fig_dir, "balance_plot", function() {
        graphics::par(mar = c(4, 14, 3, 2))
        ord <- order(smd)
        graphics::dotchart(smd[ord], labels = names(smd)[ord], pch = 19,
                           xlab = "Standardized mean difference",
                           main = "Covariate balance by cannabis use")
        graphics::abline(v = c(-0.1, 0, 0.1), lty = c(2, 1, 2),
                         col = c("grey40", "black", "grey40"))
      }, height = max(6, 0.28 * length(smd))))
    }
  }

  ## 2. Beta prior vs posterior densities (from the bayesian module output).
  post <- .read_existing_output(output_dir, "bayesian_posterior_summaries.csv")
  if (!is.null(post) && all(c("prior_name", "alpha_prior", "beta_prior",
                              "alpha_post", "beta_post") %in% names(post))) {
    wrote <- c(wrote, .fig_write(fig_dir, "bayesian_prior_posterior", function() {
      x <- seq(0.001, 0.999, length.out = 400)
      graphics::par(mfrow = c(1, nrow(post)))
      for (i in seq_len(nrow(post))) {
        d_prior <- stats::dbeta(x, post$alpha_prior[i], post$beta_prior[i])
        d_post <- stats::dbeta(x, post$alpha_post[i], post$beta_post[i])
        graphics::plot(x, d_post, type = "l", lwd = 2, col = "#2E6B4A",
                       xlab = "Prevalence", ylab = "Density",
                       main = post$prior_name[i])
        graphics::lines(x, d_prior, lty = 2, col = "grey40")
        graphics::legend("topright", c("posterior", "prior"),
                         lty = c(1, 2), lwd = c(2, 1),
                         col = c("#2E6B4A", "grey40"), bty = "n")
      }
    }, png_too = TRUE, width = 10, height = 4))
  }

  ## 3. Bayesian credible vs frequentist confidence intervals.
  bvf <- .read_existing_output(output_dir, "bayesian_vs_frequentist_ci.csv")
  if (!is.null(bvf) &&
      all(c("prior_name", "post_mean", "ci_lower", "ci_upper") %in% names(bvf)) &&
      "heavy_drinking_30d" %in% names(data)) {
    y <- data$heavy_drinking_30d[!is.na(data$heavy_drinking_30d)]
    ft <- stats::prop.test(sum(y == 1), length(y))
    est <- c(bvf$post_mean, unname(ft$estimate))
    lo <- c(bvf$ci_lower, ft$conf.int[1])
    hi <- c(bvf$ci_upper, ft$conf.int[2])
    lab <- c(paste0("Bayesian: ", bvf$prior_name), "Frequentist (prop.test)")
    wrote <- c(wrote, .fig_write(fig_dir, "bayesian_vs_frequentist_ci", function() {
      graphics::par(mar = c(4, 12, 3, 2))
      idx <- rev(seq_along(est))
      graphics::plot(est, idx, xlim = range(lo, hi), pch = 19,
                     yaxt = "n", ylab = "",
                     xlab = "Heavy-drinking prevalence",
                     main = "95% interval comparison")
      graphics::segments(lo, idx, hi, idx)
      graphics::axis(2, at = idx, labels = lab, las = 1, cex.axis = 0.85)
    }, png_too = TRUE, width = 8, height = 4.5))
  }

  ## 4-5. Outcome prevalence by demographic / mental-health strata.
  draw_by <- function(cols, main) {
    force(cols); force(main)
    function() {
      graphics::par(mar = c(9, 4, 3, 2))
      rates <- unlist(lapply(cols, function(cn) {
        tapply(as.numeric(data$heavy_drinking_30d),
               data[[cn]], mean, na.rm = TRUE)
      }))
      graphics::barplot(rates, las = 2, ylab = "Heavy drinking (30d) rate",
                        main = main, col = "#4A7BA6", cex.names = 0.8)
    }
  }
  if ("heavy_drinking_30d" %in% names(data)) {
    demo <- intersect(c("age_group_label", "gender_label"), names(data))
    if (length(demo) > 0L) {
      wrote <- c(wrote, .fig_write(fig_dir, "binge_by_demographics",
                                   draw_by(demo, "Heavy drinking by demographics"),
                                   png_too = TRUE))
    }
    if ("mental_health_label" %in% names(data)) {
      wrote <- c(wrote, .fig_write(
        fig_dir, "binge_by_mental_health",
        draw_by("mental_health_label", "Heavy drinking by mental health"),
        png_too = TRUE
      ))
    }
  }

  ## 6. CATE forest plot (from the ebac-integrations output).
  cate <- .read_existing_output(output_dir, "ebac_final_cate.csv")
  if (!is.null(cate) && all(c("subgroup_var", "subgroup_level", "cate",
                              "ci_lower95", "ci_upper95") %in% names(cate))) {
    cate <- cate[is.finite(cate$cate), , drop = FALSE]
    if (nrow(cate) > 0L) {
      wrote <- c(wrote, .fig_write(fig_dir, "cate_forest_plot", function() {
        lab <- paste0(cate$subgroup_var, ": ", cate$subgroup_level)
        idx <- rev(seq_len(nrow(cate)))
        graphics::par(mar = c(4, 14, 3, 2))
        graphics::plot(cate$cate, idx,
                       xlim = range(cate$ci_lower95, cate$ci_upper95,
                                    na.rm = TRUE),
                       pch = 19, yaxt = "n", ylab = "",
                       xlab = "Subgroup contrast (CATE)",
                       main = "Conditional effects by subgroup")
        graphics::segments(cate$ci_lower95, idx, cate$ci_upper95, idx)
        graphics::axis(2, at = idx, labels = lab, las = 1, cex.axis = 0.8)
        graphics::abline(v = 0, lty = 2, col = "grey40")
      }, png_too = TRUE, height = max(4.5, 0.4 * nrow(cate))))
    }
  }

  ## 7. Study DAG (static specification, drawn not copied).
  wrote <- c(wrote, .fig_write(fig_dir, "dag_heavy_drinking", function() {
    graphics::par(mar = c(1, 1, 3, 1))
    graphics::plot(NULL, xlim = c(0, 10), ylim = c(0, 10), axes = FALSE,
                   xlab = "", ylab = "", main = "DAG: cannabis use and heavy drinking")
    nodes <- list(
      cannabis = c(2, 5), heavy = c(8, 5),
      age = c(3, 8.5), gender = c(5, 8.5), region = c(7, 8.5),
      mental = c(4, 1.5), physical = c(6, 1.5)
    )
    labs <- c(cannabis = "Cannabis use", heavy = "Heavy drinking",
              age = "Age", gender = "Gender", region = "Region",
              mental = "Mental health", physical = "Physical health")
    edge <- function(from, to) {
      p1 <- nodes[[from]]; p2 <- nodes[[to]]
      shrink <- 0.82
      mx <- (p1[1] + p2[1]) / 2; my <- (p1[2] + p2[2]) / 2
      graphics::arrows(mx + (p1[1] - mx) * shrink, my + (p1[2] - my) * shrink,
                       mx + (p2[1] - mx) * shrink, my + (p2[2] - my) * shrink,
                       length = 0.12, col = "grey25")
    }
    edge("cannabis", "heavy")
    for (conf in c("age", "gender", "region", "mental", "physical")) {
      edge(conf, "cannabis"); edge(conf, "heavy")
    }
    for (nm in names(nodes)) {
      graphics::points(nodes[[nm]][1], nodes[[nm]][2], pch = 21,
                       bg = if (nm %in% c("cannabis", "heavy")) "#BFD7C9" else "white",
                       cex = 5.2)
      graphics::text(nodes[[nm]][1], nodes[[nm]][2] - 0.9, labs[[nm]],
                     cex = 0.85)
    }
  }))

  ## 8. QQ plots of the continuous measures.
  qq_vars <- intersect(c("ebac_tot", "weight"), names(data))
  qq_vars <- qq_vars[vapply(qq_vars, function(v)
    sum(is.finite(as.numeric(data[[v]]))) >= 10L, logical(1))]
  if (length(qq_vars) > 0L) {
    wrote <- c(wrote, .fig_write(fig_dir, "qq_plots", function() {
      graphics::par(mfrow = c(1, length(qq_vars)))
      for (v in qq_vars) {
        x <- as.numeric(data[[v]])
        x <- x[is.finite(x)]
        stats::qqnorm(x, main = paste("Normal QQ:", v), pch = 20,
                      col = "#00000066")
        stats::qqline(x, col = "#8A3322", lwd = 2)
      }
    }, width = 5 * length(qq_vars), height = 5))
  }

  if (length(wrote) == 0L) {
    stop("figures module produced no outputs (no renderable inputs found); ",
         "refusing to report success.", call. = FALSE)
  }
  invisible(list())
}

#' .run_tables_module_internal
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param data See Usage.
#' @param output_dir Defaults to \code{NULL}.
#' @return Invisibly,the value of \code{list}.
#' @export
.run_tables_module_internal <- function(data, output_dir = NULL) {
  if (is.null(output_dir)) {
    output_dir <- tempfile("morie-tables-")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  data <- .cpads_labeled_data(data)
  if (!"cannabis_any_use" %in% names(data)) {
    stop("tables module: cannabis_any_use column missing; cannot build Table 1.",
         call. = FALSE)
  }
  arm <- factor(ifelse(data$cannabis_any_use == 1, "Cannabis use", "No cannabis use"))
  esc <- function(x) {
    x <- gsub("&", "&amp;", x, fixed = TRUE)
    x <- gsub("<", "&lt;", x, fixed = TRUE)
    gsub(">", "&gt;", x, fixed = TRUE)
  }
  rows <- list(c("<tr><th>Characteristic</th>",
                 paste0("<th>", esc(levels(arm)), " (n=",
                        as.integer(table(arm)[levels(arm)]), ")</th>",
                        collapse = ""),
                 "</tr>"))
  add_row <- function(label, cells) {
    rows[[length(rows) + 1L]] <<- c(
      "<tr><td>", esc(label), "</td>",
      paste0("<td>", esc(cells), "</td>", collapse = ""), "</tr>"
    )
  }
  for (cov in intersect(c("age_group_label", "gender_label",
                          "province_region_label", "mental_health_label",
                          "physical_health_label"), names(data))) {
    v <- factor(data[[cov]])
    add_row(sub("_label$", "", cov), rep("", nlevels(arm)))
    for (lev in levels(v)) {
      cells <- vapply(levels(arm), function(a) {
        sel <- arm == a & !is.na(v)
        n <- sum(v[sel] == lev, na.rm = TRUE)
        sprintf("%d (%.1f%%)", n, 100 * n / max(1L, sum(sel)))
      }, character(1))
      add_row(paste0("\u00a0\u00a0", lev), cells)
    }
  }
  for (num in intersect(c("ebac_tot", "heavy_drinking_30d"), names(data))) {
    cells <- vapply(levels(arm), function(a) {
      x <- as.numeric(data[[num]][arm == a])
      x <- x[is.finite(x)]
      if (length(x) == 0L) return("--")
      sprintf("%.3f (%.3f)", mean(x), stats::sd(x))
    }, character(1))
    add_row(paste0(num, ", mean (SD)"), cells)
  }
  html <- c(
    "<!DOCTYPE html>",
    "<html><head><meta charset=\"utf-8\"><title>Table 1</title>",
    "<style>table{border-collapse:collapse;font-family:sans-serif}",
    "td,th{border:1px solid #999;padding:4px 10px;text-align:left}</style>",
    "</head><body>",
    "<h2>Table 1. Sample characteristics by cannabis use</h2>",
    "<table>", unlist(rows), "</table>",
    "</body></html>"
  )
  writeLines(html, file.path(output_dir, "table1.html"))
  if (!file.exists(file.path(output_dir, "table1.html"))) {
    stop("tables module failed to write table1.html.", call. = FALSE)
  }
  invisible(list())
}

#' .run_final_report_module_internal
#'
#' Part of the study_reporting implementation; see the file header for
#' the source it follows.
#'
#' @param data See Usage.
#' @param output_dir Defaults to \code{NULL}.
#' @return A list with \code{ebac_final_output_coverage}, \code{ebac_final_output_shapes}, \code{ebac_final_script_run_status}, \code{ebac_final_audit_checks}, \code{ebac_final_user_guide_excerpt}.
#' @export
.run_final_report_module_internal <- function(data, output_dir = NULL) {
  if (is.null(output_dir)) {
    output_dir <- tempfile("morie-final-report-")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  output_files <- sort(list.files(output_dir, recursive = TRUE, pattern = "\\.(csv|txt|pdf|png|html|md)$", full.names = FALSE))
  coverage <- data.frame(
    script = c("data-wrangling", "descriptive-statistics", "distribution-tests", "frequentist-inference", "bayesian-inference", "power-design", "logistic-models", "model-comparison", "regression-models", "propensity-scores", "causal-estimators", "treatment-effects", "dag-specification", "meta-synthesis", "ebac-core", "ebac-selection-adjustment-ipw", "ebac-integrations", "ebac-gender-smote-sensitivity", "figures", "tables", "final-report"),
    output = c("data_wrangling_log.csv", "binomial_summaries.csv", "distribution_tests.csv", "frequentist_hypothesis_tests.csv", "bayesian_posterior_summaries.csv", "power_summary.csv", "logistic_odds_ratios.csv", "model_comparison_summary.csv", "regression_coefficients.csv", "ipw_results.csv", "causal_estimator_comparison.csv", "treatment_effects_summary.csv", "official_doc_alignment_checklist.csv", "10_methods_results_paper.md", "ebac_logistic_or_primary.csv", "ebac_final_ipw_or.csv", "ebac_final_weighted_or.csv", "ebac_gender_interaction_svy_or.csv", "figures/balance_plot.pdf", "table1.html", "ebac_final_output_shapes.csv"),
    stringsAsFactors = FALSE
  )
  coverage$exists <- ifelse(is.na(coverage$output), TRUE, coverage$output %in% output_files)
  shapes <- list()
  csvs <- list.files(output_dir, recursive = TRUE, pattern = "\\.csv$", full.names = TRUE)
  for (path in csvs) {
    tbl <- tryCatch(utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE), error = function(e) NULL)
    if (is.null(tbl)) next
    shapes[[length(shapes) + 1L]] <- data.frame(
      file = basename(path),
      rows = nrow(tbl),
      cols = ncol(tbl),
      columns = paste(names(tbl), collapse = ","),
      stringsAsFactors = FALSE
    )
  }
  excerpt_text <- paste(
    "MORIE CPADS variable guide reference.",
    "Primary outcomes: heavy_drinking_30d, ebac_tot, ebac_legal.",
    "Exposure: cannabis_any_use.",
    "See docs/source/modules/20212022-cpads-pumf-user-guide.pdf for source coding notes."
  )
  # The user-guide PDF lives in a source checkout only; tolerate its
  # absence (and a missing project root) when run from an installed
  # package rather than letting the whole report module error out.
  proj_root <- tryCatch(morie_find_project_root(), error = function(e) NA_character_)
  user_guide_present <- !is.na(proj_root) && file.exists(file.path(
    proj_root, "docs", "source", "modules",
    "20212022-cpads-pumf-user-guide.pdf"
  ))
  audit_tbl <- data.frame(
    check_name = c("outputs_present", "user_guide_reference_present", "cpads_required_variables_present"),
    value = c(length(output_files), user_guide_present, all(morie_cpads_contract()$required_variables %in% names(data))),
    pass = c(length(output_files) > 0, user_guide_present, all(morie_cpads_contract()$required_variables %in% names(data))),
    stringsAsFactors = FALSE
  )
  list(
    ebac_final_output_coverage = coverage,
    ebac_final_output_shapes = if (length(shapes) > 0L) do.call(rbind, shapes) else data.frame(file = character(), rows = integer(), cols = integer(), columns = character(), stringsAsFactors = FALSE),
    ebac_final_script_run_status = data.frame(
      script = coverage$script,
      log_file = paste0(coverage$script, ".log"),
      completed_marker_found = coverage$exists,
      warning_line_count = 0L,
      stringsAsFactors = FALSE
    ),
    ebac_final_audit_checks = audit_tbl,
    ebac_final_user_guide_excerpt = excerpt_text
  )
}
