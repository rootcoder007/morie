% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.run_ebac_integrations_module_internal}
\alias{.run_ebac_integrations_module_internal}
\title{.run_ebac_integrations_module_internal}
\usage{
.run_ebac_integrations_module_internal(data, output_dir = NULL)
}
\arguments{
\item{data}{A list; the body reads \code{$alcohol_past12m}, \code{$cannabis_any_use}, \code{$ebac_tot}, \code{$weight} from it.}
\item{output_dir}{Defaults to \code{NULL}.}
}
\description{
Part of the study_reporting implementation; see the file header for the source it follows.
}
\value{
A list with \code{ebac_final_domain_samples}, \code{ebac_final_formula_input_audit}, \code{ebac_final_formula_validation}, \code{ebac_final_interaction_tests}, \code{ebac_final_weighted_descriptives}, \code{ebac_final_weighted_linear}, \code{ebac_final_weighted_or}, \code{ebac_final_smote_compare}, \code{ebac_final_smote_or}, \code{ebac_final_smote_status}, \code{ebac_final_causal_effects}, \code{ebac_final_cate}, \code{ebac_final_consistency_checks}, \code{ebac_final_crosswalk_previous}, \code{ebac_final_dml_results}, \code{ebac_final_dml_status}, \code{ebac_final_key_summary}, \code{ebac_final_user_guide_variable_map}, \code{ebac_final_variable_audit}.
}
\keyword{internal}
