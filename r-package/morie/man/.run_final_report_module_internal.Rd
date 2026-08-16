% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.run_final_report_module_internal}
\alias{.run_final_report_module_internal}
\title{.run_final_report_module_internal}
\usage{
.run_final_report_module_internal(data, output_dir = NULL)
}
\arguments{
\item{data}{Passed to \code{names}.}
\item{output_dir}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
}
\description{
Part of the study_reporting implementation; see the file header for the source it follows.
}
\value{
A list with \code{ebac_final_output_coverage}, \code{ebac_final_output_shapes}, \code{ebac_final_script_run_status}, \code{ebac_final_audit_checks}, \code{ebac_final_user_guide_excerpt}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.run_final_report_module_internal(V)
}
