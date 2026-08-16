% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_aggregate_glm}
\alias{.otis_aggregate_glm}
\title{.otis_aggregate_glm}
\usage{
.otis_aggregate_glm(work, treatment, outcome, covariates = character(0), year_col = "EndFiscalYear", cluster_group = NULL, ds_id, source_label, title, interpretation)
}
\arguments{
\item{work}{A matrix; indexed by row and column.}
\item{treatment}{Carried through into a list the body builds.}
\item{outcome}{Carried through into a list the body builds.}
\item{covariates}{Passed to \code{\%||\%}. Defaults to \code{character(0)}.}
\item{year_col}{Carried through into a list the body builds. Defaults to \code{"EndFiscalYear"}.}
\item{cluster_group}{Optional; may be \code{NULL}. Carried through into a list the body builds.}
\item{ds_id}{Carried through into a list the body builds.}
\item{source_label}{Carried through into a list the body builds.}
\item{title}{Passed to \code{.otis_wrap}.}
\item{interpretation}{Passed to \code{paste0}.}
}
\description{
Part of the otis_all_analyze implementation; see the file header for the source it follows.
}
\value{
The value of \code{.otis_wrap}.
}
\keyword{internal}
