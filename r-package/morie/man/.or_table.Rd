% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.or_table}
\alias{.or_table}
\title{.or_table}
\usage{
.or_table(fit, model = NULL, lower_se_name = FALSE)
}
\arguments{
\item{fit}{Passed to \code{.safe_confint}.}
\item{model}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{lower_se_name}{A flag; the body branches on it. Defaults to \code{FALSE}.}
}
\description{
Part of the study_core implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
