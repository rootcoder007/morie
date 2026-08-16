% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.intervention_mechanism}
\alias{.intervention_mechanism}
\title{Intervention mechanism g_{A(0)} and g_{A(1)}, with positivity trim}
\usage{
.intervention_mechanism(L0, A0, L1, A1, trim, known)
}
\arguments{
\item{L0}{A matrix; indexed by row and column.}
\item{A0}{A vector; its length is taken and its elements indexed.}
\item{L1}{A matrix; indexed by row and column.}
\item{A1}{Passed to \code{.tmldyn_logit_irls}.}
\item{trim}{Numeric; passed to \code{max}.}
\item{known}{Optional; may be \code{NULL}. A vector; indexed elementwise.}
}
\description{
Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{g0}, \code{g1}, \code{info}.
}
\keyword{internal}
