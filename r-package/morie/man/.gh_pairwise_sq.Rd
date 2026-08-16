% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gh_pairwise_sq}
\alias{.gh_pairwise_sq}
\title{.gh_pairwise_sq}
\usage{
.gh_pairwise_sq(a, b = a)
}
\arguments{
\item{a}{A matrix; passed to \code{\%*\%}.}
\item{b}{A matrix; passed to \code{t}. Defaults to \code{a}.}
}
\description{
A step of the helpers_ghosal_bnp implementation. Called by \code{morie_ghosal_gp_matern}, \code{morie_ghosal_gp_squared_exponential}, \code{morie_ghosal_np_classification}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
