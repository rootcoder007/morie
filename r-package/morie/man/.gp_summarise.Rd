% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_summarise}
\alias{.gp_summarise}
\title{.gp_summarise}
\usage{
.gp_summarise(x, summary)
}
\arguments{
\item{x}{See Usage.}
\item{summary}{See Usage.}
}
\description{
A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler}, \code{gabc_log_likelihood}, \code{synthetic_log_likelihood}. See the file header for the source the module follows.
}
\value{
A vector, from \code{as.numeric}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.gp_summarise(V, V)
}
