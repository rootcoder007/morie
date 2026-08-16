% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gp_mvn_logpdf}
\alias{.gp_mvn_logpdf}
\title{.gp_mvn_logpdf}
\usage{
.gp_mvn_logpdf(y, mu, cov)
}
\arguments{
\item{y}{A vector; its length is taken.}
\item{mu}{Numeric; combined arithmetically in the body.}
\item{cov}{Passed to \code{.gp_chol}.}
}
\description{
A step of the abcgp_native implementation. Called by \code{.gp_mw_sampler}, \code{synthetic_log_likelihood}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
