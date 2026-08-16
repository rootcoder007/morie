% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.bca_interval}
\alias{.bca_interval}
\title{BCa (bias-corrected and accelerated) percentile interval}
\usage{
.bca_interval(data, statistic, boot_stats, original, ci_level)
}
\arguments{
\item{data}{Passed to \code{.nrow_like}.}
\item{statistic}{Accepted by the signature and not used anywhere in the body.}
\item{boot_stats}{Passed to \code{.pct}.}
\item{original}{Passed to \code{<}.}
\item{ci_level}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the bootstrap_methods implementation. Called by \code{bootstrap}. See the file header for the source the module follows.
}
\value{
A list with \code{ci_lo}, \code{ci_hi}, \code{acc}.
}
\keyword{internal}
