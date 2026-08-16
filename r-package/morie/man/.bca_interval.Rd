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
\item{data}{See Usage.}
\item{statistic}{See Usage.}
\item{boot_stats}{See Usage.}
\item{original}{See Usage.}
\item{ci_level}{See Usage.}
}
\description{
Part of the bootstrap_methods implementation; see the file header for the source it follows.
}
\value{
A list with \code{ci_lo}, \code{ci_hi}, \code{acc}.
}
\keyword{internal}
