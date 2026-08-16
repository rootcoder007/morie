% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.bootstrap_ci}
\alias{.bootstrap_ci}
\title{.bootstrap_ci}
\usage{
.bootstrap_ci(func, args, n_boot = 2000L, confidence = 0.95, seed = 42L)
}
\arguments{
\item{func}{See Usage.}
\item{args}{See Usage.}
\item{n_boot}{Defaults to \code{2000L}.}
\item{confidence}{Defaults to \code{0.95}.}
\item{seed}{Defaults to \code{42L}.}
}
\description{
Part of the effect_sizes implementation; see the file header for the source it follows.
}
\value{
A list with \code{se}, \code{ci_lo}, \code{ci_hi}.
}
\keyword{internal}
