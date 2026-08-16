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
\item{args}{Iterated over elementwise, with \code{lapply}.}
\item{n_boot}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{2000L}.}
\item{confidence}{Numeric; combined arithmetically in the body. Defaults to \code{0.95}.}
\item{seed}{Defaults to \code{42L}.}
}
\description{
A step of the effect_sizes implementation. Called by \code{bootstrap_effect_size_ci}, \code{cles}, \code{cliffs_delta} and 2 others in the module. See the file header for the source the module follows.
}
\value{
A list with \code{se}, \code{ci_lo}, \code{ci_hi}.
}
\keyword{internal}
