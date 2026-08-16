% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.split_specific_tmle}
\alias{.split_specific_tmle}
\title{Solve the per-split TMLE at fixed levels}
\usage{
.split_specific_tmle(ys, A_, W, levels, aL, aH, fit_rows, est_rows, ridge, trim, target)
}
\arguments{
\item{ys}{See Usage.}
\item{A_}{See Usage.}
\item{W}{See Usage.}
\item{levels}{See Usage.}
\item{aL}{See Usage.}
\item{aH}{See Usage.}
\item{fit_rows}{See Usage.}
\item{est_rows}{See Usage.}
\item{ridge}{See Usage.}
\item{trim}{See Usage.}
\item{target}{See Usage.}
}
\description{
Part of the tmldta_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{psi}, \code{D}, \code{info}.
}
\keyword{internal}
