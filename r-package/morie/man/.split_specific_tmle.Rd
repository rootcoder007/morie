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
\item{ys}{A vector; its length is taken and its elements indexed.}
\item{A_}{A vector; indexed elementwise.}
\item{W}{Passed to \code{.fit_q_dta}.}
\item{levels}{Passed to \code{.fit_q_dta}.}
\item{aL}{Passed to \code{.fit_g_dta}.}
\item{aH}{Passed to \code{.fit_g_dta}.}
\item{fit_rows}{Passed to \code{.fit_q_dta}.}
\item{est_rows}{A vector; its length is taken.}
\item{ridge}{Passed to \code{.fit_q_dta}.}
\item{trim}{Passed to \code{.fit_g_dta}.}
\item{target}{A flag; the body branches on it.}
}
\description{
Part of the tmldta_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{psi}, \code{D}, \code{info}.
}
\keyword{internal}
