% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sequential_blips}
\alias{.sequential_blips}
\title{Theorem 22.1: the two blips and the V-optimal rule}
\usage{
.sequential_blips(ys, L0, A0, L1, A1, V0, V1, ridge)
}
\arguments{
\item{ys}{A vector; its length is taken.}
\item{L0}{Passed to \code{.fit_q2}.}
\item{A0}{A vector; indexed elementwise.}
\item{L1}{Passed to \code{.fit_q2}.}
\item{A1}{Passed to \code{.fit_q2}.}
\item{V0}{See Usage.}
\item{V1}{See Usage.}
\item{ridge}{Passed to \code{.fit_q2}.}
}
\description{
Part of the tmldyn_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{blip1}, \code{blip2}, \code{d0}, \code{d1}, \code{q2}, \code{q1}, \code{coef_q2}, \code{coef_q1}, \code{pseudo}.
}
\keyword{internal}
