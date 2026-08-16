% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.discover_levels}
\alias{.discover_levels}
\title{Discover the data-adaptive levels (argmin and argmax of mean Q)}
\usage{
.discover_levels(ys, A_, W, levels, rows, eval_rows, ridge)
}
\arguments{
\item{ys}{Passed to \code{.fit_q_dta}.}
\item{A_}{Passed to \code{.fit_q_dta}.}
\item{W}{Passed to \code{.fit_q_dta}.}
\item{levels}{A vector; indexed elementwise.}
\item{rows}{Passed to \code{.fit_q_dta}.}
\item{eval_rows}{Iterated over elementwise, with \code{vapply}.}
\item{ridge}{Passed to \code{.fit_q_dta}.}
}
\description{
Part of the tmldta_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{aL}, \code{aH}, \code{info}.
}
\keyword{internal}
