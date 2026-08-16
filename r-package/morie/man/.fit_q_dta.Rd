% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fit_q_dta}
\alias{.fit_q_dta}
\title{Q surface for the data-adaptive target parameter}
\usage{
.fit_q_dta(ys, A_, W, levels, rows, ridge)
}
\arguments{
\item{ys}{A vector; indexed elementwise.}
\item{A_}{A vector; indexed elementwise.}
\item{W}{A matrix; indexed by row and column.}
\item{levels}{A vector; indexed elementwise.}
\item{rows}{See Usage.}
\item{ridge}{Numeric; passed to \code{max}.}
}
\description{
Part of the tmldta_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{q}, \code{b}.
}
\keyword{internal}
