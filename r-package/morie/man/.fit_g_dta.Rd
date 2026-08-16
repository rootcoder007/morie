% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fit_g_dta}
\alias{.fit_g_dta}
\title{Three-category treatment propensity, normalised}
\usage{
.fit_g_dta(A_, W, aL, aH, rows, ridge, trim)
}
\arguments{
\item{A_}{A vector; its length is taken.}
\item{W}{A vector; its length is taken.}
\item{aL}{Passed to \code{==}.}
\item{aH}{Passed to \code{==}.}
\item{rows}{See Usage.}
\item{ridge}{Numeric; passed to \code{max}.}
\item{trim}{Numeric; passed to \code{max}.}
}
\description{
Part of the tmldta_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{gH}, \code{gL}.
}
\keyword{internal}
