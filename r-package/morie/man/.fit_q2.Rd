% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fit_q2}
\alias{.fit_q2}
\title{Fit Q2 (E[Y | Abar(1), Lbar(1)]) on the rows in idx, with full}
\usage{
.fit_q2(ys, L0, A0, L1, A1, idx, ridge)
}
\arguments{
\item{ys}{See Usage.}
\item{L0}{See Usage.}
\item{A0}{See Usage.}
\item{L1}{See Usage.}
\item{A1}{See Usage.}
\item{idx}{See Usage.}
\item{ridge}{See Usage.}
}
\description{
treatment-by-covariate and treatment-by-treatment interactions.
}
\value{
A list with \code{q2}, \code{b}.
}
\keyword{internal}
