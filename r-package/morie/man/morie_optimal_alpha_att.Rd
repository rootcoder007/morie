% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_optimal_alpha_att}
\alias{morie_optimal_alpha_att}
\title{Theorem 5.3: the one-sided threshold for the effect on the treated}
\usage{
morie_optimal_alpha_att(pscore, treated, tol = 1e-12, max_iter = 200) 
}
\arguments{
\item{pscore}{Numeric propensity scores.}
\item{treated}{Integer 0/1 treatment indicators.}
\item{tol,max_iter}{Convergence parameters.}
}
\description{
Homoskedastic, as in the paper.
}
\value{
A list with \code{alpha_t}, \code{keep}, \code{trim},   \code{no_trimming}.
}
\keyword{internal}
