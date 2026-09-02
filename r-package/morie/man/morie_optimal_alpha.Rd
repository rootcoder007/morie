% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_optimal_alpha}
\alias{morie_optimal_alpha}
\title{Theorem 5.2: the variance-minimising trimming threshold}
\usage{
morie_optimal_alpha(pscore, sigma2_treated = NULL, sigma2_control = NULL, tol = 1e-12, 
    max_iter = 200) 
}
\arguments{
\item{pscore}{Numeric vector of propensity scores in (0, 1).}
\item{sigma2_treated,sigma2_control}{Optional conditional variances.}
\item{tol,max_iter}{Convergence tolerance and iteration cap.}
}
\description{
Solves gamma = 2 E[k(X) | k(X) < gamma] for gamma by fixed point iteration. Homoskedastic unless conditional variances are supplied.
}
\value{
A list with \code{alpha}, \code{gamma}, \code{keep},   \code{trim}, \code{no_trimming}, \code{k}.
}
\keyword{internal}
