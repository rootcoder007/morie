% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{andrews_monahan_hac}
\alias{andrews_monahan_hac}
\title{The full VAR prewhitened kernel HAC estimator, eq. 2.4}
\usage{
andrews_monahan_hac(e, X = NULL, prewhiten = TRUE, var_order = 1L, kernel = "qs", bandwidth = NULL, weights = NULL, n_params = NULL, cap = .EIGENVALUE_CAP, adjust = TRUE)
}
\description{
The full VAR prewhitened kernel HAC estimator, eq. 2.4
}
\value{
A list with \code{J}, \code{J_star}, \code{D}, \code{A}, \code{bandwidth}, \code{bandwidth_automatic}, \code{alpha}, \code{ar1_fits}, \code{kernel}, \code{var_order}, \code{n}, \code{p}, \code{n_params}, \code{prewhitened}, \code{method}, \code{note}.
}
\keyword{internal}
