% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_lasso_path}
\alias{morie_lasso_path}
\title{Coordinate-descent lasso path}
\usage{
morie_lasso_path(X, y, lam, iters = 500L, tol = 1e-09)
}
\arguments{
\item{X}{Numeric matrix of predictors (without intercept).}
\item{y}{Numeric response vector.}
\item{lam}{Non-negative penalty strength.}
\item{iters}{Maximum coordinate-descent iterations.}
}
\description{
Returns the L1-penalised least squares fit at a single \code{lambda}, with intercept and the selected support.
}
\value{
A list with \code{beta}, \code{intercept}, \code{support},   \code{lambda}.
}
\keyword{internal}
