% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_post_lasso}
\alias{morie_post_lasso}
\title{Refit by OLS on the lasso\'s selected support}
\usage{
morie_post_lasso(X, y, lam)
}
\arguments{
\item{X}{Numeric predictor matrix.}
\item{y}{Numeric response vector.}
\item{lam}{Lasso penalty used for the selection step.}
}
\description{
Refit by OLS on the lasso\'s selected support
}
\value{
A list with \code{support}, \code{coef}, \code{intercept},   \code{predict}, and the underlying lasso \code{beta}.
}
\keyword{internal}
\examples{
morie_post_lasso(X = c(1, 2, 3, 4, 5, 6, 7, 8), y = c(1, 2, 3, 4, 5, 6, 7, 8), lam = 5L)
}
