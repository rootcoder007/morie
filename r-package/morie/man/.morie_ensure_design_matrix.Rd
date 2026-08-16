% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_ensure_design_matrix}
\alias{.morie_ensure_design_matrix}
\title{This is invoked at the top of every morie ML-callable function so}
\usage{
.morie_ensure_design_matrix(x)
}
\arguments{
\item{x}{See Usage.}
}
\description{
users can pass a vector predictor without hitting glmnet\\\'s "x should be a matrix with 2 or more columns" error.
}
\value{
The value of \code{x}, as built in the body.
}
\keyword{internal}
