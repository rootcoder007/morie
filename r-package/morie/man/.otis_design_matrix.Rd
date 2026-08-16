% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.otis_design_matrix}
\alias{.otis_design_matrix}
\title{Build a numeric design matrix (intercept + drop-first dummies) from a}
\usage{
.otis_design_matrix(data, covariates)
}
\arguments{
\item{data}{See Usage.}
\item{covariates}{See Usage.}
}
\description{
data frame and a vector of covariate names. Mirrors python _design_matrix.
}
\value{
The value of \code{mf}, as built in the body.
}
\keyword{internal}
