% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.new_residual_diag}
\alias{.new_residual_diag}
\title{.new_residual_diag}
\usage{
.new_residual_diag(raw, std, student, deviance, pearson, fitted, normality, hetero, autoc, outlier_indices)
}
\arguments{
\item{raw}{See Usage.}
\item{std}{See Usage.}
\item{student}{See Usage.}
\item{deviance}{See Usage.}
\item{pearson}{See Usage.}
\item{fitted}{See Usage.}
\item{normality}{See Usage.}
\item{hetero}{See Usage.}
\item{autoc}{See Usage.}
\item{outlier_indices}{A vector; its length is taken.}
}
\description{
A step of the diagnostics implementation. Called by \code{compute_residuals}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
