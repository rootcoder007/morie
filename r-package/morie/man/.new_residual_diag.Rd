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
\item{raw}{Carried through into a list the body builds.}
\item{std}{Carried through into a list the body builds.}
\item{student}{Carried through into a list the body builds.}
\item{deviance}{Carried through into a list the body builds.}
\item{pearson}{Carried through into a list the body builds.}
\item{fitted}{Carried through into a list the body builds.}
\item{normality}{Carried through into a list the body builds.}
\item{hetero}{Carried through into a list the body builds.}
\item{autoc}{Carried through into a list the body builds.}
\item{outlier_indices}{A vector; its length is taken.}
}
\description{
A step of the diagnostics implementation. Called by \code{compute_residuals}. See the file header for the source the module follows.
}
\value{
The value of \code{structure}.
}
\keyword{internal}
