% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_cgnonl}
\alias{morie_cgnonl}
\title{morie_cgnonl}
\usage{
morie_cgnonl(f, grad, x0, beta = "fletcher-reeves", restart = NULL, max_iter = NULL, tol = 1e-10, est = NULL, line_search = "fletcher-reeves", hess_vec = NULL, keep_path = FALSE)
}
\arguments{
\item{f}{Passed to \code{cgnonl_nonlinear_cg}.}
\item{grad}{Passed to \code{cgnonl_nonlinear_cg}.}
\item{x0}{Passed to \code{cgnonl_nonlinear_cg}.}
\item{beta}{Passed to \code{cgnonl_nonlinear_cg}. Defaults to \code{"fletcher-reeves"}.}
\item{restart}{Passed to \code{cgnonl_nonlinear_cg}.}
\item{max_iter}{Passed to \code{cgnonl_nonlinear_cg}.}
\item{tol}{Passed to \code{cgnonl_nonlinear_cg}. Defaults to \code{1e-10}.}
\item{est}{Passed to \code{cgnonl_nonlinear_cg}.}
\item{line_search}{Passed to \code{cgnonl_nonlinear_cg}. Defaults to \code{"fletcher-reeves"}.}
\item{hess_vec}{Passed to \code{cgnonl_nonlinear_cg}.}
\item{keep_path}{Passed to \code{cgnonl_nonlinear_cg}. Defaults to \code{FALSE}.}
}
\description{
A step of the cgnonl_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
The value of \code{cgnonl_nonlinear_cg}.
}
\keyword{internal}
