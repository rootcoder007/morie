% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hawkes_nll_cpp}
\alias{.hawkes_nll_cpp}
\title{.hawkes_nll_cpp}
\usage{
.hawkes_nll_cpp(theta, times, end_time, kernel)
}
\arguments{
\item{theta}{A vector; indexed elementwise.}
\item{times}{Passed to \code{morie_hawkes_ll_exp_const_cpp}.}
\item{end_time}{Passed to \code{morie_hawkes_ll_exp_const_cpp}.}
\item{kernel}{See Usage.}
}
\description{
A step of the hawkes_fit implementation. Called by \code{morie_hawkes_fit}. See the file header for the source the module follows.
}
\value{
The value of \code{switch}.
}
\keyword{internal}
