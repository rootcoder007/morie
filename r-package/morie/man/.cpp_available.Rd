% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.cpp_available}
\alias{.cpp_available}
\title{Internal: detect whether the Rcpp .so was successfully built}
\usage{
.cpp_available()
}
\description{
A step of the fast implementation. Called by \code{morie_cor_pearson}, \code{morie_fast_available}, \code{morie_hawkes_fit} and 3 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{tryCatch}.
}
\keyword{internal}
\examples{
.cpp_available()
}
