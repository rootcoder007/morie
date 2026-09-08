% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_normal_pdf_cpp}
\alias{morie_normal_pdf_cpp}
\title{morie_normal_pdf_cpp}
\usage{
morie_normal_pdf_cpp(x, mean, sd) 
}
\arguments{
\item{x}{Passed to \code{.Call}.}
\item{mean}{Passed to \code{.Call}.}
\item{sd}{Passed to \code{.Call}.}
}
\description{
Part of the RcppExports implementation; see the file header for the source it follows.
}
\value{
The value of \code{.Call}.
}
\keyword{internal}
\examples{
set.seed(20260516)
x <- rnorm(500)
y <- 0.6 * x + rnorm(500)
morie:::morie_mean_cpp(x)
morie:::morie_var_cpp(x, 1L)
morie:::morie_var_cpp(x, 0L)
morie:::morie_cor_pearson_cpp(x, y)
morie:::morie_normal_pdf_cpp(x, 0.3, 1.7)
}
