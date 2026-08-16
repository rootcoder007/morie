% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrz_gauss_kernel}
\alias{.hrz_gauss_kernel}
\title{.hrz_gauss_kernel}
\usage{
.hrz_gauss_kernel(u)
}
\arguments{
\item{u}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the helpers_horowitz implementation. Called by \code{morie_kde_h}, \code{morie_kernel_quantile}, \code{morie_lewbel_binary} and 2 others in the module. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.hrz_gauss_kernel(V)
}
