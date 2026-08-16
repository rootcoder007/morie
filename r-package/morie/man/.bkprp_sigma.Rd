% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.bkprp_sigma}
\alias{.bkprp_sigma}
\title{.bkprp_sigma}
\usage{
.bkprp_sigma(z, activation)
}
\arguments{
\item{z}{Numeric; combined arithmetically in the body.}
\item{activation}{Passed to \code{switch}.}
}
\description{
A step of the bkprp implementation. Called by \code{morie_bkprp_backpropagation}. See the file header for the source the module follows.
}
\value{
The value of \code{switch}.
}
\keyword{internal}
\examples{
.bkprp_sigma(z = c(1, 2, 3, 4, 5, 6, 7, 8), activation = 5L)
}
