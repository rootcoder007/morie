% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.gru4r_sigmoid}
\alias{.gru4r_sigmoid}
\title{Numerically stable sigmoid: below -700 it is effectively 0}
\usage{
.gru4r_sigmoid(x)
}
\arguments{
\item{x}{Numeric; combined arithmetically in the body.}
}
\description{
A step of the gru4r_native implementation. Called by \code{morie_gru4r_bpr}, \code{morie_gru4r_gru}, \code{morie_gru4r_top1}. See the file header for the source the module follows.
}
\value{
The value of \code{ifelse}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.gru4r_sigmoid(V)
}
