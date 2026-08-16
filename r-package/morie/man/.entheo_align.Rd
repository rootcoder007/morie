% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.entheo_align}
\alias{.entheo_align}
\title{.entheo_align}
\usage{
.entheo_align(e, f)
}
\arguments{
\item{e}{A matrix; the body checks with \code{is.matrix}.}
\item{f}{A matrix; the body checks with \code{is.matrix}.}
}
\description{
A step of the entheo_analysis implementation. Called by \code{.entheo_binding_per_frame}, \code{.entheo_san_per_frame}. See the file header for the source the module follows.
}
\value{
A list with \code{e}, \code{f}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.entheo_align(V, V)
}
