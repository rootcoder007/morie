% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrz_silverman}
\alias{.hrz_silverman}
\title{.hrz_silverman}
\usage{
.hrz_silverman(x)
}
\arguments{
\item{x}{A vector; its length is taken.}
}
\description{
A step of the helpers_horowitz implementation. Called by \code{hrzb2}, \code{Hrzctrl}, \code{hrzi1} and 24 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{unname}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.hrz_silverman(V)
}
