% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.hrz_hermite}
\alias{.hrz_hermite}
\title{.hrz_hermite}
\usage{
.hrz_hermite(t, J)
}
\arguments{
\item{t}{A vector; its length is taken.}
\item{J}{A count; the body uses it as \code{matrix(...)}.}
}
\description{
A step of the helpers_horowitz implementation. Called by \code{hrzn1}. See the file header for the source the module follows.
}
\value{
The value of \code{H}, as built in the body.
}
\keyword{internal}
