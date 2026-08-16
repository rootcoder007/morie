% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{miprgr_enumerate_integer}
\alias{miprgr_enumerate_integer}
\title{miprgr_enumerate_integer}
\usage{
miprgr_enumerate_integer(A, b, c, integer_vars, upper = 10, maximise = TRUE)
}
\arguments{
\item{A}{A matrix; indexed by row and column.}
\item{b}{A vector; indexed elementwise.}
\item{c}{A vector; its length is taken.}
\item{integer_vars}{See Usage.}
\item{upper}{Defaults to \code{10}.}
\item{maximise}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the miprgr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{value}, \code{x}, \code{note}.
}
\references{
  Land, A. H. & Doig, A. G. (1960) Econometrica 28(3), 497-520.
  Dakin, R. J. (1965) The Computer Journal 8(3), 250-255.
Base R only.
}
\keyword{internal}
