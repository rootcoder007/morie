% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.seq_hamming}
\alias{.seq_hamming}
\title{.seq_hamming}
\usage{
.seq_hamming(a, b)
}
\arguments{
\item{a}{A vector; its length is taken.}
\item{b}{A vector; its length is taken.}
}
\description{
A step of the cluseq implementation. Called by \code{Cluseq}. See the file header for the source the module follows.
}
\value{
A numeric value.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.seq_hamming(V, V)
}
