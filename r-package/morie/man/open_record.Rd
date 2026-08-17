% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{open_record}
\alias{open_record}
\title{Open a sealed record}
\usage{
open_record(sealed, dek)
}
\description{
Open a sealed record
}
\value{
The value of \code{$}.
}
\keyword{internal}
\examples{
dek <- as.raw(101:132); nonce <- as.raw(1:12)
s <- seal_record(charToRaw("classified"), dek, nonce)
rawToChar(open_record(s, dek))
}
