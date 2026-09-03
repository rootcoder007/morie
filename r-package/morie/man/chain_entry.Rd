% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{chain_entry}
\alias{chain_entry}
\title{One step of the hash chain}
\usage{
chain_entry(previous_hash, entry, key = NULL) 
}
\description{
One step of the hash chain
}
\arguments{
\item{previous_hash}{Argument \code{previous_hash}; see Usage.}
\item{entry}{Argument \code{entry}; see Usage.}
\item{key}{Argument \code{key}; see Usage.}
}
\value{
A list with \code{hash}, \code{keyed}, \code{note}.
}
\keyword{internal}
\examples{
if (morie_crypto_sodium_available()) {
  e1 <- chain_entry(charToRaw("genesis"), "first record")
  c(e1$keyed, length(e1$hash) == 32L)
}
}
