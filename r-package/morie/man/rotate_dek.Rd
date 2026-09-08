% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{rotate_dek}
\alias{rotate_dek}
\title{Rotate a DEK: re-encrypt one record}
\usage{
rotate_dek(sealed, old_dek, new_dek, new_nonce) 
}
\description{
Rotate a DEK: re-encrypt one record
}
\arguments{
\item{sealed}{Argument \code{sealed}; see Usage.}
\item{old_dek}{Argument \code{old_dek}; see Usage.}
\item{new_dek}{Argument \code{new_dek}; see Usage.}
\item{new_nonce}{Argument \code{new_nonce}; see Usage.}
}
\value{
A list with \code{sealed}, \code{records_reencrypted}, \code{note}.
}
\keyword{internal}
\examples{
if (morie_crypto_sodium_available()) {
  old_dek <- as.raw(101:132); new_dek <- as.raw(11:42); nonce <- as.raw(1:12)
  s <- seal_record(charToRaw("payload"), old_dek, nonce)
  rd <- rotate_dek(s, old_dek, new_dek, as.raw(71:82))
  rawToChar(open_record(rd$sealed, new_dek))
}
}
