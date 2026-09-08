% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{crypto_shred}
\alias{crypto_shred}
\title{Crypto-shred: destroy a KEK and report what that covers}
\usage{
crypto_shred(kek_id, wrapped_deks) 
}
\description{
Crypto-shred: destroy a KEK and report what that covers
}
\arguments{
\item{kek_id}{Argument \code{kek_id}; see Usage.}
\item{wrapped_deks}{Argument \code{wrapped_deks}; see Usage.}
}
\value{
A list with \code{kek_id}, \code{records_shredded}, \code{indices}, \code{still_recoverable}, \code{complete}, \code{note}.
}
\keyword{internal}
\examples{
if (morie_crypto_sodium_available()) {
  kek <- as.raw(1:32); nonce <- as.raw(1:12); dek <- as.raw(101:132)
  w <- wrap_dek(dek, kek, nonce, kek_id = "kek-1")
  sh <- crypto_shred("kek-1", list(w))
  sh$shredded
}
}
