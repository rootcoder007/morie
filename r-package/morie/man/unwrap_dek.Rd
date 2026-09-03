% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{unwrap_dek}
\alias{unwrap_dek}
\title{Unwrap a DEK, optionally logging the call}
\usage{
unwrap_dek(wrapped, kek, audit_log = NULL) 
}
\description{
Unwrap a DEK, optionally logging the call
}
\arguments{
\item{wrapped}{Argument \code{wrapped}; see Usage.}
\item{kek}{Argument \code{kek}; see Usage.}
\item{audit_log}{Argument \code{audit_log}; see Usage.}
}
\value{
A list with \code{dek}, \code{kek_id}, \code{audited}.
}
\keyword{internal}
\examples{
if (morie_crypto_sodium_available()) {
  kek <- as.raw(1:32); nonce <- as.raw(1:12); dek <- as.raw(101:132)
  w <- wrap_dek(dek, kek, nonce)
  u <- unwrap_dek(w, kek)
  identical(as.raw(u$dek), dek)
}
}
