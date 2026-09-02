% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{rotate_kek}
\alias{rotate_kek}
\title{Rotate the KEK: re-wrap every DEK, no record ciphertext touched}
\usage{
rotate_kek(wrapped_deks, old_kek, new_kek, new_nonces, new_kek_id = "kek-2", 
    audit_log = NULL) 
}
\description{
Rotate the KEK: re-wrap every DEK, no record ciphertext touched
}
\arguments{
\item{wrapped_deks}{Argument \code{wrapped_deks}; see Usage.}
\item{old_kek}{Argument \code{old_kek}; see Usage.}
\item{new_kek}{Argument \code{new_kek}; see Usage.}
\item{new_nonces}{Argument \code{new_nonces}; see Usage.}
\item{new_kek_id}{Argument \code{new_kek_id}; see Usage.}
\item{audit_log}{Argument \code{audit_log}; see Usage.}
}
\value{
A list with \code{estimate}, \code{wrapped}, \code{n}, \code{records_reencrypted}, \code{kek_id}, \code{method}, \code{note}.
}
\keyword{internal}
\examples{
kek <- as.raw(1:32); nonce <- as.raw(1:12); dek <- as.raw(101:132)
w <- wrap_dek(dek, kek, nonce)
new_kek <- as.raw(201:232)
rk <- rotate_kek(list(w), kek, new_kek, list(as.raw(51:62)))
u <- unwrap_dek(rk$wrapped[[1]], new_kek)
identical(as.raw(u$dek), dek)
}
