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
\value{
A list with \code{sealed}, \code{records_reencrypted}, \code{note}.
}
\keyword{internal}
