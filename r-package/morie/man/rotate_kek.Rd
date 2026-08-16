% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{rotate_kek}
\alias{rotate_kek}
\title{Rotate the KEK: re-wrap every DEK, no record ciphertext touched}
\usage{
rotate_kek(wrapped_deks, old_kek, new_kek, new_nonces, new_kek_id = "kek-2", audit_log = NULL)
}
\description{
Rotate the KEK: re-wrap every DEK, no record ciphertext touched
}
\value{
A list with \code{estimate}, \code{wrapped}, \code{n}, \code{records_reencrypted}, \code{kek_id}, \code{method}, \code{note}.
}
\keyword{internal}
