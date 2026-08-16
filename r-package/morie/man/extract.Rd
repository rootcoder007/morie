% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{extract}
\alias{extract}
\title{HKDF Extract: PRK = HMAC(salt, IKM)}
\usage{
extract(ikm, salt = NULL)
}
\description{
HKDF Extract: PRK = HMAC(salt, IKM)
}
\value{
A list with \code{prk}, \code{salt_supplied}, \code{note}.
}
\keyword{internal}
