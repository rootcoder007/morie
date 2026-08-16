% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{hkdf}
\alias{hkdf}
\title{HKDF: extract-then-expand, or expand-only on uniform input}
\usage{
hkdf(ikm, salt = NULL, info = raw(0), length = 32L, skip_extract = FALSE)
}
\description{
HKDF: extract-then-expand, or expand-only on uniform input
}
\value{
A list with \code{estimate}, \code{okm}, \code{okm_hex}, \code{prk}, \code{prk_hex}, \code{length}, \code{blocks}, \code{salt_supplied}, \code{extract_skipped}, \code{method}, \code{note}.
}
\keyword{internal}
