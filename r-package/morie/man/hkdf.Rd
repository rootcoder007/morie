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
\examples{
ikm <- as.raw(rep(0x0b, 22))
salt <- as.raw(0:12)
info <- as.raw(0xf0:0xf9)
r <- hkdf(ikm, salt, info, 42L)
okm_hex <- paste(sprintf("\%02x", as.integer(r$okm)), collapse = "")
okm_hex == paste0("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db0",
                  "2d56ecc4c5bf34007208d5b887185865")
}
