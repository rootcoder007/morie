% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{expand}
\alias{expand}
\title{HKDF Expand: counter-mode OKM}
\usage{
expand(prk, info = raw(0), length = 32L) 
}
\description{
HKDF Expand: counter-mode OKM
}
\arguments{
\item{prk}{Argument \code{prk}; see Usage.}
\item{info}{Argument \code{info}; see Usage.}
\item{length}{Argument \code{length}; see Usage.}
}
\value{
A list with \code{okm}, \code{blocks}, \code{length}.
}
\keyword{internal}
\examples{
ikm <- as.raw(rep(0x0b, 22))
salt <- as.raw(0:12)
info <- as.raw(0xf0:0xf9)
prk <- extract(ikm, salt)$prk
r <- expand(prk, info, 42L)
okm_hex <- paste(sprintf("\%02x", as.integer(r$okm)), collapse = "")
okm_hex == paste0("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db0",
                  "2d56ecc4c5bf34007208d5b887185865")
}
