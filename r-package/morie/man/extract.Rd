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
\arguments{
\item{ikm}{Argument \code{ikm}; see Usage.}
\item{salt}{Argument \code{salt}; see Usage.}
}
\value{
A list with \code{prk}, \code{salt_supplied}, \code{note}.
}
\keyword{internal}
\examples{
ikm <- as.raw(rep(0x0b, 22))
salt <- as.raw(0:12)
e <- extract(ikm, salt)
prk_hex <- paste(sprintf("\%02x", as.integer(e$prk)), collapse = "")
prk_hex == "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
}
