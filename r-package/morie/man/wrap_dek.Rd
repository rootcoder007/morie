% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{wrap_dek}
\alias{wrap_dek}
\title{Wrap a DEK under a KEK}
\usage{
wrap_dek(dek, kek, nonce, kek_id = "kek-1", aad = raw(0)) 
}
\description{
Wrap a DEK under a KEK
}
\arguments{
\item{dek}{Argument \code{dek}; see Usage.}
\item{kek}{Argument \code{kek}; see Usage.}
\item{nonce}{Argument \code{nonce}; see Usage.}
\item{kek_id}{Argument \code{kek_id}; see Usage.}
\item{aad}{Argument \code{aad}; see Usage.}
}
\value{
A list with \code{wrapped}, \code{tag}, \code{nonce}, \code{kek_id}, \code{wrapped_hex}, \code{note}.
}
\keyword{internal}
\examples{
kek <- as.raw(1:32); nonce <- as.raw(1:12); dek <- as.raw(101:132)
w <- wrap_dek(dek, kek, nonce, kek_id = "kek-1")
c(length(w$wrapped), length(w$tag), w$kek_id)
}
