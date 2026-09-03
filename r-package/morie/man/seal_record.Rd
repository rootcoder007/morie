% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{seal_record}
\alias{seal_record}
\title{Seal a record under its DEK}
\usage{
seal_record(plaintext, dek, nonce, aad = raw(0)) 
}
\description{
Seal a record under its DEK
}
\arguments{
\item{plaintext}{Argument \code{plaintext}; see Usage.}
\item{dek}{Argument \code{dek}; see Usage.}
\item{nonce}{Argument \code{nonce}; see Usage.}
\item{aad}{Argument \code{aad}; see Usage.}
}
\value{
A list with \code{ciphertext}, \code{tag}, \code{nonce}, \code{aad}.
}
\keyword{internal}
\examples{
if (morie_crypto_sodium_available()) {
  dek <- as.raw(101:132); nonce <- as.raw(1:12)
  s <- seal_record(charToRaw("classified"), dek, nonce)
  length(s$tag) == 16L
}
}
