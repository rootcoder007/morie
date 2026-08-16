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
\value{
A list with \code{ciphertext}, \code{tag}, \code{nonce}, \code{aad}.
}
\keyword{internal}
