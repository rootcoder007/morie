% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{crypto_shred}
\alias{crypto_shred}
\title{Crypto-shred: destroy a KEK and report what that covers}
\usage{
crypto_shred(kek_id, wrapped_deks)
}
\description{
Crypto-shred: destroy a KEK and report what that covers
}
\value{
A list with \code{kek_id}, \code{records_shredded}, \code{indices}, \code{still_recoverable}, \code{complete}, \code{note}.
}
\keyword{internal}
