% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_hex_to_raw}
\alias{.morie_hex_to_raw}
\title{.morie_hex_to_raw}
\usage{
.morie_hex_to_raw(h)
}
\arguments{
\item{h}{A vector; its length is taken.}
}
\description{
A step of the crypto_keystore implementation. Called by \code{morie_crypto_keystore_list}, \code{morie_crypto_keystore_load}, \code{morie_crypto_keystore_store}. See the file header for the source the module follows.
}
\value{
The value of \code{as.raw}.
}
\keyword{internal}
