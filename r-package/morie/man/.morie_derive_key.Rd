% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_derive_key}
\alias{.morie_derive_key}
\title{.morie_derive_key}
\usage{
.morie_derive_key(password, salt)
}
\arguments{
\item{password}{A vector; its length is taken.}
\item{salt}{See Usage.}
}
\description{
A step of the crypto_keystore implementation. Called by \code{morie_crypto_keystore_create}, \code{morie_crypto_keystore_list}, \code{morie_crypto_keystore_load} and 1 others in the module. See the file header for the source the module follows.
}
\value{
Nothing; this branch always raises.
}
\keyword{internal}
