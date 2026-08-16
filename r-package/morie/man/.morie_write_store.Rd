% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_write_store}
\alias{.morie_write_store}
\title{.morie_write_store}
\usage{
.morie_write_store(data, path)
}
\arguments{
\item{data}{Passed to \code{.morie_to_json}.}
\item{path}{Passed to \code{.morie_resolve_path}.}
}
\description{
A step of the crypto_keystore implementation. Called by \code{morie_crypto_keystore_create}, \code{morie_crypto_keystore_store}. See the file header for the source the module follows.
}
\value{
Invisibly,nothing; the function is called for its effect.
}
\keyword{internal}
