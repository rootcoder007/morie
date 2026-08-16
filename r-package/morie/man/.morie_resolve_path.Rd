% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_resolve_path}
\alias{.morie_resolve_path}
\title{.morie_resolve_path}
\usage{
.morie_resolve_path(path)
}
\arguments{
\item{path}{Passed to \code{path.expand}.}
}
\description{
A step of the crypto_keystore implementation. Called by \code{.morie_read_store}, \code{.morie_write_store}, \code{morie_crypto_keystore_create}. See the file header for the source the module follows.
}
\value{
The value of \code{normalizePath}.
}
\keyword{internal}
\examples{
S <- c("a", "b", "c")
.morie_resolve_path(S)
}
