% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{derive_context_keys}
\alias{derive_context_keys}
\title{Derive one key per context, all independent}
\usage{
derive_context_keys(ikm, contexts, salt = NULL, length = 32L)
}
\description{
Derive one key per context, all independent
}
\value{
A list with \code{keys}, \code{hex}, \code{prk}, \code{all_distinct}, \code{note}.
}
\keyword{internal}
\examples{
ikm <- as.raw(rep(0x0b, 22))
keys <- derive_context_keys(ikm, contexts = c("app-encrypt", "app-mac"),
                            salt = as.raw(0:12), length = 32L)
c(length(keys), length(keys[["app-mac"]]) == 32L)
}
