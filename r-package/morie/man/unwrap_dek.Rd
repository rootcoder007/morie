% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{unwrap_dek}
\alias{unwrap_dek}
\title{Unwrap a DEK, optionally logging the call}
\usage{
unwrap_dek(wrapped, kek, audit_log = NULL)
}
\description{
Unwrap a DEK, optionally logging the call
}
\value{
A list with \code{dek}, \code{kek_id}, \code{audited}.
}
\keyword{internal}
