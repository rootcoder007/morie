% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_forensics_require_fbi_key}
\alias{.morie_forensics_require_fbi_key}
\title{Internal: resolve FBI CDE API key from arg -> env, or stop}
\usage{
.morie_forensics_require_fbi_key(api_key = NULL)
}
\arguments{
\item{api_key}{Defaults to \code{NULL}.}
}
\description{
A step of the ingest_forensics implementation. Called by \code{morie_ingest_forensics_nibrs}. See the file header for the source the module follows.
}
\value{
The value of \code{key}, as built in the body.
}
\keyword{internal}
