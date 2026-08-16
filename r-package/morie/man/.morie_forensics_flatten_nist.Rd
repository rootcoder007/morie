% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_forensics_flatten_nist}
\alias{.morie_forensics_flatten_nist}
\title{Internal: pull morie\\\'s documented columns out of one NIST RDS record}
\usage{
.morie_forensics_flatten_nist(rec)
}
\arguments{
\item{rec}{See Usage.}
}
\description{
Part of the ingest_forensics implementation; see the file header for the source it follows.
}
\value{
A list with \code{dataset_id}, \code{title}, \code{description}, \code{publisher}, \code{issued}, \code{modified}, \code{keyword}, \code{landing_page}, \code{size_bytes}, \code{license}.
}
\keyword{internal}
