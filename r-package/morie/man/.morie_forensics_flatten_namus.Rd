% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_forensics_flatten_namus}
\alias{.morie_forensics_flatten_namus}
\title{Internal: pull morie\\\'s documented columns out of one NamUs record}
\usage{
.morie_forensics_flatten_namus(rec)
}
\arguments{
\item{rec}{See Usage.}
}
\description{
Part of the ingest_forensics implementation; see the file header for the source it follows.
}
\value{
A list with \code{case_number}, \code{state}, \code{county}, \code{dlc_date}, \code{sex}, \code{race}, \code{age_min}, \code{age_max}, \code{height_cm_min}, \code{height_cm_max}, \code{weight_kg_min}, \code{weight_kg_max}, \code{first_name}, \code{last_name}, \code{city}, \code{circumstances}.
}
\keyword{internal}
