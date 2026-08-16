% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_forensics_flatten_nist}
\alias{.morie_forensics_flatten_nist}
\title{Internal: pull morie\\'s documented columns out of one NIST RDS record}
\usage{
.morie_forensics_flatten_nist(rec)
}
\arguments{
\item{rec}{A list; the body reads \code{$@id}, \code{$byteSize}, \code{$description}, \code{$ediid}, \code{$identifier}, \code{$issued}, \code{$keyword}, \code{$landing_page}, \code{$landingPage}, \code{$license}, \code{$modified}, \code{$publisher}, \code{$rights}, \code{$size}, \code{$theme}, \code{$title} from it.}
}
\description{
A step of the ingest_forensics implementation. Called by \code{morie_ingest_forensics_nist_rds}. See the file header for the source the module follows.
}
\value{
A list with \code{dataset_id}, \code{title}, \code{description}, \code{publisher}, \code{issued}, \code{modified}, \code{keyword}, \code{landing_page}, \code{size_bytes}, \code{license}.
}
\keyword{internal}
\examples{
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
.morie_forensics_flatten_nist(D)
}
