% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{generate_dek}
\alias{generate_dek}
\title{Derive a distinct DEK per record}
\usage{
generate_dek(master_seed, record_id, salt = NULL) 
}
\description{
Derive a distinct DEK per record
}
\arguments{
\item{master_seed}{Argument \code{master_seed}; see Usage.}
\item{record_id}{Argument \code{record_id}; see Usage.}
\item{salt}{Argument \code{salt}; see Usage.}
}
\value{
A list with \code{dek}, \code{dek_hex}, \code{record_id}, \code{note}.
}
\keyword{internal}
\examples{
g <- generate_dek(as.raw(1:16), "record-42")
c(length(g$dek), nchar(g$dek_hex))
}
