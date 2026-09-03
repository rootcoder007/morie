% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{rotation_cost}
\alias{rotation_cost}
\title{Cost model: single-key vs envelope rotation}
\usage{
rotation_cost(n_records, mean_record_bytes, dek_bytes = 32L) 
}
\description{
Cost model: single-key vs envelope rotation
}
\arguments{
\item{n_records}{Argument \code{n_records}; see Usage.}
\item{mean_record_bytes}{Argument \code{mean_record_bytes}; see Usage.}
\item{dek_bytes}{Argument \code{dek_bytes}; see Usage.}
}
\value{
A list with \code{single_key_bytes}, \code{envelope_kek_bytes}, \code{ratio}, \code{records_touched_single}, \code{records_touched_envelope}, \code{note}.
}
\keyword{internal}
\examples{
if (morie_crypto_sodium_available()) {
  rotation_cost(n_records = 5L, mean_record_bytes = 5L)
}
}
