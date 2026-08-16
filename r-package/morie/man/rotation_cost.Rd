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
\value{
A list with \code{single_key_bytes}, \code{envelope_kek_bytes}, \code{ratio}, \code{records_touched_single}, \code{records_touched_envelope}, \code{note}.
}
\keyword{internal}
