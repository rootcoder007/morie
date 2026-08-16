% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_forensics_rows_to_df}
\alias{.morie_forensics_rows_to_df}
\title{Internal: rbind a list of named-list rows into a data.frame,}
\usage{
.morie_forensics_rows_to_df(rows, columns = NULL)
}
\arguments{
\item{rows}{A vector; its length is taken.}
\item{columns}{Optional; may be \code{NULL}. A vector; its length is taken.}
}
\description{
tolerating heterogeneous columns (missing -> NA).
}
\value{
The value of \code{do.call}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.morie_forensics_rows_to_df(V)
}
