% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mafft_default_raw}
\alias{mafft_default_raw}
\title{mafft_default_raw}
\usage{
mafft_default_raw(seq_type, which = "jtt200") 
}
\arguments{
\item{seq_type}{Compared against \code{"nt"}.}
\item{which}{Compared against \code{"grantham"}. Defaults to \code{"jtt200"}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{M}, \code{f}.
}
\keyword{internal}
\examples{
mafft_default_raw(seq_type = 5L)
}
