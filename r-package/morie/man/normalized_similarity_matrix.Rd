% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{normalized_similarity_matrix}
\alias{normalized_similarity_matrix}
\title{normalized_similarity_matrix}
\usage{
normalized_similarity_matrix(raw_matrix = NULL, freqs = NULL, s_a = 0.06, seq_type = "aa", 
    mode = "normalized", default = "jtt200") 
}
\arguments{
\item{raw_matrix}{The body requires: mafft: raw_matrix has no signal (average1 equals average2).}
\item{freqs}{Optional; may be \code{NULL}. A vector; indexed elementwise.}
\item{s_a}{Numeric; combined arithmetically in the body. Defaults to \code{0.06}.}
\item{seq_type}{Compared against \code{"nt"}. Defaults to \code{"aa"}.}
\item{mode}{Compared against \code{"all_positive"}. Defaults to \code{"normalized"}.}
\item{default}{One of \code{"grantham"}, \code{"jtt200"}. Defaults to \code{"jtt200"}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{matrix}, \code{s_a}, \code{alphabet}, \code{average1}, \code{average2}, \code{freqs}, \code{mode}.
}
\keyword{internal}
\examples{
normalized_similarity_matrix()
}
