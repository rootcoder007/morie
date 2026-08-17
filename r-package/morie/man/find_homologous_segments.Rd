% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{find_homologous_segments}
\alias{find_homologous_segments}
\title{find_homologous_segments}
\usage{
find_homologous_segments(group1, group2, scoring, weights1 = NULL, weights2 = NULL, seq_type = "aa", window = 30L, n_peaks = 20L, threshold = 0.7, max_len = 150L, corr_method = "fft")
}
\arguments{
\item{group1}{Passed to \code{unlist}.}
\item{group2}{Passed to \code{unlist}.}
\item{scoring}{A list; the body reads \code{$matrix} from it.}
\item{weights1}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
\item{weights2}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
\item{seq_type}{Passed to \code{correlation}. Defaults to \code{"aa"}.}
\item{window}{Numeric; combined arithmetically in the body. Defaults to \code{30L}.}
\item{n_peaks}{The body requires: mafft: window, n_peaks and max_len must be positive. Defaults to \code{20L}.}
\item{threshold}{Passed to \code{>}. Defaults to \code{0.7}.}
\item{max_len}{Numeric; combined arithmetically in the body. Defaults to \code{150L}.}
\item{corr_method}{Passed to \code{correlation}. Defaults to \code{"fft"}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
scoring <- list(A_A = 1, C_C = 1, G_G = 1, T_T = 1)
g1 <- c("ACGTACGTAC")
g2 <- c("ACGTACGTAC")
r <- find_homologous_segments(g1, g2, scoring, seq_type = "dna",
                              window = 5L)
is.list(r)
}
