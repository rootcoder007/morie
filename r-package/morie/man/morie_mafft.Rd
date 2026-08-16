% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_mafft}
\alias{morie_mafft}
\title{morie_mafft}
\usage{
morie_mafft(sequences, method = "FFT-NS-2", seq_type = NULL, s_a = 0.06, s_op = 2.4, matrix = "normalized", max_iterate = 16L)
}
\arguments{
\item{sequences}{Passed to \code{mafft_alignment}.}
\item{method}{Passed to \code{mafft_alignment}. Defaults to \code{"FFT-NS-2"}.}
\item{seq_type}{Passed to \code{mafft_alignment}.}
\item{s_a}{Passed to \code{mafft_alignment}. Defaults to \code{0.06}.}
\item{s_op}{Passed to \code{mafft_alignment}. Defaults to \code{2.4}.}
\item{matrix}{Passed to \code{mafft_alignment}. Defaults to \code{"normalized"}.}
\item{max_iterate}{Passed to \code{mafft_alignment}. Defaults to \code{16L}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{mafft_alignment}.
}
\keyword{internal}
\examples{
r <- morie_mafft(c("MKVLATLLPLAA", "MKVLATLLPLAG", "MRVLATGLPLAA"),
                 method = "FFT-NS-2")
str(r, max.level = 1)
}
