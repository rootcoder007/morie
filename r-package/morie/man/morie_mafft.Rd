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
\item{sequences}{See Usage.}
\item{method}{Defaults to \code{"FFT-NS-2"}.}
\item{seq_type}{Defaults to \code{NULL}.}
\item{s_a}{Defaults to \code{0.06}.}
\item{s_op}{Defaults to \code{2.4}.}
\item{matrix}{Defaults to \code{"normalized"}.}
\item{max_iterate}{Defaults to \code{16L}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{mafft_alignment}.
}
\keyword{internal}
