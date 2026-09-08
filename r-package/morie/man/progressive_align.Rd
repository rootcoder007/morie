% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{progressive_align}
\alias{progressive_align}
\title{progressive_align}
\usage{
progressive_align(seqs, scoring, tree = NULL, seq_type = "aa", s_op = 2.4, use_fft = TRUE, 
    ...) 
}
\arguments{
\item{seqs}{A vector; its length is taken and its elements indexed.}
\item{scoring}{Passed to \code{find_homologous_segments}.}
\item{tree}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{seq_type}{Passed to \code{find_homologous_segments}. Defaults to \code{"aa"}.}
\item{s_op}{Passed to \code{group_align}. Defaults to \code{2.4}.}
\item{use_fft}{A flag; the body branches on it. Defaults to \code{TRUE}.}
\item{...}{Passed through.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
sc <- normalized_similarity_matrix(seq_type = "aa")
progressive_align(c("MKVLA", "MKVLG", "MRVLA"), sc, use_fft = FALSE)
}
